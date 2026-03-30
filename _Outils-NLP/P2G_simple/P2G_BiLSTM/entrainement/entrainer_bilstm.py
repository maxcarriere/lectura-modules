#!/usr/bin/env python3
"""Entraine un modele BiLSTM P2G et l'exporte en ONNX.

Architecture : char embedding (64d) → BiLSTM 2 couches (128h) → Linear → labels
  - Entree  : caracteres IPA (ex: "bɔ̃ʒuʁ")
  - Labels  : grapheme par position (ex: "b", "on", _CONT, "j", "ou", "r")
  - Sortie  : concatenation des labels non-_CONT = "bonjour"

Pre-requis :
    pip install torch onnx onnxruntime

Usage :
    python entrainer_bilstm.py --train train_p2g.csv --output ../modele/p2g_bilstm.onnx
    python entrainer_bilstm.py --train train_p2g.csv --eval eval_p2g.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

_CONT = "_CONT"


# ── Chargement ─────────────────────────────────────────────────


def load_dataset(path: Path) -> list[tuple[str, list[str]]]:
    """Charge un CSV (ipa, ortho, aligned_labels) → [(ipa, labels)]."""
    data: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ipa = row.get("ipa", "").strip()
            labels_str = row.get("aligned_labels", "").strip()
            if not ipa or not labels_str:
                continue
            labels = labels_str.split(",")
            if len(labels) == len(ipa):
                data.append((ipa, labels))
    return data


def reconstruct_ortho(labels: list[str]) -> str:
    """Reconstruit l'orthographe depuis les labels P2G."""
    parts: list[str] = []
    for lab in labels:
        if lab == _CONT:
            continue
        clean = lab.replace("\u00b0", "").replace("\u00b2", "")
        parts.append(clean)
    return "".join(parts)


# ── Vocabulaires ────────────────────────────────────────────────


def build_vocabs(
    data: list[tuple[str, list[str]]],
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    """Construit les vocabulaires char→idx (entree) et label→idx (sortie)."""
    chars: set[str] = set()
    labels: set[str] = set()
    for phone_str, word_labels in data:
        chars.update(phone_str)
        labels.update(word_labels)

    char2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for ch in sorted(chars):
        char2idx[ch] = len(char2idx)

    label2idx: dict[str, int] = {"<PAD>": 0}
    for lab in sorted(labels):
        label2idx[lab] = len(label2idx)

    idx2label = {v: k for k, v in label2idx.items()}
    return char2idx, label2idx, idx2label


# ── Levenshtein ─────────────────────────────────────────────────


def _levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


# ── Main ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement BiLSTM P2G → ONNX")
    parser.add_argument("--train", required=True, help="CSV d'entrainement")
    parser.add_argument("--eval", default=None, help="CSV d'evaluation")
    parser.add_argument("--output", default="../modele/p2g_bilstm.onnx",
                        help="Fichier ONNX de sortie")
    parser.add_argument("--vocab-output", default="../modele/p2g_vocab.json",
                        help="Fichier vocab JSON de sortie")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import (
            pack_padded_sequence,
            pad_packed_sequence,
            pad_sequence,
        )
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("ERREUR : torch requis. pip install torch", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Chargement ──

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"ERREUR : {train_path} non trouve", file=sys.stderr)
        sys.exit(1)

    print("Chargement des donnees...")
    train_data = load_dataset(train_path)
    print(f"  {len(train_data)} mots charges")

    eval_data: list[tuple[str, list[str]]] = []
    if args.eval:
        eval_path = Path(args.eval)
        if eval_path.exists():
            eval_data = load_dataset(eval_path)
            print(f"  {len(eval_data)} mots eval")

    # ── Vocabulaires ──

    char2idx, label2idx, idx2label = build_vocabs(train_data)
    n_chars = len(char2idx)
    n_labels = len(label2idx)
    print(f"Vocabulaire : {n_chars} chars IPA, {n_labels} labels grapheme")

    # ── Dataset ──

    class P2GDataset(Dataset):
        def __init__(self, data: list[tuple[str, list[str]]]):
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> tuple:
            phone_str, labels = self.data[idx]
            char_ids = torch.tensor(
                [char2idx.get(ch, 1) for ch in phone_str], dtype=torch.long,
            )
            label_ids = torch.tensor(
                [label2idx.get(lab, 0) for lab in labels], dtype=torch.long,
            )
            return char_ids, label_ids

    def collate_fn(batch: list) -> tuple:
        chars, labels = zip(*batch)
        lengths = torch.tensor([len(c) for c in chars], dtype=torch.long)
        chars_padded = pad_sequence(chars, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
        return chars_padded, labels_padded, lengths

    train_loader = DataLoader(
        P2GDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Modele ──

    class BiLSTMP2G(nn.Module):
        def __init__(
            self, n_chars: int, n_labels: int,
            embed_dim: int, hidden_dim: int, num_layers: int, dropout: float,
        ):
            super().__init__()
            self.embedding = nn.Embedding(n_chars, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers=num_layers, batch_first=True, bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, n_labels)

        def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
            emb = self.embedding(x)
            emb = self.dropout(emb)
            if lengths is not None and not torch.onnx.is_in_onnx_export():
                packed = pack_padded_sequence(
                    emb, lengths.cpu().clamp(min=1),
                    batch_first=True, enforce_sorted=False,
                )
                lstm_out, _ = self.lstm(packed)
                lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            else:
                lstm_out, _ = self.lstm(emb)
            lstm_out = self.dropout(lstm_out)
            return self.fc(lstm_out)

    model = BiLSTMP2G(
        n_chars, n_labels,
        args.embed_dim, args.hidden_dim, args.num_layers, args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametres : {n_params:,}")

    # ── Entrainement ──

    print(f"\nEntrainement ({args.epochs} epoques)...")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for chars, labels, lengths in train_loader:
            chars = chars.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(chars, lengths)
            logits_flat = logits.view(-1, n_labels)
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | {elapsed:.1f}s")

    # ── Evaluation ──

    if eval_data:
        max_eval = min(len(eval_data), 5000)
        eval_sample = (
            random.sample(eval_data, max_eval) if len(eval_data) > max_eval
            else eval_data
        )
        print(f"\n--- Evaluation ({max_eval} mots) ---")
        model.eval()
        correct = 0
        total = 0
        cer_sum = 0.0

        with torch.no_grad():
            for phone_str, gold_labels in eval_sample:
                char_ids = torch.tensor(
                    [[char2idx.get(ch, 1) for ch in phone_str]], dtype=torch.long,
                ).to(device)
                logits = model(char_ids)
                pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
                pred_labels = [
                    idx2label.get(idx, _CONT) for idx in pred_ids[:len(phone_str)]
                ]

                pred_ortho = reconstruct_ortho(pred_labels)
                gold_ortho = reconstruct_ortho(gold_labels)

                total += 1
                if pred_ortho == gold_ortho:
                    correct += 1
                cer_sum += _levenshtein(pred_ortho, gold_ortho) / max(
                    len(gold_ortho), 1
                )

        word_acc = correct / total if total else 0
        cer = cer_sum / total if total else 0
        print(f"  Precision mot : {word_acc:.1%} ({correct}/{total})")
        print(f"  CER moyen     : {cer:.1%}")

    # ── Export ONNX ──

    print(f"\nExport ONNX vers {args.output}...")
    model.eval()
    model.cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(1, 10, dtype=torch.long)

    try:
        import onnx

        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["char_ids"],
            output_names=["logits"],
            dynamic_axes={
                "char_ids": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
            },
            opset_version=14,
            dynamo=False,
        )

        size_kb = output_path.stat().st_size / 1024
        print(f"  Modele ONNX : {output_path} ({size_kb:.0f} Ko)")

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quant_path = str(output_path).replace(".onnx", "_int8.onnx")
            quantize_dynamic(str(output_path), quant_path, weight_type=QuantType.QInt8)
            quant_size = Path(quant_path).stat().st_size / 1024
            print(f"  Quantifie INT8 : {quant_path} ({quant_size:.0f} Ko)")
        except ImportError:
            print("  onnxruntime.quantization non disponible")

    except ImportError:
        print("  ERREUR : onnx requis. pip install onnx", file=sys.stderr)

    # ── Vocabulaire ──

    vocab_path = Path(args.vocab_output)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "char2idx": char2idx,
                "idx2label": {str(k): v for k, v in idx2label.items()},
                "pad_idx": 0,
                "unk_idx": 1,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"  Vocab : {vocab_path}")
    print("\nTermine.")


if __name__ == "__main__":
    main()
