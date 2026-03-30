#!/usr/bin/env python3
"""Entraine un modele Seq2Seq P2G et l'exporte en ONNX.

Architecture :
  Encoder : embedding IPA (128d) → BiLSTM 2 couches (256h/direction)
  Decoder : embedding ortho (128d) → LSTM 2 couches (512h) + attention → Linear

Entree  : chaine IPA (ex: "bɔ̃ʒuʁ")
Sortie  : orthographe francaise (ex: "bonjour")

Pre-requis :
    pip install torch onnx onnxruntime

Usage :
    python entrainer_seq2seq.py --train train_p2g.csv --output ../modele/p2g_seq2seq
    python entrainer_seq2seq.py --train train_p2g.csv --eval eval_p2g.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path


# ── Chargement ─────────────────────────────────────────────────


def load_pairs(path: Path, max_pairs: int = 0) -> list[tuple[str, str]]:
    """Charge les paires (phone, ortho) depuis un CSV."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phone = row.get("phone", "").strip()
            ortho = row.get("ortho", "").strip().lower()
            if not phone or not ortho:
                continue
            if " " in ortho or "'" in ortho or "\u2019" in ortho or "-" in ortho:
                continue
            key = (phone, ortho)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
                if max_pairs and len(pairs) >= max_pairs:
                    break
    return pairs


# ── Vocabulaires ────────────────────────────────────────────────


def build_vocabs(
    pairs: list[tuple[str, str]],
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    """Construit les vocabulaires source (IPA) et cible (ortho)."""
    src_chars: set[str] = set()
    tgt_chars: set[str] = set()
    for phone, ortho in pairs:
        src_chars.update(phone)
        tgt_chars.update(ortho)

    src_char2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for ch in sorted(src_chars):
        src_char2idx[ch] = len(src_char2idx)

    tgt_char2idx: dict[str, int] = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for ch in sorted(tgt_chars):
        tgt_char2idx[ch] = len(tgt_char2idx)

    tgt_idx2char = {v: k for k, v in tgt_char2idx.items()}
    return src_char2idx, tgt_char2idx, tgt_idx2char


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
    parser = argparse.ArgumentParser(description="Entrainement Seq2Seq P2G → ONNX")
    parser.add_argument("--train", required=True, help="CSV d'entrainement")
    parser.add_argument("--eval", default=None, help="CSV d'evaluation")
    parser.add_argument("--output", default="../modele/p2g_seq2seq",
                        help="Prefixe de sortie (sans extension)")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dec-hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-forcing", type=float, default=0.5,
                        help="Ratio de teacher forcing initial")
    parser.add_argument("--max-train", type=int, default=0)
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
    train_pairs = load_pairs(train_path, max_pairs=args.max_train)
    print(f"  {len(train_pairs)} paires train")

    eval_pairs: list[tuple[str, str]] = []
    if args.eval:
        eval_path = Path(args.eval)
        if eval_path.exists():
            eval_pairs = load_pairs(eval_path)
            print(f"  {len(eval_pairs)} paires eval")

    # ── Vocabulaires ──

    src_char2idx, tgt_char2idx, tgt_idx2char = build_vocabs(train_pairs)
    n_src = len(src_char2idx)
    n_tgt = len(tgt_char2idx)
    sos_idx = tgt_char2idx["<SOS>"]
    eos_idx = tgt_char2idx["<EOS>"]
    print(f"Vocabulaire : {n_src} src (IPA), {n_tgt} tgt (ortho)")

    # ── Dataset ──

    class P2GDataset(Dataset):
        def __init__(self, pairs: list[tuple[str, str]]):
            self.pairs = pairs

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> tuple:
            phone, ortho = self.pairs[idx]
            src_ids = torch.tensor(
                [src_char2idx.get(ch, 1) for ch in phone], dtype=torch.long,
            )
            tgt_ids = torch.tensor(
                [sos_idx] + [tgt_char2idx.get(ch, 3) for ch in ortho] + [eos_idx],
                dtype=torch.long,
            )
            return src_ids, tgt_ids

    def collate_fn(batch: list) -> tuple:
        srcs, tgts = zip(*batch)
        src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
        tgt_lens = torch.tensor([len(t) for t in tgts], dtype=torch.long)
        srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
        tgts_padded = pad_sequence(tgts, batch_first=True, padding_value=0)
        return srcs_padded, tgts_padded, src_lens, tgt_lens

    train_loader = DataLoader(
        P2GDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Modele ──

    class Encoder(nn.Module):
        def __init__(self, n_src, embed_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.embedding = nn.Embedding(n_src, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers=num_layers, batch_first=True, bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc_h = nn.Linear(hidden_dim * 2, args.dec_hidden_dim)
            self.fc_c = nn.Linear(hidden_dim * 2, args.dec_hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, src, src_lens=None):
            emb = self.dropout(self.embedding(src))
            if src_lens is not None and not torch.onnx.is_in_onnx_export():
                packed = pack_padded_sequence(
                    emb, src_lens.cpu().clamp(min=1),
                    batch_first=True, enforce_sorted=False,
                )
                out, (h, c) = self.lstm(packed)
                out, _ = pad_packed_sequence(out, batch_first=True)
            else:
                out, (h, c) = self.lstm(emb)
            # Concatener directions pour h et c
            h = torch.cat([h[-2], h[-1]], dim=1)
            c = torch.cat([c[-2], c[-1]], dim=1)
            h = torch.tanh(self.fc_h(h)).unsqueeze(0).repeat(args.num_layers, 1, 1)
            c = torch.tanh(self.fc_c(c)).unsqueeze(0).repeat(args.num_layers, 1, 1)
            return out, h, c

    class Attention(nn.Module):
        def __init__(self, enc_dim, dec_dim):
            super().__init__()
            self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
            self.v = nn.Linear(dec_dim, 1, bias=False)

        def forward(self, decoder_hidden, encoder_outputs):
            seq_len = encoder_outputs.size(1)
            hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
            return torch.softmax(self.v(energy).squeeze(2), dim=1)

    class Decoder(nn.Module):
        def __init__(self, n_tgt, embed_dim, hidden_dim, enc_dim, num_layers, dropout):
            super().__init__()
            self.embedding = nn.Embedding(n_tgt, embed_dim, padding_idx=0)
            self.attention = Attention(enc_dim, hidden_dim)
            self.lstm = nn.LSTM(
                embed_dim + enc_dim, hidden_dim,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc_out = nn.Linear(hidden_dim + enc_dim + embed_dim, n_tgt)
            self.dropout = nn.Dropout(dropout)

        def forward(self, inp, h, c, encoder_outputs):
            emb = self.dropout(self.embedding(inp.unsqueeze(1)))
            attn_weights = self.attention(h[-1], encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            lstm_input = torch.cat([emb, context], dim=2)
            output, (h, c) = self.lstm(lstm_input, (h, c))
            prediction = self.fc_out(torch.cat([output, context, emb], dim=2).squeeze(1))
            return prediction, h, c

    enc_dim = args.hidden_dim * 2  # BiLSTM
    encoder = Encoder(n_src, args.embed_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
    decoder = Decoder(n_tgt, args.embed_dim, args.dec_hidden_dim, enc_dim, args.num_layers, args.dropout).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    n_params = sum(p.numel() for p in params)
    print(f"Parametres : {n_params:,}")

    # ── Entrainement ──

    print(f"\nEntrainement ({args.epochs} epoques)...")
    tf_ratio = args.teacher_forcing

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # Teacher forcing progressif
        current_tf = max(0.1, tf_ratio * (1.0 - epoch / args.epochs))

        for srcs, tgts, src_lens, tgt_lens in train_loader:
            srcs = srcs.to(device)
            tgts = tgts.to(device)
            src_lens = src_lens.to(device)
            batch_size = srcs.size(0)
            tgt_len = tgts.size(1)

            enc_out, h, c = encoder(srcs, src_lens)

            inp = tgts[:, 0]  # <SOS>
            loss = torch.tensor(0.0, device=device)

            for t in range(1, tgt_len):
                output, h, c = decoder(inp, h, c, enc_out)
                loss = loss + criterion(output, tgts[:, t])

                if random.random() < current_tf:
                    inp = tgts[:, t]
                else:
                    inp = output.argmax(dim=1)

            loss = loss / (tgt_len - 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | tf={current_tf:.2f} | {elapsed:.1f}s")

    # ── Evaluation ──

    if eval_pairs:
        max_eval = min(len(eval_pairs), 5000)
        eval_sample = (
            random.sample(eval_pairs, max_eval) if len(eval_pairs) > max_eval
            else eval_pairs
        )
        print(f"\n--- Evaluation ({max_eval} mots) ---")
        encoder.eval()
        decoder.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for phone, gold_ortho in eval_sample:
                src_ids = torch.tensor(
                    [[src_char2idx.get(ch, 1) for ch in phone]], dtype=torch.long,
                ).to(device)
                enc_out, h, c = encoder(src_ids)
                inp = torch.tensor([sos_idx], dtype=torch.long, device=device)
                pred_chars: list[str] = []

                for _ in range(50):
                    output, h, c = decoder(inp, h, c, enc_out)
                    idx = output.argmax(dim=1).item()
                    if idx == eos_idx:
                        break
                    ch = tgt_idx2char.get(idx, "")
                    if ch not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>"):
                        pred_chars.append(ch)
                    inp = torch.tensor([idx], dtype=torch.long, device=device)

                pred_ortho = "".join(pred_chars)
                total += 1
                if pred_ortho == gold_ortho:
                    correct += 1

        word_acc = correct / total if total else 0
        print(f"  Precision mot : {word_acc:.1%} ({correct}/{total})")

    # ── Export ONNX ──

    print(f"\nExport ONNX vers {args.output}...")
    encoder.eval()
    decoder.eval()
    encoder.cpu()
    decoder.cpu()

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Export encoder
    enc_path = f"{output_prefix}_encoder.onnx"
    dummy_src = torch.zeros(1, 10, dtype=torch.long)
    try:
        import onnx

        torch.onnx.export(
            encoder, (dummy_src,),
            enc_path,
            input_names=["src"],
            output_names=["encoder_outputs", "h", "c"],
            dynamic_axes={"src": {0: "batch", 1: "seq_len"},
                          "encoder_outputs": {0: "batch", 1: "seq_len"}},
            opset_version=14,
            dynamo=False,
        )
        print(f"  Encoder : {enc_path}")

        # Export decoder
        dec_path = f"{output_prefix}_decoder.onnx"
        dummy_inp = torch.zeros(1, dtype=torch.long)
        dummy_h = torch.zeros(args.num_layers, 1, args.dec_hidden_dim)
        dummy_c = torch.zeros(args.num_layers, 1, args.dec_hidden_dim)
        dummy_enc = torch.zeros(1, 10, enc_dim)

        torch.onnx.export(
            decoder, (dummy_inp, dummy_h, dummy_c, dummy_enc),
            dec_path,
            input_names=["input_token", "h", "c", "encoder_outputs"],
            output_names=["logits", "h_out", "c_out"],
            dynamic_axes={"encoder_outputs": {0: "batch", 1: "enc_len"}},
            opset_version=14,
            dynamo=False,
        )
        print(f"  Decoder : {dec_path}")

        # Quantification INT8
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            for path in [enc_path, dec_path]:
                quant_path = path.replace(".onnx", "_int8.onnx")
                quantize_dynamic(path, quant_path, weight_type=QuantType.QInt8)
                quant_size = Path(quant_path).stat().st_size / 1024
                print(f"  Quantifie : {quant_path} ({quant_size:.0f} Ko)")
        except ImportError:
            print("  onnxruntime.quantization non disponible")

    except ImportError:
        print("  ERREUR : onnx requis. pip install onnx", file=sys.stderr)

    # ── Vocabulaire ──

    vocab_path = f"{output_prefix}_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "src_char2idx": src_char2idx,
                "tgt_char2idx": {k: v for k, v in tgt_char2idx.items()},
                "tgt_idx2char": {str(k): v for k, v in tgt_idx2char.items()},
                "sos_idx": sos_idx,
                "eos_idx": eos_idx,
                "max_len": 50,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"  Vocab : {vocab_path}")
    print("\nTermine.")


if __name__ == "__main__":
    main()
