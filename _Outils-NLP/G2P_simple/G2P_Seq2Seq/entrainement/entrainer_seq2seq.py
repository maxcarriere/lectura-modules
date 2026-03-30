#!/usr/bin/env python3
"""Entraine un modele Seq2Seq G2P et l'exporte en ONNX.

Architecture :
  Encoder : embedding ortho (128d) → BiLSTM 2 couches (128h/direction)
  Decoder : embedding phone (64d) → LSTM 2 couches (256h) + attention → Linear
  ~1.5M parametres → ~6 Mo float32 → ~2.1 Mo INT8

Le dataset CSV contient des paires (ortho, phone). Les phonemes cible
sont tokenises par caractere de base + combining marks.

Pre-requis :
    pip install torch onnx onnxruntime

Usage :
    python entrainer_seq2seq.py --train train_g2p.csv --eval eval_g2p.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import unicodedata
from pathlib import Path


# ── Tokenisation IPA ──────────────────────────────────────────────


def iter_phonemes(ipa: str) -> list[str]:
    """Regroupe chaque caractere de base avec ses combining marks."""
    if not ipa:
        return []
    phonemes: list[str] = []
    current = ""
    for ch in ipa:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            current += ch
        else:
            if current:
                phonemes.append(current)
            current = ch
    if current:
        phonemes.append(current)
    return phonemes


# ── Chargement donnees ────────────────────────────────────────────


def load_pairs(path: Path, max_pairs: int = 0) -> list[tuple[str, str]]:
    """Charge les paires (ortho, phone) depuis un CSV G2P."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()
            if not ortho or not phone:
                continue
            if " " in ortho or "'" in ortho or "\u2019" in ortho or "-" in ortho:
                continue
            key = (ortho, phone)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
                if max_pairs and len(pairs) >= max_pairs:
                    break
    return pairs


def build_vocabs(
    pairs: list[tuple[str, str]],
) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    """Construit les vocabulaires source (ortho) et cible (phonemes)."""
    src_chars: set[str] = set()
    tgt_phones: set[str] = set()
    for ortho, phone in pairs:
        src_chars.update(ortho)
        tgt_phones.update(iter_phonemes(phone))

    src_char2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for ch in sorted(src_chars):
        src_char2idx[ch] = len(src_char2idx)

    tgt_phone2idx: dict[str, int] = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for ph in sorted(tgt_phones):
        tgt_phone2idx[ph] = len(tgt_phone2idx)

    tgt_idx2phone = {v: k for k, v in tgt_phone2idx.items()}
    return src_char2idx, tgt_phone2idx, tgt_idx2phone


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement Seq2Seq G2P")
    parser.add_argument("--train", required=True, help="CSV d'entrainement (ortho,phone)")
    parser.add_argument("--eval", default=None, help="CSV d'evaluation")
    parser.add_argument("--output", default="../modele/g2p_seq2seq",
                        help="Prefixe de sortie (defaut: ../modele/g2p_seq2seq)")
    parser.add_argument("--epochs", type=int, default=30, help="Nombre d'epoques")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Taille embedding source")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Taille LSTM (par direction)")
    parser.add_argument("--num-layers", type=int, default=2, help="Couches LSTM")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--teacher-forcing", type=float, default=0.5)
    parser.add_argument("--max-len", type=int, default=50, help="Longueur max decodage")
    parser.add_argument("--max-train", type=int, default=0, help="Limiter les paires train")
    parser.add_argument("--quantize", action="store_true", help="Quantize en INT8")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.nn.utils.rnn import (
            pack_padded_sequence,
            pad_packed_sequence,
            pad_sequence,
        )
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("ERREUR : PyTorch non installe.", file=sys.stderr)
        print("  pip install torch", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Donnees ──

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"ERREUR : {train_path} non trouve", file=sys.stderr)
        sys.exit(1)

    print("Chargement des donnees...")
    train_pairs = load_pairs(train_path, max_pairs=args.max_train)
    print(f"  {len(train_pairs)} paires")

    eval_pairs: list[tuple[str, str]] = []
    if args.eval:
        eval_path = Path(args.eval)
        if eval_path.exists():
            eval_pairs = load_pairs(eval_path)
            print(f"  {len(eval_pairs)} paires eval")

    src_char2idx, tgt_phone2idx, tgt_idx2phone = build_vocabs(train_pairs)
    n_src = len(src_char2idx)
    n_tgt = len(tgt_phone2idx)
    SOS_IDX = tgt_phone2idx["<SOS>"]
    EOS_IDX = tgt_phone2idx["<EOS>"]
    print(f"  Vocabulaire : {n_src} src, {n_tgt} tgt")

    # ── Dataset ──

    enc_out_dim = args.hidden_dim * 2
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    embed_dim = args.embed_dim
    dec_embed_dim = 64
    dropout_rate = args.dropout

    class G2PDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            ortho, phone = self.pairs[idx]
            src = torch.tensor(
                [src_char2idx.get(c, 1) for c in ortho], dtype=torch.long,
            )
            tgt = torch.tensor(
                [SOS_IDX]
                + [tgt_phone2idx.get(ph, 3) for ph in iter_phonemes(phone)]
                + [EOS_IDX],
                dtype=torch.long,
            )
            return src, tgt

    def collate_fn(batch):
        srcs, tgts = zip(*batch)
        src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=0)
        return src_padded, tgt_padded, src_lens

    train_loader = DataLoader(
        G2PDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Modele ──

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(n_src, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim,
                num_layers=num_layers, batch_first=True, bidirectional=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout_rate)
            self.h_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.c_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        def forward(self, x, lengths=None):
            emb = self.dropout(self.embedding(x))
            if lengths is not None and not torch.onnx.is_in_onnx_export():
                packed = pack_padded_sequence(
                    emb, lengths.cpu().clamp(min=1),
                    batch_first=True, enforce_sorted=False,
                )
                output, (h, c) = self.lstm(packed)
                output, _ = pad_packed_sequence(output, batch_first=True)
            else:
                output, (h, c) = self.lstm(emb)

            B = x.size(0)
            h = h.view(num_layers, 2, B, hidden_dim)
            h = h.permute(0, 2, 1, 3).contiguous().view(num_layers, B, -1)
            c = c.view(num_layers, 2, B, hidden_dim)
            c = c.permute(0, 2, 1, 3).contiguous().view(num_layers, B, -1)

            h = torch.tanh(self.h_proj(h))
            c = torch.tanh(self.c_proj(c))
            return output, h, c

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(enc_out_dim + enc_out_dim, enc_out_dim)
            self.v = nn.Linear(enc_out_dim, 1, bias=False)

        def forward(self, dec_hidden, enc_outputs):
            src_len = enc_outputs.size(1)
            hidden_exp = dec_hidden.unsqueeze(1).expand(-1, src_len, -1)
            energy = torch.tanh(
                self.attn(torch.cat([hidden_exp, enc_outputs], dim=-1)),
            )
            scores = self.v(energy).squeeze(-1)
            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
            return context, weights

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(n_tgt, dec_embed_dim, padding_idx=0)
            self.attention = Attention()
            self.lstm = nn.LSTM(
                dec_embed_dim + enc_out_dim, enc_out_dim,
                num_layers=num_layers, batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(enc_out_dim + enc_out_dim + dec_embed_dim, n_tgt)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, input_token, h, c, enc_outputs):
            emb = self.dropout(self.embedding(input_token.unsqueeze(1)))
            context, _ = self.attention(h[-1], enc_outputs)
            lstm_in = torch.cat([emb.squeeze(1), context], dim=-1).unsqueeze(1)
            output, (h_new, c_new) = self.lstm(lstm_in, (h, c))
            logits = self.fc(torch.cat(
                [output.squeeze(1), context, emb.squeeze(1)], dim=-1,
            ))
            return logits, h_new, c_new

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, src, tgt, src_lens, teacher_forcing_ratio=0.5):
            B = src.size(0)
            tgt_len = tgt.size(1)
            enc_out, h, c = self.encoder(src, src_lens)
            outputs = torch.zeros(B, tgt_len - 1, n_tgt, device=src.device)
            inp = tgt[:, 0]
            for t in range(1, tgt_len):
                pred, h, c = self.decoder(inp, h, c, enc_out)
                outputs[:, t - 1] = pred
                if random.random() < teacher_forcing_ratio:
                    inp = tgt[:, t]
                else:
                    inp = pred.argmax(dim=-1)
            return outputs

    model = Seq2Seq().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametres : {n_params:,}")

    # ── Entrainement ──

    print(f"\nEntrainement ({args.epochs} epoques)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for src, tgt, src_lens in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_lens = src_lens.to(device)

            output = model(src, tgt, src_lens, args.teacher_forcing)
            output_flat = output.view(-1, n_tgt)
            tgt_flat = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output_flat, tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} — loss: {avg_loss:.4f} ({elapsed:.1f}s)")

    # ── Export ONNX ──

    print("\nExport ONNX...")
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    try:
        import onnx

        # Encoder
        enc_path = str(output_base) + "_encoder.onnx"
        dummy_src = torch.zeros(1, 10, dtype=torch.long)

        torch.onnx.export(
            model.encoder,
            (dummy_src,),
            enc_path,
            input_names=["src"],
            output_names=["encoder_outputs", "h", "c"],
            dynamic_axes={
                "src": {0: "batch", 1: "src_len"},
                "encoder_outputs": {0: "batch", 1: "src_len"},
                "h": {1: "batch"},
                "c": {1: "batch"},
            },
            opset_version=14,
        )
        print(f"  Encoder : {enc_path}")

        # Decoder step
        dec_path = str(output_base) + "_decoder.onnx"
        dummy_token = torch.zeros(1, dtype=torch.long)
        dummy_h = torch.zeros(num_layers, 1, enc_out_dim)
        dummy_c = torch.zeros(num_layers, 1, enc_out_dim)
        dummy_enc_out = torch.zeros(1, 10, enc_out_dim)

        torch.onnx.export(
            model.decoder,
            (dummy_token, dummy_h, dummy_c, dummy_enc_out),
            dec_path,
            input_names=["input_token", "h", "c", "encoder_outputs"],
            output_names=["logits", "h_new", "c_new"],
            dynamic_axes={
                "input_token": {0: "batch"},
                "h": {1: "batch"},
                "c": {1: "batch"},
                "encoder_outputs": {0: "batch", 1: "src_len"},
                "logits": {0: "batch"},
                "h_new": {1: "batch"},
                "c_new": {1: "batch"},
            },
            opset_version=14,
        )
        print(f"  Decoder : {dec_path}")

        # Quantification INT8
        if args.quantize:
            try:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                for src_path in [enc_path, dec_path]:
                    q_path = src_path.replace(".onnx", "_int8.onnx")
                    quantize_dynamic(src_path, q_path, weight_type=QuantType.QInt8)
                    print(f"  Quantifie : {q_path}")
            except ImportError:
                print("  (quantization non disponible, onnxruntime.quantization manquant)")

    except ImportError:
        print("  ERREUR : onnx requis. pip install onnx")

    # Vocabulaire
    vocab_path = str(output_base) + "_vocab.json"
    vocab = {
        "type": "seq2seq",
        "char2idx": src_char2idx,
        "phone2idx": tgt_phone2idx,
        "idx2phone": {str(k): v for k, v in tgt_idx2phone.items()},
        "sos_idx": SOS_IDX,
        "eos_idx": EOS_IDX,
        "pad_idx": 0,
        "max_len": args.max_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    print(f"  Vocabulaire : {vocab_path}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
