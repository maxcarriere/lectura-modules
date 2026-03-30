#!/usr/bin/env python3
"""Entraine un modele BiLSTM G2P et l'exporte en ONNX.

Architecture : char embedding (64d) → BiLSTM 2 couches (128h) → Linear → labels
Meme cadrage que le CRF : sequence-labeling caractere par caractere.

Pre-requis :
    pip install torch onnx onnxruntime

Usage :
    python entrainer_bilstm.py --train train_g2p.csv --eval eval_g2p.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_CONT = "_CONT"


def load_dataset(path: Path) -> list[tuple[str, list[str]]]:
    """Charge un CSV → [(word, labels)]."""
    data: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            labels_str = row.get("aligned_labels", "").strip()
            if not ortho or not labels_str:
                continue
            labels = labels_str.split(",")
            if len(labels) == len(ortho):
                data.append((ortho, labels))
    return data


def build_vocab(data):
    """Construit les vocabulaires char2idx et label2idx."""
    chars = set()
    labels = set()
    for word, word_labels in data:
        chars.update(word)
        labels.update(word_labels)

    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for ch in sorted(chars):
        char2idx[ch] = len(char2idx)

    label2idx = {"<PAD>": 0}
    for lab in sorted(labels):
        label2idx[lab] = len(label2idx)

    idx2label = {v: k for k, v in label2idx.items()}

    return char2idx, label2idx, idx2label


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement BiLSTM G2P")
    parser.add_argument("--train", required=True, help="CSV d'entrainement")
    parser.add_argument("--eval", default=None, help="CSV d'evaluation")
    parser.add_argument("--output", default="../modele/g2p_model_bilstm.onnx",
                        help="Fichier ONNX de sortie")
    parser.add_argument("--epochs", type=int, default=30, help="Nombre d'epoques")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="Taille LSTM")
    parser.add_argument("--embed", type=int, default=64, help="Taille embedding")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize le modele en INT8")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERREUR : PyTorch non installe.", file=sys.stderr)
        print("  pip install torch", file=sys.stderr)
        sys.exit(1)

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"ERREUR : {train_path} non trouve", file=sys.stderr)
        sys.exit(1)

    print("Chargement des donnees...")
    train_data = load_dataset(train_path)
    print(f"  {len(train_data)} mots charges")

    char2idx, label2idx, idx2label = build_vocab(train_data)
    n_chars = len(char2idx)
    n_labels = len(label2idx)
    print(f"  Chars : {n_chars}, Labels : {n_labels}")

    # Modele BiLSTM
    class BiLSTMG2P(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(n_chars, args.embed, padding_idx=0)
            self.lstm = nn.LSTM(
                args.embed, args.hidden, num_layers=2,
                bidirectional=True, batch_first=True,
            )
            self.fc = nn.Linear(args.hidden * 2, n_labels)

        def forward(self, x):
            emb = self.embed(x)
            out, _ = self.lstm(emb)
            return self.fc(out)

    model = BiLSTMG2P()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\nEntrainement ({args.epochs} epoques)...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for word, labels in train_data:
            char_ids = torch.tensor(
                [[char2idx.get(ch, 1) for ch in word]], dtype=torch.long,
            )
            label_ids = torch.tensor(
                [[label2idx.get(lab, 0) for lab in labels]], dtype=torch.long,
            )

            logits = model(char_ids)
            loss = criterion(logits.view(-1, n_labels), label_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{args.epochs} — loss: {avg_loss:.4f}")

    # Export ONNX
    print("\nExport ONNX...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.tensor([[1, 2, 3]], dtype=torch.long)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
    )
    print(f"  Modele : {output_path}")

    if args.quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            int8_path = output_path.with_name(
                output_path.stem + "_int8" + output_path.suffix,
            )
            quantize_dynamic(str(output_path), str(int8_path), weight_type=QuantType.QInt8)
            print(f"  Quantize : {int8_path}")
        except ImportError:
            print("  (quantization non disponible, onnxruntime.quantization manquant)")

    # Sauvegarder le vocabulaire
    vocab_path = output_path.with_name("g2p_vocab.json")
    vocab_data = {
        "char2idx": char2idx,
        "idx2label": {str(k): v for k, v in idx2label.items()},
        "pad_idx": 0,
        "unk_idx": 1,
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    print(f"  Vocabulaire : {vocab_path}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
