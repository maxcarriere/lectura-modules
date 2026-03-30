#!/usr/bin/env python3
"""Entraîne un modèle BiLSTM POS et l'exporte en ONNX — Script autonome.

Architecture : char-CNN (64d) + word embedding (128d) → BiLSTM 2 couches (128h)
               → émissions (18 tags).

Sortie : modèle ONNX + vocabulaire JSON.

Pré-requis :
    pip install torch onnx onnxruntime

Usage :
    python entrainer_bilstm.py \\
        --corpus donnees/pos_train_merged.conllu \\
        --dev donnees/pos_dev_merged.conllu

    # Personnaliser la sortie :
    python entrainer_bilstm.py \\
        --corpus donnees/pos_train_merged.conllu \\
        --output ../modele/pos_model_bilstm.onnx \\
        --vocab-output ../modele/pos_vocab_bilstm.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# ── Mapping UD → tags projet ────────────────────────────────────────────────

_UPOS_MAP: dict[str, str] = {
    "NOUN": "NOM", "PROPN": "NOM", "VERB": "VER", "AUX": "AUX",
    "ADJ": "ADJ", "ADV": "ADV", "ADP": "PRE", "CCONJ": "CON",
    "SCONJ": "CON", "INTJ": "INTJ", "NUM": "NOM", "SYM": "NOM", "X": "NOM",
}

_FEATURE_MAP: list[tuple[str, str, str, str]] = [
    ("DET", "Definite", "Def", "ART:def"),
    ("DET", "Definite", "Ind", "ART:ind"),
    ("DET", "PronType", "Art", "ART:def"),
    ("DET", "Poss", "Yes", "ADJ:pos"),
    ("DET", "PronType", "Dem", "ADJ:dem"),
    ("DET", "PronType", "Int", "ADJ:int"),
    ("PRON", "PronType", "Prs", "PRO:per"),
    ("PRON", "PronType", "Rel", "PRO:rel"),
    ("PRON", "PronType", "Dem", "PRO:dem"),
    ("PRON", "Poss", "Yes", "PRO:pos"),
    ("PRON", "PronType", "Int", "PRO:int"),
    ("PRON", "PronType", "Ind", "PRO:ind"),
]

_UPOS_FALLBACK: dict[str, str] = {"DET": "ART:ind", "PRON": "PRO:per"}
_IGNORE_UPOS = {"PUNCT", "SPACE"}


def _parse_ud_feats(feat_string: str) -> dict[str, str]:
    if not feat_string or feat_string == "_":
        return {}
    result: dict[str, str] = {}
    for part in feat_string.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def _ud_to_project_tag(upos: str, feats: dict[str, str] | None = None) -> str | None:
    if upos in _IGNORE_UPOS:
        return None
    if feats:
        for rule_upos, feat_key, feat_val, tag in _FEATURE_MAP:
            if upos == rule_upos and feats.get(feat_key) == feat_val:
                return tag
    if upos in _UPOS_FALLBACK:
        return _UPOS_FALLBACK[upos]
    return _UPOS_MAP.get(upos, "NOM")


# ── Parsing CoNLL-U ─────────────────────────────────────────────────────────

def parse_conllu(path: str) -> list[list[tuple[str, str, dict[str, str]]]]:
    """Parse un fichier CoNLL-U."""
    sentences: list[list[tuple[str, str, dict[str, str]]]] = []
    current: list[tuple[str, str, dict[str, str]]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue
            word = fields[1]
            upos = fields[3]
            feats = _parse_ud_feats(fields[5])
            current.append((word, upos, feats))

    if current:
        sentences.append(current)
    return sentences


def convert_sentences(
    raw_sentences: list[list[tuple[str, str, dict[str, str]]]],
) -> list[tuple[list[str], list[str]]]:
    """Convertit en (words, tags) avec tags projet."""
    result = []
    for sent in raw_sentences:
        words: list[str] = []
        tags: list[str] = []
        for word, upos, feats in sent:
            tag = _ud_to_project_tag(upos, feats)
            if tag is None:
                continue
            words.append(word)
            tags.append(tag)
        if words:
            result.append((words, tags))
    return result


# ── Construction du vocabulaire ──────────────────────────────────────────────

def build_vocabs(
    sentences: list[tuple[list[str], list[str]]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[int, str], list[str]]:
    """Construit word2idx, char2idx, tag2idx, idx2tag, tags."""
    words_set: set[str] = set()
    chars_set: set[str] = set()
    tags_set: set[str] = set()

    for words, tags in sentences:
        for w in words:
            words_set.add(w.lower())
            chars_set.update(w.lower())
        tags_set.update(tags)

    word2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for w in sorted(words_set):
        word2idx[w] = len(word2idx)

    char2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for ch in sorted(chars_set):
        char2idx[ch] = len(char2idx)

    tag2idx: dict[str, int] = {}
    tags_list = sorted(tags_set)
    for tag in tags_list:
        tag2idx[tag] = len(tag2idx)

    idx2tag = {v: k for k, v in tag2idx.items()}
    return word2idx, char2idx, tag2idx, idx2tag, tags_list


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    here = Path(__file__).parent
    default_corpus = here / "donnees" / "pos_train_merged.conllu"
    default_dev = here / "donnees" / "pos_dev_merged.conllu"

    # Fallback : chercher dans POS_CRF si les données ne sont pas ici
    if not default_corpus.exists():
        crf_donnees = here.parent.parent / "POS_CRF" / "entrainement" / "donnees"
        if (crf_donnees / "pos_train_merged.conllu").exists():
            default_corpus = crf_donnees / "pos_train_merged.conllu"
            default_dev = crf_donnees / "pos_dev_merged.conllu"

    parser = argparse.ArgumentParser(description="Entraîne un BiLSTM POS → ONNX")
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument("--dev", type=Path, default=default_dev)
    parser.add_argument("--output", default=str(here / ".." / "modele" / "pos_model_bilstm.onnx"))
    parser.add_argument("--vocab-output", default=str(here / ".." / "modele" / "pos_vocab_bilstm.json"))
    parser.add_argument("--word-embed-dim", type=int, default=128)
    parser.add_argument("--char-embed-dim", type=int, default=32)
    parser.add_argument("--char-cnn-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-word-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
    except ImportError:
        print("ERREUR : torch requis. pip install torch")
        sys.exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Chargement ──
    print(f"Parsing train : {args.corpus}")
    raw_train = parse_conllu(str(args.corpus))
    train_data = convert_sentences(raw_train)
    print(f"  {len(train_data)} phrases")

    dev_data: list[tuple[list[str], list[str]]] = []
    if args.dev.exists():
        raw_dev = parse_conllu(str(args.dev))
        dev_data = convert_sentences(raw_dev)
        print(f"  Dev : {len(dev_data)} phrases")

    word2idx, char2idx, tag2idx, idx2tag, tags_list = build_vocabs(train_data)
    n_words = len(word2idx)
    n_chars = len(char2idx)
    n_tags = len(tag2idx)
    max_wl = args.max_word_len
    print(f"Vocabulaire : {n_words} mots, {n_chars} chars, {n_tags} tags")

    # ── Dataset ──
    class POSDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            words, tags = self.data[idx]
            word_ids = torch.tensor(
                [word2idx.get(w.lower(), 1) for w in words], dtype=torch.long
            )
            tag_ids = torch.tensor(
                [tag2idx[t] for t in tags], dtype=torch.long
            )
            char_ids = []
            for w in words:
                w_low = w.lower()[:max_wl]
                ids = [char2idx.get(ch, 1) for ch in w_low]
                ids += [0] * (max_wl - len(ids))
                char_ids.append(ids)
            char_ids_t = torch.tensor(char_ids, dtype=torch.long)
            return word_ids, char_ids_t, tag_ids

    def collate_fn(batch):
        word_ids_list, char_ids_list, tag_ids_list = zip(*batch)
        lengths = torch.tensor([len(w) for w in word_ids_list], dtype=torch.long)
        word_ids = pad_sequence(word_ids_list, batch_first=True, padding_value=0)
        tag_ids = pad_sequence(tag_ids_list, batch_first=True, padding_value=-1)
        max_seq_len = word_ids.size(1)
        char_ids_padded = torch.zeros(len(batch), max_seq_len, max_wl, dtype=torch.long)
        for i, cids in enumerate(char_ids_list):
            char_ids_padded[i, : cids.size(0), :] = cids
        return word_ids, char_ids_padded, tag_ids, lengths

    train_loader = DataLoader(
        POSDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # ── Modèle ──
    class CharCNN(nn.Module):
        def __init__(self, n_chars, embed_dim, out_dim):
            super().__init__()
            self.embedding = nn.Embedding(n_chars, embed_dim, padding_idx=0)
            self.conv = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)

        def forward(self, x):
            emb = self.embedding(x)       # (B*S, W, E)
            emb = emb.transpose(1, 2)     # (B*S, E, W)
            conv_out = self.conv(emb)      # (B*S, out_dim, W)
            pooled, _ = conv_out.max(dim=2)  # (B*S, out_dim)
            return pooled

    class BiLSTMPOS(nn.Module):
        def __init__(
            self, n_words, n_chars, n_tags,
            word_embed_dim, char_embed_dim, char_cnn_dim,
            hidden_dim, num_layers, dropout, max_word_len,
        ):
            super().__init__()
            self.word_embedding = nn.Embedding(n_words, word_embed_dim, padding_idx=0)
            self.char_cnn = CharCNN(n_chars, char_embed_dim, char_cnn_dim)
            self.max_word_len = max_word_len

            input_dim = word_embed_dim + char_cnn_dim
            self.lstm = nn.LSTM(
                input_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, n_tags)

        def forward(self, word_ids, char_ids, lengths=None):
            batch_size, seq_len = word_ids.size()

            word_emb = self.word_embedding(word_ids)

            char_flat = char_ids.view(batch_size * seq_len, self.max_word_len)
            char_repr = self.char_cnn(char_flat)
            char_repr = char_repr.view(batch_size, seq_len, -1)

            combined = torch.cat([word_emb, char_repr], dim=2)
            combined = self.dropout(combined)

            if lengths is not None and not torch.onnx.is_in_onnx_export():
                packed = pack_padded_sequence(
                    combined, lengths.cpu().clamp(min=1),
                    batch_first=True, enforce_sorted=False,
                )
                lstm_out, _ = self.lstm(packed)
                lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            else:
                lstm_out, _ = self.lstm(combined)

            lstm_out = self.dropout(lstm_out)
            emissions = self.fc(lstm_out)
            return emissions

    model = BiLSTMPOS(
        n_words, n_chars, n_tags,
        args.word_embed_dim, args.char_embed_dim, args.char_cnn_dim,
        args.hidden_dim, args.num_layers, args.dropout, max_wl,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    # ── Entraînement ──
    print(f"\nEntraînement ({args.epochs} époques)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for word_ids, char_ids, tag_ids, lengths in train_loader:
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            tag_ids = tag_ids.to(device)
            lengths = lengths.to(device)

            emissions = model(word_ids, char_ids, lengths)
            emissions_flat = emissions.view(-1, n_tags)
            tags_flat = tag_ids.view(-1)
            loss = criterion(emissions_flat, tags_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f}")

    # ── Évaluation ──
    def evaluate_data(data, label):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for words, tags in data:
                word_ids = torch.tensor(
                    [[word2idx.get(w.lower(), 1) for w in words]], dtype=torch.long
                ).to(device)
                char_ids_list = []
                for w in words:
                    w_low = w.lower()[:max_wl]
                    ids = [char2idx.get(ch, 1) for ch in w_low]
                    ids += [0] * (max_wl - len(ids))
                    char_ids_list.append(ids)
                char_ids = torch.tensor([char_ids_list], dtype=torch.long).to(device)

                emissions = model(word_ids, char_ids)
                pred_ids = emissions[0].argmax(dim=-1).cpu().tolist()

                for pred_idx, gold_tag in zip(pred_ids[: len(tags)], tags):
                    total += 1
                    pred_tag = idx2tag.get(pred_idx, "")
                    if pred_tag == gold_tag:
                        correct += 1

        acc = correct / total if total else 0
        print(f"  {label} Accuracy : {acc:.1%} ({correct}/{total})")

    print("\n--- Évaluation sur train ---")
    evaluate_data(train_data[:2000], "Train")

    if dev_data:
        print("--- Évaluation sur dev ---")
        evaluate_data(dev_data, "Dev")

    # ── Export ONNX ──
    print(f"\nExport ONNX vers {args.output}...")
    model.eval()
    model.cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_words = torch.zeros(1, 5, dtype=torch.long)
    dummy_chars = torch.zeros(1, 5, max_wl, dtype=torch.long)

    try:
        import onnx

        torch.onnx.export(
            model,
            (dummy_words, dummy_chars),
            str(output_path),
            input_names=["word_ids", "char_ids"],
            output_names=["emissions"],
            dynamic_axes={
                "word_ids": {0: "batch", 1: "seq_len"},
                "char_ids": {0: "batch", 1: "seq_len"},
                "emissions": {0: "batch", 1: "seq_len"},
            },
            opset_version=14,
            dynamo=False,
        )

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path = str(output_path).replace(".onnx", "_int8.onnx")
            quantize_dynamic(str(output_path), quant_path, weight_type=QuantType.QInt8)
            quant_size = Path(quant_path).stat().st_size / 1024
            print(f"  Quantifié INT8 : {quant_path} ({quant_size:.0f} Ko)")
        except ImportError:
            pass

        size_kb = output_path.stat().st_size / 1024
        print(f"  Modèle ONNX : {output_path} ({size_kb:.0f} Ko)")

    except ImportError:
        print("  ERREUR : onnx requis. pip install onnx")

    # ── Vocabulaire ──
    vocab_path = Path(args.vocab_output)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "word2idx": word2idx,
                "char2idx": char2idx,
                "idx2tag": {str(k): v for k, v in idx2tag.items()},
                "tags": tags_list,
                "tag2idx": tag2idx,
                "pad_idx": 0,
                "unk_idx": 1,
                "max_word_len": max_wl,
                "use_lexicon": False,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"  Vocab : {vocab_path}")
    print("Terminé.")


if __name__ == "__main__":
    main()
