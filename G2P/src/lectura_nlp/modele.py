"""Modèle unifié G2P+POS+Morpho+Liaison pour le français.

Architecture :
    Char embedding → Shared BiLSTM (char-level, phrase entière)
    ├── G2P Head (per-char classification)
    └── Word representations (fwd[last_char] || bwd[first_char])
        → Word BiLSTM
        ├── POS Head
        ├── Morpho Heads (Number, Gender, VerbForm, Mood, Tense, Person)
        └── Liaison Head

~950K paramètres → ~250 Ko INT8
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class UnifiedFrenchNLP(nn.Module):
    """Modèle unifié multi-tâche char-level sur la phrase entière."""

    def __init__(
        self,
        n_chars: int,
        n_g2p_labels: int,
        n_pos_labels: int,
        n_liaison_labels: int,
        morpho_label_sizes: dict[str, int],
        char_embed_dim: int = 64,
        char_hidden_dim: int = 160,
        char_num_layers: int = 2,
        word_hidden_dim: int = 192,
        word_num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_chars = n_chars
        self.char_hidden_dim = char_hidden_dim
        self.word_hidden_dim = word_hidden_dim

        # ── Shared character-level encoder ──
        self.char_embedding = nn.Embedding(n_chars, char_embed_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(
            char_embed_dim, char_hidden_dim,
            num_layers=char_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if char_num_layers > 1 else 0.0,
        )
        self.char_dropout = nn.Dropout(dropout)

        char_out_dim = char_hidden_dim * 2  # bidirectional

        # ── G2P Head (per-character) ──
        self.g2p_head = nn.Linear(char_out_dim, n_g2p_labels)

        # ── Word-level encoder ──
        # Word representations are formed by: fwd[last_char] || bwd[first_char]
        # This gives char_out_dim per word (char_hidden_dim * 2)
        self.word_lstm = nn.LSTM(
            char_out_dim, word_hidden_dim,
            num_layers=word_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if word_num_layers > 1 else 0.0,
        )
        self.word_dropout = nn.Dropout(dropout)

        word_out_dim = word_hidden_dim * 2  # bidirectional

        # ── POS Head ──
        self.pos_head = nn.Linear(word_out_dim, n_pos_labels)

        # ── Morpho Heads (factorized) ──
        self.morpho_heads = nn.ModuleDict()
        for feat_name, n_labels in morpho_label_sizes.items():
            self.morpho_heads[feat_name] = nn.Linear(word_out_dim, n_labels)

        # ── Liaison Head ──
        self.liaison_head = nn.Linear(word_out_dim, n_liaison_labels)

        # Store config for serialization
        self._config = {
            "n_chars": n_chars,
            "n_g2p_labels": n_g2p_labels,
            "n_pos_labels": n_pos_labels,
            "n_liaison_labels": n_liaison_labels,
            "morpho_label_sizes": morpho_label_sizes,
            "char_embed_dim": char_embed_dim,
            "char_hidden_dim": char_hidden_dim,
            "char_num_layers": char_num_layers,
            "word_hidden_dim": word_hidden_dim,
            "word_num_layers": word_num_layers,
            "dropout": dropout,
        }

    def forward(
        self,
        char_ids: torch.Tensor,
        char_lengths: torch.Tensor | None = None,
        word_starts: torch.Tensor | None = None,
        word_ends: torch.Tensor | None = None,
        word_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            char_ids: (batch, max_char_len) character indices
            char_lengths: (batch,) actual char sequence lengths
            word_starts: (batch, max_n_words) start positions in char sequence
            word_ends: (batch, max_n_words) end positions in char sequence
            word_lengths: (batch,) number of words per sentence

        Returns:
            Dict with keys: g2p_logits, pos_logits, liaison_logits,
            morpho_{feat}_logits for each morpho feature.
        """
        batch_size = char_ids.size(0)

        # ── Character-level encoding ──
        emb = self.char_embedding(char_ids)
        emb = self.char_dropout(emb)

        if char_lengths is not None and not torch.onnx.is_in_onnx_export():
            packed = pack_padded_sequence(
                emb, char_lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False,
            )
            char_out, _ = self.char_lstm(packed)
            char_out, _ = pad_packed_sequence(char_out, batch_first=True)
        else:
            char_out, _ = self.char_lstm(emb)

        char_out = self.char_dropout(char_out)

        # ── G2P logits (per character) ──
        g2p_logits = self.g2p_head(char_out)

        result: dict[str, torch.Tensor] = {"g2p_logits": g2p_logits}

        # If word boundaries not provided, skip word-level tasks
        if word_starts is None or word_ends is None or word_lengths is None:
            return result

        # ── Extract word representations ──
        # For each word: fwd_hidden[last_char] || bwd_hidden[first_char]
        # char_out shape: (batch, max_char_len, char_hidden_dim * 2)
        # fwd = char_out[:, :, :char_hidden_dim]
        # bwd = char_out[:, :, char_hidden_dim:]

        max_n_words = word_starts.size(1)
        device = char_ids.device

        fwd = char_out[:, :, :self.char_hidden_dim]
        bwd = char_out[:, :, self.char_hidden_dim:]

        # Gather word representations
        # word_ends[b, w] is the index of last char of word w in batch b
        # word_starts[b, w] is the index of first char of word w in batch b
        word_end_idx = word_ends.unsqueeze(-1).expand(-1, -1, self.char_hidden_dim)
        word_start_idx = word_starts.unsqueeze(-1).expand(-1, -1, self.char_hidden_dim)

        # Clamp to valid range
        max_seq = fwd.size(1) - 1
        word_end_idx = word_end_idx.clamp(0, max_seq)
        word_start_idx = word_start_idx.clamp(0, max_seq)

        fwd_at_end = torch.gather(fwd, 1, word_end_idx)   # (batch, max_words, hidden)
        bwd_at_start = torch.gather(bwd, 1, word_start_idx)  # (batch, max_words, hidden)

        word_repr = torch.cat([fwd_at_end, bwd_at_start], dim=-1)
        # word_repr: (batch, max_words, char_hidden_dim * 2)

        # ── Word-level BiLSTM ──
        if not torch.onnx.is_in_onnx_export():
            packed_words = pack_padded_sequence(
                word_repr, word_lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False,
            )
            word_out, _ = self.word_lstm(packed_words)
            word_out, _ = pad_packed_sequence(word_out, batch_first=True)
        else:
            word_out, _ = self.word_lstm(word_repr)

        word_out = self.word_dropout(word_out)
        # word_out: (batch, max_words, word_hidden_dim * 2)

        # ── Word-level heads ──
        result["pos_logits"] = self.pos_head(word_out)
        result["liaison_logits"] = self.liaison_head(word_out)

        for feat_name, head in self.morpho_heads.items():
            result[f"morpho_{feat_name}_logits"] = head(word_out)

        return result

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict) -> "UnifiedFrenchNLP":
        return cls(**config)


# ── Dataset pour l'entraînement multi-tâche ────────────────────────

class UnifiedDataset:
    """Dataset pour phrases avec toutes les annotations."""

    def __init__(
        self,
        sentences: list[dict],
        vocabs: dict,
        phone_to_graphs: dict[str, list[str]],
    ):
        from lectura_nlp.utils.aligneur import aligner
        from lectura_nlp.utils.g2p_labels import _CONT, labels_from_alignment, reconstruct_ipa

        self.vocabs = vocabs
        self.char2idx = vocabs["char2idx"]
        self.g2p_label2idx = vocabs["g2p_label2idx"]
        self.pos2idx = vocabs["pos2idx"]
        self.liaison2idx = vocabs["liaison2idx"]
        self.morpho_vocabs = vocabs["morpho_vocabs"]

        self.items: list[dict] = []

        for sent in sentences:
            tokens = sent["tokens"]
            # Skip sentences with no valid tokens
            if not tokens:
                continue

            # Build char sequence: <BOS> w1_chars <SEP> w2_chars ... <EOS>
            chars: list[str] = ["<BOS>"]
            g2p_labels: list[str] = ["<PAD>"]
            word_start_positions: list[int] = []
            word_end_positions: list[int] = []
            pos_labels: list[str] = []
            liaison_labels: list[str] = []
            morpho_labels: dict[str, list[str]] = {f: [] for f in self.morpho_vocabs}

            for w_idx, tok in enumerate(tokens):
                if w_idx > 0:
                    chars.append("<SEP>")
                    g2p_labels.append("<PAD>")

                form_lower = tok["form"].lower()
                word_start = len(chars)

                # G2P labels for this word
                word_g2p: list[str] = []
                if tok["phone"] and not any(c in form_lower for c in " '-\u2019"):
                    dec_ph, dec_gr, dec_spans, ok = aligner(
                        form_lower, tok["phone"], phone_to_graphs
                    )
                    if ok:
                        wlabels = labels_from_alignment(form_lower, dec_ph, dec_spans)
                        if len(wlabels) == len(form_lower):
                            recon = reconstruct_ipa(wlabels)
                            expected = "".join(dec_ph)
                            if recon == expected:
                                word_g2p = wlabels

                # If alignment failed, use PAD for G2P
                if not word_g2p:
                    word_g2p = ["<PAD>"] * len(form_lower)

                for ch_idx, ch in enumerate(form_lower):
                    chars.append(ch)
                    g2p_labels.append(word_g2p[ch_idx])

                word_end = len(chars) - 1
                word_start_positions.append(word_start)
                word_end_positions.append(word_end)

                pos_labels.append(tok["pos_tag"])
                liaison_labels.append(tok["liaison"])
                for feat_name in self.morpho_vocabs:
                    morpho_labels[feat_name].append(
                        tok["morpho"].get(feat_name, "_")
                    )

            chars.append("<EOS>")
            g2p_labels.append("<PAD>")

            self.items.append({
                "chars": chars,
                "g2p_labels": g2p_labels,
                "word_starts": word_start_positions,
                "word_ends": word_end_positions,
                "pos_labels": pos_labels,
                "liaison_labels": liaison_labels,
                "morpho_labels": morpho_labels,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]

        char_ids = torch.tensor(
            [self.char2idx.get(ch, 1) for ch in item["chars"]],
            dtype=torch.long,
        )
        g2p_ids = torch.tensor(
            [self.g2p_label2idx.get(lab, 0) for lab in item["g2p_labels"]],
            dtype=torch.long,
        )
        word_starts = torch.tensor(item["word_starts"], dtype=torch.long)
        word_ends = torch.tensor(item["word_ends"], dtype=torch.long)
        pos_ids = torch.tensor(
            [self.pos2idx.get(tag, 0) for tag in item["pos_labels"]],
            dtype=torch.long,
        )
        liaison_ids = torch.tensor(
            [self.liaison2idx.get(lab, 0) for lab in item["liaison_labels"]],
            dtype=torch.long,
        )

        morpho_ids = {}
        for feat_name, feat_vocab in self.morpho_vocabs.items():
            morpho_ids[feat_name] = torch.tensor(
                [feat_vocab.get(v, 0) for v in item["morpho_labels"][feat_name]],
                dtype=torch.long,
            )

        return {
            "char_ids": char_ids,
            "g2p_ids": g2p_ids,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "pos_ids": pos_ids,
            "liaison_ids": liaison_ids,
            "morpho_ids": morpho_ids,
        }


class LexiqueG2PDataset:
    """Dataset G2P pour mots isolés (pré-entraînement phase 1)."""

    def __init__(
        self,
        data: list[dict],
        char2idx: dict[str, int],
        g2p_label2idx: dict[str, int],
    ):
        self.data = data
        self.char2idx = char2idx
        self.g2p_label2idx = g2p_label2idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        word = item["word"]
        labels = item["labels"]

        char_ids = torch.tensor(
            [self.char2idx.get(ch, 1) for ch in word],
            dtype=torch.long,
        )
        label_ids = torch.tensor(
            [self.g2p_label2idx.get(lab, 0) for lab in labels],
            dtype=torch.long,
        )
        return char_ids, label_ids


# ── Collate functions ──────────────────────────────────────────────

def collate_lexique(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for LexiqueG2PDataset (phase 1)."""
    chars, labels = zip(*batch)
    lengths = torch.tensor([len(c) for c in chars], dtype=torch.long)
    chars_padded = nn.utils.rnn.pad_sequence(chars, batch_first=True, padding_value=0)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return chars_padded, labels_padded, lengths


def collate_unified(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate for UnifiedDataset (phase 2)."""
    batch_size = len(batch)

    # Pad character sequences
    char_ids = [b["char_ids"] for b in batch]
    g2p_ids = [b["g2p_ids"] for b in batch]
    char_lengths = torch.tensor([len(c) for c in char_ids], dtype=torch.long)
    char_ids_padded = nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=0)
    g2p_ids_padded = nn.utils.rnn.pad_sequence(g2p_ids, batch_first=True, padding_value=0)

    # Pad word-level sequences
    word_starts = [b["word_starts"] for b in batch]
    word_ends = [b["word_ends"] for b in batch]
    pos_ids = [b["pos_ids"] for b in batch]
    liaison_ids = [b["liaison_ids"] for b in batch]

    word_lengths = torch.tensor([len(ws) for ws in word_starts], dtype=torch.long)
    word_starts_padded = nn.utils.rnn.pad_sequence(word_starts, batch_first=True, padding_value=0)
    word_ends_padded = nn.utils.rnn.pad_sequence(word_ends, batch_first=True, padding_value=0)
    pos_ids_padded = nn.utils.rnn.pad_sequence(pos_ids, batch_first=True, padding_value=0)
    liaison_ids_padded = nn.utils.rnn.pad_sequence(liaison_ids, batch_first=True, padding_value=0)

    # Morpho features
    morpho_padded = {}
    feat_names = list(batch[0]["morpho_ids"].keys())
    for feat_name in feat_names:
        feat_tensors = [b["morpho_ids"][feat_name] for b in batch]
        morpho_padded[feat_name] = nn.utils.rnn.pad_sequence(
            feat_tensors, batch_first=True, padding_value=0
        )

    return {
        "char_ids": char_ids_padded,
        "char_lengths": char_lengths,
        "g2p_ids": g2p_ids_padded,
        "word_starts": word_starts_padded,
        "word_ends": word_ends_padded,
        "word_lengths": word_lengths,
        "pos_ids": pos_ids_padded,
        "liaison_ids": liaison_ids_padded,
        "morpho_ids": morpho_padded,
    }


# ── Multi-task loss ────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss for the unified model."""

    def __init__(
        self,
        w_g2p: float = 1.0,
        w_pos: float = 1.0,
        w_morpho: float = 0.5,
        w_liaison: float = 2.0,
        liaison_class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.w_g2p = w_g2p
        self.w_pos = w_pos
        self.w_morpho = w_morpho
        self.w_liaison = w_liaison
        self.label_smoothing = label_smoothing

        self.g2p_loss = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing,
        )
        self.pos_loss = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing,
        )
        self.morpho_losses = {}  # Created dynamically
        if liaison_class_weights is not None:
            self.liaison_loss = nn.CrossEntropyLoss(
                weight=liaison_class_weights, ignore_index=0,
                label_smoothing=label_smoothing,
            )
        else:
            self.liaison_loss = nn.CrossEntropyLoss(
                ignore_index=0, label_smoothing=label_smoothing,
            )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task loss.

        Returns:
            (total_loss, loss_dict) with individual losses for logging.
        """
        losses = {}

        # G2P loss (per-character)
        g2p_logits = outputs["g2p_logits"]
        g2p_targets = targets["g2p_ids"]
        g2p_l = self.g2p_loss(
            g2p_logits.reshape(-1, g2p_logits.size(-1)),
            g2p_targets.reshape(-1),
        )
        losses["g2p"] = g2p_l.item()

        total = self.w_g2p * g2p_l

        # POS loss
        if "pos_logits" in outputs:
            pos_logits = outputs["pos_logits"]
            pos_targets = targets["pos_ids"]
            pos_l = self.pos_loss(
                pos_logits.reshape(-1, pos_logits.size(-1)),
                pos_targets.reshape(-1),
            )
            losses["pos"] = pos_l.item()
            total = total + self.w_pos * pos_l

        # Liaison loss
        if "liaison_logits" in outputs:
            liaison_logits = outputs["liaison_logits"]
            liaison_targets = targets["liaison_ids"]
            liaison_l = self.liaison_loss(
                liaison_logits.reshape(-1, liaison_logits.size(-1)),
                liaison_targets.reshape(-1),
            )
            losses["liaison"] = liaison_l.item()
            total = total + self.w_liaison * liaison_l

        # Morpho losses
        morpho_total = 0.0
        for feat_name in targets.get("morpho_ids", {}):
            key = f"morpho_{feat_name}_logits"
            if key in outputs:
                feat_logits = outputs[key]
                feat_targets = targets["morpho_ids"][feat_name]
                if feat_name not in self.morpho_losses:
                    self.morpho_losses[feat_name] = nn.CrossEntropyLoss(
                        ignore_index=0, label_smoothing=self.label_smoothing,
                    ).to(feat_logits.device)
                feat_l = self.morpho_losses[feat_name](
                    feat_logits.reshape(-1, feat_logits.size(-1)),
                    feat_targets.reshape(-1),
                )
                losses[f"morpho_{feat_name}"] = feat_l.item()
                morpho_total += feat_l

        if morpho_total:
            total = total + self.w_morpho * morpho_total

        losses["total"] = total.item()
        return total, losses
