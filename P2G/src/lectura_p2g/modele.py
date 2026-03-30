"""Modèle unifié P2G+POS+Morpho pour le français.

Architecture :
    IPA Char Embedding → Shared BiLSTM (char-level, phrase entière)
    ├── P2G Head (per-char classification : graphème par car. IPA)
    └── Word representations (fwd[last_ipa_char] || bwd[first_ipa_char])
        → Word BiLSTM
        ├── POS Head
        └── Morpho Heads (Number, Gender, VerbForm, Mood, Tense, Person)

Pas de tête Liaison (la liaison est dans l'input IPA, pas une sortie).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class UnifiedP2G(nn.Module):
    """Modèle unifié multi-tâche P2G char-level sur la phrase entière."""

    def __init__(
        self,
        n_chars: int,
        n_p2g_labels: int,
        n_pos_labels: int,
        morpho_label_sizes: dict[str, int],
        char_embed_dim: int = 64,
        char_hidden_dim: int = 160,
        char_num_layers: int = 2,
        word_hidden_dim: int = 192,
        word_num_layers: int = 1,
        dropout: float = 0.3,
        word_feedback: bool = False,
    ):
        super().__init__()
        self.n_chars = n_chars
        self.char_hidden_dim = char_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_feedback = word_feedback

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

        # ── Word-level encoder ──
        self.word_lstm = nn.LSTM(
            char_out_dim, word_hidden_dim,
            num_layers=word_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if word_num_layers > 1 else 0.0,
        )
        self.word_dropout = nn.Dropout(dropout)

        word_out_dim = word_hidden_dim * 2  # bidirectional

        # ── P2G Head (per-character) ──
        # With word_feedback: char_out + word_out broadcast to char positions
        p2g_input_dim = char_out_dim + word_out_dim if word_feedback else char_out_dim
        self.p2g_head = nn.Linear(p2g_input_dim, n_p2g_labels)

        # ── POS Head ──
        self.pos_head = nn.Linear(word_out_dim, n_pos_labels)

        # ── Morpho Heads (factorized) ──
        self.morpho_heads = nn.ModuleDict()
        for feat_name, n_labels in morpho_label_sizes.items():
            self.morpho_heads[feat_name] = nn.Linear(word_out_dim, n_labels)

        # Store config for serialization
        self._config = {
            "n_chars": n_chars,
            "n_p2g_labels": n_p2g_labels,
            "n_pos_labels": n_pos_labels,
            "morpho_label_sizes": morpho_label_sizes,
            "char_embed_dim": char_embed_dim,
            "char_hidden_dim": char_hidden_dim,
            "char_num_layers": char_num_layers,
            "word_hidden_dim": word_hidden_dim,
            "word_num_layers": word_num_layers,
            "dropout": dropout,
            "word_feedback": word_feedback,
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
            char_ids: (batch, max_char_len) IPA character indices
            char_lengths: (batch,) actual char sequence lengths
            word_starts: (batch, max_n_words) start positions in char sequence
            word_ends: (batch, max_n_words) end positions in char sequence
            word_lengths: (batch,) number of words per sentence

        Returns:
            Dict with keys: p2g_logits, pos_logits,
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

        result: dict[str, torch.Tensor] = {}

        # If word boundaries not provided, P2G only (pad word part with zeros)
        if word_starts is None or word_ends is None or word_lengths is None:
            if self.word_feedback:
                word_out_dim = self.word_hidden_dim * 2
                zeros = torch.zeros(
                    batch_size, char_out.size(1), word_out_dim,
                    device=char_out.device, dtype=char_out.dtype,
                )
                p2g_input = torch.cat([char_out, zeros], dim=-1)
                result["p2g_logits"] = self.p2g_head(p2g_input)
            else:
                result["p2g_logits"] = self.p2g_head(char_out)
            return result

        # ── Extract word representations ──
        max_n_words = word_starts.size(1)
        device = char_ids.device

        fwd = char_out[:, :, :self.char_hidden_dim]
        bwd = char_out[:, :, self.char_hidden_dim:]

        word_end_idx = word_ends.unsqueeze(-1).expand(-1, -1, self.char_hidden_dim)
        word_start_idx = word_starts.unsqueeze(-1).expand(-1, -1, self.char_hidden_dim)

        max_seq = fwd.size(1) - 1
        word_end_idx = word_end_idx.clamp(0, max_seq)
        word_start_idx = word_start_idx.clamp(0, max_seq)

        fwd_at_end = torch.gather(fwd, 1, word_end_idx)
        bwd_at_start = torch.gather(bwd, 1, word_start_idx)

        word_repr = torch.cat([fwd_at_end, bwd_at_start], dim=-1)

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

        # ── P2G logits (per character) ──
        if self.word_feedback:
            # Broadcast word representations to character positions (vectorized)
            max_char_len = char_out.size(1)
            word_out_dim = word_out.size(-1)

            # (1, max_chars, 1) position indices
            pos = torch.arange(max_char_len, device=device).view(1, -1, 1)
            # (batch, max_chars, max_words): True if char i is in word w
            in_word = (pos >= word_starts.unsqueeze(1)) & (pos <= word_ends.unsqueeze(1))
            # (batch, max_chars): word index for each char position
            char_to_word_idx = in_word.long().argmax(dim=-1)
            # Gather: (batch, max_words, dim) → (batch, max_chars, dim)
            word_at_char = torch.gather(
                word_out, 1,
                char_to_word_idx.unsqueeze(-1).expand(-1, -1, word_out_dim),
            )
            # Zero out chars not belonging to any word
            any_word = in_word.any(dim=-1).unsqueeze(-1)
            word_at_char = word_at_char * any_word.float()

            p2g_input = torch.cat([char_out, word_at_char], dim=-1)
            result["p2g_logits"] = self.p2g_head(p2g_input)
        else:
            result["p2g_logits"] = self.p2g_head(char_out)

        # ── Word-level heads ──
        result["pos_logits"] = self.pos_head(word_out)

        for feat_name, head in self.morpho_heads.items():
            result[f"morpho_{feat_name}_logits"] = head(word_out)

        return result

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict) -> "UnifiedP2G":
        return cls(**config)


# ── Dataset pour l'entraînement multi-tâche ────────────────────────

class UnifiedP2GDataset:
    """Dataset pour phrases avec toutes les annotations (P2G, POS, morpho)."""

    def __init__(
        self,
        sentences: list[dict],
        vocabs: dict,
        phone_to_graphs: dict[str, list[str]],
    ):
        from lectura_p2g.utils.aligneur import aligner
        from lectura_p2g.utils.p2g_labels import _CONT, labels_from_p2g_alignment, reconstruct_ortho

        self.vocabs = vocabs
        self.char2idx = vocabs["char2idx"]
        self.p2g_label2idx = vocabs["p2g_label2idx"]
        self.pos2idx = vocabs["pos2idx"]
        self.morpho_vocabs = vocabs["morpho_vocabs"]

        self.items: list[dict] = []

        for sent in sentences:
            tokens = sent["tokens"]
            if not tokens:
                continue

            # Build IPA char sequence: <BOS> ipa_chars_mot1 <SEP> ipa_chars_mot2 ... <EOS>
            chars: list[str] = ["<BOS>"]
            p2g_labels: list[str] = ["<PAD>"]
            word_start_positions: list[int] = []
            word_end_positions: list[int] = []
            pos_labels: list[str] = []
            morpho_labels: dict[str, list[str]] = {f: [] for f in self.morpho_vocabs}

            for w_idx, tok in enumerate(tokens):
                if w_idx > 0:
                    chars.append("<SEP>")
                    p2g_labels.append("<PAD>")

                form_lower = tok["form"].lower()
                phone = tok["phone"]
                word_start = len(chars)

                # P2G labels for this word
                word_p2g: list[str] = []
                if phone and not any(c in form_lower for c in " '-\u2019"):
                    dec_ph, dec_gr, dec_spans, ok = aligner(
                        form_lower, phone, phone_to_graphs
                    )
                    if ok:
                        wlabels = labels_from_p2g_alignment(phone, dec_ph, dec_gr)
                        if len(wlabels) == len(phone):
                            recon = reconstruct_ortho(wlabels)
                            expected = "".join(
                                g.replace("°", "").replace("²", "") for g in dec_gr
                            )
                            if recon == expected:
                                word_p2g = wlabels

                # If alignment failed, use PAD for P2G
                if not word_p2g:
                    word_p2g = ["<PAD>"] * len(phone) if phone else []

                # Add IPA characters to the sequence
                if phone:
                    for ch_idx, ch in enumerate(phone):
                        chars.append(ch)
                        if ch_idx < len(word_p2g):
                            p2g_labels.append(word_p2g[ch_idx])
                        else:
                            p2g_labels.append("<PAD>")
                else:
                    # No phone for this token — skip char-level
                    # but still need a word entry
                    chars.append("<UNK>")
                    p2g_labels.append("<PAD>")

                word_end = len(chars) - 1
                word_start_positions.append(word_start)
                word_end_positions.append(word_end)

                pos_labels.append(tok["pos_tag"])
                for feat_name in self.morpho_vocabs:
                    morpho_labels[feat_name].append(
                        tok["morpho"].get(feat_name, "_")
                    )

            chars.append("<EOS>")
            p2g_labels.append("<PAD>")

            self.items.append({
                "chars": chars,
                "p2g_labels": p2g_labels,
                "word_starts": word_start_positions,
                "word_ends": word_end_positions,
                "pos_labels": pos_labels,
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
        p2g_ids = torch.tensor(
            [self.p2g_label2idx.get(lab, 0) for lab in item["p2g_labels"]],
            dtype=torch.long,
        )
        word_starts = torch.tensor(item["word_starts"], dtype=torch.long)
        word_ends = torch.tensor(item["word_ends"], dtype=torch.long)
        pos_ids = torch.tensor(
            [self.pos2idx.get(tag, 0) for tag in item["pos_labels"]],
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
            "p2g_ids": p2g_ids,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "pos_ids": pos_ids,
            "morpho_ids": morpho_ids,
        }


class LexiqueP2GDataset:
    """Dataset P2G pour mots isolés (pré-entraînement phase 1).

    Input = caractères IPA du mot, labels = graphèmes par car. IPA.
    Format : {"ipa": "bɔ̃ʒuʁ", "labels": ["b", "on", "_CONT", "j", "ou", "r"]}
    """

    def __init__(
        self,
        data: list[dict],
        char2idx: dict[str, int],
        p2g_label2idx: dict[str, int],
    ):
        self.data = data
        self.char2idx = char2idx
        self.p2g_label2idx = p2g_label2idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        ipa = item["ipa"]
        labels = item["labels"]

        char_ids = torch.tensor(
            [self.char2idx.get(ch, 1) for ch in ipa],
            dtype=torch.long,
        )
        label_ids = torch.tensor(
            [self.p2g_label2idx.get(lab, 0) for lab in labels],
            dtype=torch.long,
        )
        return char_ids, label_ids


# ── Collate functions ──────────────────────────────────────────────

def collate_lexique(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for LexiqueP2GDataset (phase 1)."""
    chars, labels = zip(*batch)
    lengths = torch.tensor([len(c) for c in chars], dtype=torch.long)
    chars_padded = nn.utils.rnn.pad_sequence(chars, batch_first=True, padding_value=0)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return chars_padded, labels_padded, lengths


def collate_unified(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate for UnifiedP2GDataset (phase 2)."""
    batch_size = len(batch)

    # Pad character sequences
    char_ids = [b["char_ids"] for b in batch]
    p2g_ids = [b["p2g_ids"] for b in batch]
    char_lengths = torch.tensor([len(c) for c in char_ids], dtype=torch.long)
    char_ids_padded = nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=0)
    p2g_ids_padded = nn.utils.rnn.pad_sequence(p2g_ids, batch_first=True, padding_value=0)

    # Pad word-level sequences
    word_starts = [b["word_starts"] for b in batch]
    word_ends = [b["word_ends"] for b in batch]
    pos_ids = [b["pos_ids"] for b in batch]

    word_lengths = torch.tensor([len(ws) for ws in word_starts], dtype=torch.long)
    word_starts_padded = nn.utils.rnn.pad_sequence(word_starts, batch_first=True, padding_value=0)
    word_ends_padded = nn.utils.rnn.pad_sequence(word_ends, batch_first=True, padding_value=0)
    pos_ids_padded = nn.utils.rnn.pad_sequence(pos_ids, batch_first=True, padding_value=0)

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
        "p2g_ids": p2g_ids_padded,
        "word_starts": word_starts_padded,
        "word_ends": word_ends_padded,
        "word_lengths": word_lengths,
        "pos_ids": pos_ids_padded,
        "morpho_ids": morpho_padded,
    }


# ── Multi-task loss ────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss for the unified P2G model (no liaison)."""

    def __init__(
        self,
        w_p2g: float = 1.0,
        w_pos: float = 1.0,
        w_morpho: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.w_p2g = w_p2g
        self.w_pos = w_pos
        self.w_morpho = w_morpho
        self.label_smoothing = label_smoothing

        self.p2g_loss = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing,
        )
        self.pos_loss = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing,
        )
        self.morpho_losses = {}  # Created dynamically

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

        # P2G loss (per-character)
        p2g_logits = outputs["p2g_logits"]
        p2g_targets = targets["p2g_ids"]
        p2g_l = self.p2g_loss(
            p2g_logits.reshape(-1, p2g_logits.size(-1)),
            p2g_targets.reshape(-1),
        )
        losses["p2g"] = p2g_l.item()

        total = self.w_p2g * p2g_l

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
