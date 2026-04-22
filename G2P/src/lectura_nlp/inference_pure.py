"""Inférence pur Python (zéro dépendance) pour le modèle unifié V2 (avec lex_features).

N'utilise que la bibliothèque standard Python. Charge les poids depuis JSON.
Plus lent que NumPy/ONNX mais totalement portable.

Usage :
    engine = PureInferenceEngine("modeles/unifie_weights.json", "modeles/unifie_vocab.json")
    result = engine.analyser(["Les", "enfants"])
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# POS keys for lex features (same order as training)
ALL_LEX_POS = [
    "ADJ", "ADJ:dem", "ADJ:ind", "ADJ:int", "ADJ:num", "ADJ:pos",
    "ADV", "ART:def", "ART:ind", "AUX", "CON", "INTJ",
    "NOM", "PRE",
    "PRO:dem", "PRO:ind", "PRO:int", "PRO:per", "PRO:pos", "PRO:rel",
    "VER",
]

LEX_FEATURE_DIM = len(ALL_LEX_POS) + 3  # 24


def _build_lex_features_pure(word: str, lexicon: dict[str, list[str]] | None) -> list[float]:
    """Construit le vecteur lex features (24d) pour un mot (pure Python)."""
    feats = [0.0] * LEX_FEATURE_DIM
    if lexicon is None:
        return feats
    candidates = lexicon.get(word.lower())
    if candidates is None:
        return feats
    feats[len(ALL_LEX_POS)] = 1.0
    feats[len(ALL_LEX_POS) + 1] = min(len(candidates) / 5.0, 1.0)
    feats[len(ALL_LEX_POS) + 2] = 1.0 if len(candidates) == 1 else 0.0
    for pos in candidates:
        if pos in ALL_LEX_POS:
            feats[ALL_LEX_POS.index(pos)] = 1.0
        else:
            for i, lex_pos in enumerate(ALL_LEX_POS):
                if lex_pos.startswith(pos + ":") or lex_pos == pos:
                    feats[i] = 1.0
    return feats


# ── Vector operations (pure Python lists) ──────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(len(a))]


def _mul(a: list[float], b: list[float]) -> list[float]:
    return [a[i] * b[i] for i in range(len(a))]


def _sigmoid(x: list[float]) -> list[float]:
    result = []
    for v in x:
        v = max(-500.0, min(500.0, v))
        result.append(1.0 / (1.0 + math.exp(-v)))
    return result


def _tanh(x: list[float]) -> list[float]:
    return [math.tanh(v) for v in x]


def _matvec(mat: list[list[float]], vec: list[float]) -> list[float]:
    """Matrix-vector multiply: mat (rows x cols) @ vec (cols,) → (rows,)"""
    return [_dot(row, vec) for row in mat]


def _argmax(x: list[float]) -> int:
    best_i = 0
    best_v = x[0]
    for i in range(1, len(x)):
        if x[i] > best_v:
            best_v = x[i]
            best_i = i
    return best_i


# ── LSTM Cell ──────────────────────────────────────────────────────

class _LSTMCell:
    def __init__(self, weight_ih: list[list[float]], weight_hh: list[list[float]],
                 bias_ih: list[float], bias_hh: list[float], hidden_size: int):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.hidden_size = hidden_size

    def forward(self, x: list[float], h: list[float], c: list[float]) -> tuple[list[float], list[float]]:
        hs = self.hidden_size
        gates = _add(
            _add(_matvec(self.weight_ih, x), self.bias_ih),
            _add(_matvec(self.weight_hh, h), self.bias_hh),
        )

        i_gate = _sigmoid(gates[:hs])
        f_gate = _sigmoid(gates[hs:2*hs])
        g_gate = _tanh(gates[2*hs:3*hs])
        o_gate = _sigmoid(gates[3*hs:])

        c_new = _add(_mul(f_gate, c), _mul(i_gate, g_gate))
        h_new = _mul(o_gate, _tanh(c_new))
        return h_new, c_new


class _BiLSTM:
    def __init__(self, cells_fwd: list[_LSTMCell], cells_bwd: list[_LSTMCell], hidden_size: int):
        self.cells_fwd = cells_fwd
        self.cells_bwd = cells_bwd
        self.num_layers = len(cells_fwd)
        self.hidden_size = hidden_size

    def forward(self, x_seq: list[list[float]]) -> list[list[float]]:
        seq_len = len(x_seq)
        current = x_seq

        for layer in range(self.num_layers):
            hs = self.hidden_size
            fwd_cell = self.cells_fwd[layer]
            bwd_cell = self.cells_bwd[layer]

            # Forward
            fwd_out = []
            h_f = [0.0] * hs
            c_f = [0.0] * hs
            for t in range(seq_len):
                h_f, c_f = fwd_cell.forward(current[t], h_f, c_f)
                fwd_out.append(h_f[:])

            # Backward
            bwd_out = [[0.0] * hs for _ in range(seq_len)]
            h_b = [0.0] * hs
            c_b = [0.0] * hs
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = bwd_cell.forward(current[t], h_b, c_b)
                bwd_out[t] = h_b[:]

            current = [fwd_out[t] + bwd_out[t] for t in range(seq_len)]

        return current


class PureInferenceEngine:
    """Inférence pur Python pour le modèle unifié V2."""

    def __init__(
        self,
        weights_path: str | Path,
        vocab_path: str | Path,
        lexicon_path: str | Path | None = None,
        lexicon: dict[str, list[str]] | None = None,
    ):
        logger.info("Loading G2P pure-Python model from %s", weights_path)
        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)

        self.config = data["config"]
        vocabs = data["vocabs"]
        self.char2idx = vocabs["char2idx"]
        self.idx2g2p = {int(v): k for k, v in vocabs["g2p_label2idx"].items()}
        self.idx2pos = {int(v): k for k, v in vocabs["pos2idx"].items()}
        self.idx2liaison = {int(v): k for k, v in vocabs["liaison2idx"].items()}
        self.idx2morpho = {}
        for feat, vocab in vocabs["morpho_vocabs"].items():
            self.idx2morpho[feat] = {int(v): k for k, v in vocab.items()}

        self.lex_feature_dim = self.config.get("lex_feature_dim", LEX_FEATURE_DIM)

        # Load lexicon
        if lexicon is not None:
            self.lexicon = lexicon
        elif lexicon_path is not None:
            with open(lexicon_path, encoding="utf-8") as f:
                self.lexicon = json.load(f)
            logger.info("Loaded lexicon: %d words", len(self.lexicon))
        else:
            self.lexicon = None

        # Load weights
        with open(weights_path, encoding="utf-8") as f:
            raw_weights = json.load(f)

        self.weights: dict[str, Any] = {}
        for name, w in raw_weights.items():
            shape = w["shape"]
            flat = w["data"]
            if len(shape) == 1:
                self.weights[name] = flat
            elif len(shape) == 2:
                rows, cols = shape
                mat = []
                for r in range(rows):
                    mat.append(flat[r * cols:(r + 1) * cols])
                self.weights[name] = mat
            else:
                self.weights[name] = flat

        self._build_model()

    def _get(self, name: str) -> Any:
        return self.weights[name]

    def _build_lstm(self, prefix: str, num_layers: int, hidden_size: int) -> _BiLSTM:
        cells_fwd = []
        cells_bwd = []
        for layer in range(num_layers):
            cells_fwd.append(_LSTMCell(
                self._get(f"{prefix}.weight_ih_l{layer}"),
                self._get(f"{prefix}.weight_hh_l{layer}"),
                self._get(f"{prefix}.bias_ih_l{layer}"),
                self._get(f"{prefix}.bias_hh_l{layer}"),
                hidden_size,
            ))
            cells_bwd.append(_LSTMCell(
                self._get(f"{prefix}.weight_ih_l{layer}_reverse"),
                self._get(f"{prefix}.weight_hh_l{layer}_reverse"),
                self._get(f"{prefix}.bias_ih_l{layer}_reverse"),
                self._get(f"{prefix}.bias_hh_l{layer}_reverse"),
                hidden_size,
            ))
        return _BiLSTM(cells_fwd, cells_bwd, hidden_size)

    def _build_model(self) -> None:
        # Embedding: list of lists [n_chars][embed_dim]
        self.char_embedding = self._get("char_embedding.weight")

        self.char_bilstm = self._build_lstm(
            "char_lstm",
            self.config["char_num_layers"],
            self.config["char_hidden_dim"],
        )

        self.g2p_weight = self._get("g2p_head.weight")
        self.g2p_bias = self._get("g2p_head.bias")

        # Lex features projection (V2)
        self.has_lex_proj = "lex_proj.weight" in self.weights
        if self.has_lex_proj:
            self.lex_proj_weight = self._get("lex_proj.weight")
            self.lex_proj_bias = self._get("lex_proj.bias")

        self.word_bilstm = self._build_lstm(
            "word_lstm",
            self.config.get("word_num_layers", 1),
            self.config["word_hidden_dim"],
        )

        self.pos_weight = self._get("pos_head.weight")
        self.pos_bias = self._get("pos_head.bias")

        # Liaison-specific BiLSTM (without lex features)
        self.liaison_hidden_dim = self.config.get("liaison_hidden_dim", 0)
        if self.liaison_hidden_dim > 0 and f"liaison_lstm.weight_ih_l0" in self.weights:
            self.liaison_bilstm = self._build_lstm(
                "liaison_lstm", 1, self.liaison_hidden_dim,
            )
        else:
            self.liaison_bilstm = None

        self.liaison_weight = self._get("liaison_head.weight")
        self.liaison_bias = self._get("liaison_head.bias")

        self.morpho_weights = {}
        self.morpho_biases = {}
        for feat_name in self.idx2morpho:
            self.morpho_weights[feat_name] = self._get(f"morpho_heads.{feat_name}.weight")
            self.morpho_biases[feat_name] = self._get(f"morpho_heads.{feat_name}.bias")

    def _linear(self, x: list[float], weight: list[list[float]], bias: list[float]) -> list[float]:
        return _add(_matvec(weight, x), bias)

    def _encode_sentence(
        self, tokens: list[str]
    ) -> tuple[list[int], list[int], list[int]]:
        chars: list[str] = ["<BOS>"]
        word_starts: list[int] = []
        word_ends: list[int] = []

        for w_idx, token in enumerate(tokens):
            if w_idx > 0:
                chars.append("<SEP>")
            word_starts.append(len(chars))
            for ch in token.lower():
                chars.append(ch)
            word_ends.append(len(chars) - 1)

        chars.append("<EOS>")

        char_ids = [self.char2idx.get(ch, 1) for ch in chars]
        return char_ids, word_starts, word_ends

    def analyser(self, tokens: list[str], *, use_lex: bool = True) -> dict[str, Any]:
        logger.debug("analyser() called with %s tokens", len(tokens))
        if not tokens:
            return {"tokens": [], "g2p": [], "pos": [], "liaison": [], "morpho": {}}

        char_ids, word_starts, word_ends = self._encode_sentence(tokens)

        # Character embedding
        emb = [self.char_embedding[cid][:] for cid in char_ids]

        # Character BiLSTM
        char_out = self.char_bilstm.forward(emb)

        # G2P
        n_words = len(tokens)
        g2p_results = []
        for w in range(n_words):
            ws = word_starts[w]
            we = word_ends[w]
            word_labels = []
            for i in range(ws, we + 1):
                logits = self._linear(char_out[i], self.g2p_weight, self.g2p_bias)
                pred = _argmax(logits)
                label = self.idx2g2p.get(pred, "_CONT")
                word_labels.append(label)
            ipa = "".join(lab for lab in word_labels if lab != "_CONT" and lab != "<PAD>")
            g2p_results.append(ipa)

        # Word representations
        char_hidden = self.config["char_hidden_dim"]
        word_repr = []
        for w in range(n_words):
            fwd_at_end = char_out[word_ends[w]][:char_hidden]
            bwd_at_start = char_out[word_starts[w]][char_hidden:]
            wr = fwd_at_end + bwd_at_start
            word_repr.append(wr)

        # Build word_repr with lex features for POS/Morpho path
        word_repr_for_lex = []
        for w in range(n_words):
            wr = word_repr[w][:]
            if use_lex and self.has_lex_proj:
                lex_feats = _build_lex_features_pure(tokens[w], self.lexicon)
                lex_proj = self._linear(lex_feats, self.lex_proj_weight, self.lex_proj_bias)
                wr = wr + lex_proj
            elif use_lex and self.lex_feature_dim > 0:
                lex_feats = _build_lex_features_pure(tokens[w], self.lexicon)
                wr = wr + lex_feats
            word_repr_for_lex.append(wr)

        # Word BiLSTM (POS + Morpho)
        word_out = self.word_bilstm.forward(word_repr_for_lex)

        # POS
        pos_results = []
        for w in range(n_words):
            logits = self._linear(word_out[w], self.pos_weight, self.pos_bias)
            pos_results.append(self.idx2pos.get(_argmax(logits), "NOM"))

        # Liaison (separate path without lex features)
        if self.liaison_bilstm is not None:
            liaison_out = self.liaison_bilstm.forward(word_repr)
            liaison_results = []
            for w in range(n_words):
                logits = self._linear(liaison_out[w], self.liaison_weight, self.liaison_bias)
                liaison_results.append(self.idx2liaison.get(_argmax(logits), "none"))
        else:
            liaison_results = []
            for w in range(n_words):
                logits = self._linear(word_out[w], self.liaison_weight, self.liaison_bias)
                liaison_results.append(self.idx2liaison.get(_argmax(logits), "none"))

        # Morpho
        morpho_results: dict[str, list[str]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            feat_results = []
            for w in range(n_words):
                logits = self._linear(
                    word_out[w], self.morpho_weights[feat_name], self.morpho_biases[feat_name]
                )
                feat_results.append(idx2label.get(_argmax(logits), "_"))
            morpho_results[feat_name] = feat_results

        return {
            "tokens": tokens,
            "g2p": g2p_results,
            "pos": pos_results,
            "liaison": liaison_results,
            "morpho": morpho_results,
        }
