"""Inférence NumPy (sans PyTorch/ONNX) pour le modèle unifié V2 (avec lex_features).

Charge les poids depuis le JSON et effectue l'inférence en NumPy pur.
Dépendance : numpy

Usage :
    engine = NumpyInferenceEngine("modeles/unifie_weights.json", "modeles/unifie_vocab.json")
    result = engine.analyser(["Les", "enfants"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

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


def _build_lex_features(word: str, lexicon: dict[str, list[str]] | None) -> list[float]:
    """Construit le vecteur lex features (24d) pour un mot."""
    feats = [0.0] * LEX_FEATURE_DIM
    if lexicon is None:
        return feats
    candidates = lexicon.get(word.lower())
    if candidates is None:
        return feats
    feats[len(ALL_LEX_POS)] = 1.0  # known
    feats[len(ALL_LEX_POS) + 1] = min(len(candidates) / 5.0, 1.0)  # n_cands
    feats[len(ALL_LEX_POS) + 2] = 1.0 if len(candidates) == 1 else 0.0  # unambiguous
    for pos in candidates:
        if pos in ALL_LEX_POS:
            feats[ALL_LEX_POS.index(pos)] = 1.0
        else:
            for i, lex_pos in enumerate(ALL_LEX_POS):
                if lex_pos.startswith(pos + ":") or lex_pos == pos:
                    feats[i] = 1.0
    return feats


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class LSTMCell:
    """A single LSTM cell implemented in NumPy."""

    def __init__(self, weight_ih: np.ndarray, weight_hh: np.ndarray,
                 bias_ih: np.ndarray, bias_hh: np.ndarray):
        self.weight_ih = weight_ih  # (4*hidden, input)
        self.weight_hh = weight_hh  # (4*hidden, hidden)
        self.bias_ih = bias_ih      # (4*hidden,)
        self.bias_hh = bias_hh      # (4*hidden,)
        self.hidden_size = weight_hh.shape[1]

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x: (input_size,), h: (hidden,), c: (hidden,)"""
        gates = (self.weight_ih @ x + self.bias_ih +
                 self.weight_hh @ h + self.bias_hh)

        hs = self.hidden_size
        i = _sigmoid(gates[:hs])
        f = _sigmoid(gates[hs:2*hs])
        g = _tanh(gates[2*hs:3*hs])
        o = _sigmoid(gates[3*hs:])

        c_new = f * c + i * g
        h_new = o * _tanh(c_new)
        return h_new, c_new


class BiLSTM:
    """Bidirectional LSTM using NumPy."""

    def __init__(self, cells_fwd: list[LSTMCell], cells_bwd: list[LSTMCell]):
        self.cells_fwd = cells_fwd
        self.cells_bwd = cells_bwd
        self.num_layers = len(cells_fwd)
        self.hidden_size = cells_fwd[0].hidden_size

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """x_seq: (seq_len, input_size) → (seq_len, 2*hidden_size)"""
        seq_len = x_seq.shape[0]

        current_input = x_seq

        for layer in range(self.num_layers):
            hs = self.hidden_size
            fwd_cell = self.cells_fwd[layer]
            bwd_cell = self.cells_bwd[layer]

            # Forward pass
            fwd_out = np.zeros((seq_len, hs))
            h_f = np.zeros(hs)
            c_f = np.zeros(hs)
            for t in range(seq_len):
                h_f, c_f = fwd_cell.forward(current_input[t], h_f, c_f)
                fwd_out[t] = h_f

            # Backward pass
            bwd_out = np.zeros((seq_len, hs))
            h_b = np.zeros(hs)
            c_b = np.zeros(hs)
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = bwd_cell.forward(current_input[t], h_b, c_b)
                bwd_out[t] = h_b

            current_input = np.concatenate([fwd_out, bwd_out], axis=-1)

        return current_input


class NumpyInferenceEngine:
    """Inférence NumPy pour le modèle unifié V2."""

    def __init__(
        self,
        weights_path: str | Path,
        vocab_path: str | Path,
        lexicon_path: str | Path | None = None,
        lexicon: dict[str, list[str]] | None = None,
    ):
        logger.info("Loading G2P NumPy model from %s", weights_path)
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

        self.weights = {}
        for name, w in raw_weights.items():
            arr = np.array(w["data"], dtype=np.float32).reshape(w["shape"])
            self.weights[name] = arr

        # Build components
        self._build_model()

    def _get(self, name: str) -> np.ndarray:
        return self.weights[name]

    def _build_lstm(self, prefix: str, num_layers: int) -> BiLSTM:
        cells_fwd = []
        cells_bwd = []
        for layer in range(num_layers):
            # Forward
            cells_fwd.append(LSTMCell(
                self._get(f"{prefix}.weight_ih_l{layer}"),
                self._get(f"{prefix}.weight_hh_l{layer}"),
                self._get(f"{prefix}.bias_ih_l{layer}"),
                self._get(f"{prefix}.bias_hh_l{layer}"),
            ))
            # Backward
            cells_bwd.append(LSTMCell(
                self._get(f"{prefix}.weight_ih_l{layer}_reverse"),
                self._get(f"{prefix}.weight_hh_l{layer}_reverse"),
                self._get(f"{prefix}.bias_ih_l{layer}_reverse"),
                self._get(f"{prefix}.bias_hh_l{layer}_reverse"),
            ))
        return BiLSTM(cells_fwd, cells_bwd)

    def _build_model(self) -> None:
        self.char_embedding = self._get("char_embedding.weight")
        self.char_bilstm = self._build_lstm(
            "char_lstm", self.config["char_num_layers"]
        )

        # G2P head
        self.g2p_weight = self._get("g2p_head.weight")
        self.g2p_bias = self._get("g2p_head.bias")

        # Lex features projection (V2)
        self.has_lex_proj = "lex_proj.weight" in self.weights
        if self.has_lex_proj:
            self.lex_proj_weight = self._get("lex_proj.weight")
            self.lex_proj_bias = self._get("lex_proj.bias")

        # Word BiLSTM
        self.word_bilstm = self._build_lstm(
            "word_lstm", self.config.get("word_num_layers", 1)
        )

        # POS head
        self.pos_weight = self._get("pos_head.weight")
        self.pos_bias = self._get("pos_head.bias")

        # Liaison head
        self.liaison_weight = self._get("liaison_head.weight")
        self.liaison_bias = self._get("liaison_head.bias")

        # Morpho heads
        self.morpho_weights = {}
        self.morpho_biases = {}
        for feat_name in self.idx2morpho:
            self.morpho_weights[feat_name] = self._get(f"morpho_heads.{feat_name}.weight")
            self.morpho_biases[feat_name] = self._get(f"morpho_heads.{feat_name}.bias")

    def _linear(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return x @ weight.T + bias

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
        emb = self.char_embedding[char_ids]  # (seq_len, embed_dim)

        # Character BiLSTM
        char_out = self.char_bilstm.forward(emb)  # (seq_len, 2*char_hidden)

        # G2P logits
        g2p_logits = self._linear(char_out, self.g2p_weight, self.g2p_bias)

        # G2P results
        n_words = len(tokens)
        g2p_preds = g2p_logits.argmax(axis=-1)
        g2p_results = []
        for w in range(n_words):
            ws = word_starts[w]
            we = word_ends[w]
            word_labels = [self.idx2g2p.get(int(g2p_preds[i]), "_CONT")
                          for i in range(ws, we + 1)]
            ipa = "".join(lab for lab in word_labels if lab != "_CONT" and lab != "<PAD>")
            g2p_results.append(ipa)

        # Word representations: fwd[last_char] || bwd[first_char]
        char_hidden = self.config["char_hidden_dim"]
        fwd = char_out[:, :char_hidden]
        bwd = char_out[:, char_hidden:]

        word_repr = np.zeros((n_words, char_hidden * 2))
        for w in range(n_words):
            word_repr[w, :char_hidden] = fwd[word_ends[w]]
            word_repr[w, char_hidden:] = bwd[word_starts[w]]

        # Lex features
        if use_lex and self.has_lex_proj:
            lex_feats = np.array(
                [_build_lex_features(t, self.lexicon) for t in tokens],
                dtype=np.float32,
            )
            lex_proj = self._linear(lex_feats, self.lex_proj_weight, self.lex_proj_bias)
            word_repr = np.concatenate([word_repr, lex_proj], axis=-1)

        # Word BiLSTM
        word_out = self.word_bilstm.forward(word_repr)

        # POS
        pos_logits = self._linear(word_out, self.pos_weight, self.pos_bias)
        pos_preds = pos_logits.argmax(axis=-1)
        pos_results = [self.idx2pos.get(int(pos_preds[w]), "NOM") for w in range(n_words)]

        # Liaison
        liaison_logits = self._linear(word_out, self.liaison_weight, self.liaison_bias)
        liaison_preds = liaison_logits.argmax(axis=-1)
        liaison_results = [self.idx2liaison.get(int(liaison_preds[w]), "none") for w in range(n_words)]

        # Morpho
        morpho_results: dict[str, list[str]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            feat_logits = self._linear(
                word_out, self.morpho_weights[feat_name], self.morpho_biases[feat_name]
            )
            feat_preds = feat_logits.argmax(axis=-1)
            morpho_results[feat_name] = [
                idx2label.get(int(feat_preds[w]), "_") for w in range(n_words)
            ]

        return {
            "tokens": tokens,
            "g2p": g2p_results,
            "pos": pos_results,
            "liaison": liaison_results,
            "morpho": morpho_results,
        }
