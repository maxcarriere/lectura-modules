"""Inférence NumPy (sans PyTorch/ONNX) pour le modèle unifié P2G.

Charge les poids depuis le JSON et effectue l'inférence en NumPy pur.
Dépendance : numpy

Usage :
    engine = NumpyInferenceEngine("modeles/unifie_p2g_weights.json", "modeles/unifie_p2g_vocab.json")
    result = engine.analyser(["le", "ʃa"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lectura_p2g.utils.p2g_labels import reconstruct_ortho

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class LSTMCell:
    """A single LSTM cell implemented in NumPy."""

    def __init__(self, weight_ih: np.ndarray, weight_hh: np.ndarray,
                 bias_ih: np.ndarray, bias_hh: np.ndarray):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.hidden_size = weight_hh.shape[1]

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        seq_len = x_seq.shape[0]
        current_input = x_seq

        for layer in range(self.num_layers):
            hs = self.hidden_size
            fwd_cell = self.cells_fwd[layer]
            bwd_cell = self.cells_bwd[layer]

            fwd_out = np.zeros((seq_len, hs))
            h_f = np.zeros(hs)
            c_f = np.zeros(hs)
            for t in range(seq_len):
                h_f, c_f = fwd_cell.forward(current_input[t], h_f, c_f)
                fwd_out[t] = h_f

            bwd_out = np.zeros((seq_len, hs))
            h_b = np.zeros(hs)
            c_b = np.zeros(hs)
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = bwd_cell.forward(current_input[t], h_b, c_b)
                bwd_out[t] = h_b

            current_input = np.concatenate([fwd_out, bwd_out], axis=-1)

        return current_input


class NumpyInferenceEngine:
    """Inférence NumPy pour le modèle unifié P2G."""

    def __init__(self, weights_path: str | Path, vocab_path: str | Path):
        logger.info("Loading P2G NumPy model from %s", weights_path)
        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)

        self.config = data["config"]
        vocabs = data["vocabs"]
        self.char2idx = vocabs["char2idx"]
        self.idx2p2g = {int(v): k for k, v in vocabs["p2g_label2idx"].items()}
        self.idx2pos = {int(v): k for k, v in vocabs["pos2idx"].items()}
        self.idx2morpho = {}
        for feat, vocab in vocabs["morpho_vocabs"].items():
            self.idx2morpho[feat] = {int(v): k for k, v in vocab.items()}

        with open(weights_path, encoding="utf-8") as f:
            raw_weights = json.load(f)

        self.weights = {}
        for name, w in raw_weights.items():
            arr = np.array(w["data"], dtype=np.float32).reshape(w["shape"])
            self.weights[name] = arr

        self._build_model()

    def _get(self, name: str) -> np.ndarray:
        return self.weights[name]

    def _build_lstm(self, prefix: str, num_layers: int) -> BiLSTM:
        cells_fwd = []
        cells_bwd = []
        for layer in range(num_layers):
            cells_fwd.append(LSTMCell(
                self._get(f"{prefix}.weight_ih_l{layer}"),
                self._get(f"{prefix}.weight_hh_l{layer}"),
                self._get(f"{prefix}.bias_ih_l{layer}"),
                self._get(f"{prefix}.bias_hh_l{layer}"),
            ))
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

        self.p2g_weight = self._get("p2g_head.weight")
        self.p2g_bias = self._get("p2g_head.bias")

        self.word_bilstm = self._build_lstm(
            "word_lstm", self.config.get("word_num_layers", 1)
        )

        self.pos_weight = self._get("pos_head.weight")
        self.pos_bias = self._get("pos_head.bias")

        self.morpho_weights = {}
        self.morpho_biases = {}
        for feat_name in self.idx2morpho:
            self.morpho_weights[feat_name] = self._get(f"morpho_heads.{feat_name}.weight")
            self.morpho_biases[feat_name] = self._get(f"morpho_heads.{feat_name}.bias")

    def _linear(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return x @ weight.T + bias

    def _encode_sentence(
        self, ipa_words: list[str]
    ) -> tuple[list[int], list[int], list[int]]:
        chars: list[str] = ["<BOS>"]
        word_starts: list[int] = []
        word_ends: list[int] = []

        for w_idx, word in enumerate(ipa_words):
            if w_idx > 0:
                chars.append("<SEP>")
            word_starts.append(len(chars))
            for ch in word:
                chars.append(ch)
            word_ends.append(len(chars) - 1)

        chars.append("<EOS>")

        char_ids = [self.char2idx.get(ch, 1) for ch in chars]
        return char_ids, word_starts, word_ends

    def analyser(self, ipa_words: list[str]) -> dict[str, Any]:
        logger.debug("analyser() called with %s IPA words", len(ipa_words))
        if not ipa_words:
            return {"ipa_words": [], "ortho": [], "pos": [], "morpho": {}}

        char_ids, word_starts, word_ends = self._encode_sentence(ipa_words)
        n_words = len(ipa_words)

        # Character embedding
        emb = self.char_embedding[char_ids]

        # Character BiLSTM
        char_out = self.char_bilstm.forward(emb)

        # Word representations: fwd[last_char] || bwd[first_char]
        char_hidden = self.config["char_hidden_dim"]
        fwd = char_out[:, :char_hidden]
        bwd = char_out[:, char_hidden:]

        word_repr = np.zeros((n_words, char_hidden * 2))
        for w in range(n_words):
            word_repr[w, :char_hidden] = fwd[word_ends[w]]
            word_repr[w, char_hidden:] = bwd[word_starts[w]]

        # Word BiLSTM
        word_out = self.word_bilstm.forward(word_repr)

        # POS
        pos_logits = self._linear(word_out, self.pos_weight, self.pos_bias)
        pos_preds = pos_logits.argmax(axis=-1)
        pos_results = [self.idx2pos.get(int(pos_preds[w]), "NOM") for w in range(n_words)]

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

        # Word feedback: broadcast word representations to char positions
        word_feedback = np.zeros((len(char_ids), word_out.shape[1]))
        for w in range(n_words):
            ws = word_starts[w]
            we = word_ends[w]
            for i in range(ws, we + 1):
                word_feedback[i] = word_out[w]

        # P2G input: char_out (320d) || word_feedback (384d) = 704d
        p2g_input = np.concatenate([char_out, word_feedback], axis=-1)

        # P2G logits
        p2g_logits = self._linear(p2g_input, self.p2g_weight, self.p2g_bias)

        # P2G results
        p2g_preds = p2g_logits.argmax(axis=-1)
        ortho_results = []
        for w in range(n_words):
            ws = word_starts[w]
            we = word_ends[w]
            word_labels = [self.idx2p2g.get(int(p2g_preds[i]), "_CONT")
                          for i in range(ws, we + 1)]
            ortho_results.append(reconstruct_ortho(word_labels))

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
        }

    def analyser_avec_alternatives(
        self, ipa_words: list[str], k: int = 5
    ) -> dict[str, Any]:
        """Comme analyser() mais retourne aussi les alternatives et confiances."""
        if not ipa_words:
            return {
                "ipa_words": [], "ortho": [], "pos": [], "morpho": {},
                "alternatives": [], "confiance": [],
            }

        char_ids, word_starts, word_ends = self._encode_sentence(ipa_words)
        n_words = len(ipa_words)

        # Character embedding + BiLSTM
        emb = self.char_embedding[char_ids]
        char_out = self.char_bilstm.forward(emb)

        # Word representations + Word BiLSTM (needed for word feedback)
        char_hidden = self.config["char_hidden_dim"]
        fwd = char_out[:, :char_hidden]
        bwd = char_out[:, char_hidden:]
        word_repr = np.zeros((n_words, char_hidden * 2))
        for w in range(n_words):
            word_repr[w, :char_hidden] = fwd[word_ends[w]]
            word_repr[w, char_hidden:] = bwd[word_starts[w]]
        word_out = self.word_bilstm.forward(word_repr)

        # POS
        pos_logits = self._linear(word_out, self.pos_weight, self.pos_bias)
        pos_preds = pos_logits.argmax(axis=-1)
        pos_results = [self.idx2pos.get(int(pos_preds[w]), "NOM") for w in range(n_words)]

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

        # Word feedback: broadcast word representations to char positions
        word_feedback = np.zeros((len(char_ids), word_out.shape[1]))
        for w in range(n_words):
            ws = word_starts[w]
            we = word_ends[w]
            for i in range(ws, we + 1):
                word_feedback[i] = word_out[w]

        # P2G input: char_out (320d) || word_feedback (384d) = 704d
        p2g_input = np.concatenate([char_out, word_feedback], axis=-1)

        # P2G logits + softmax
        p2g_logits = self._linear(p2g_input, self.p2g_weight, self.p2g_bias)
        exp_l = np.exp(p2g_logits - p2g_logits.max(axis=-1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
        p2g_preds = p2g_logits.argmax(axis=-1)

        ortho_results = []
        alternatives_results = []
        confiance_results = []

        for w in range(n_words):
            ws = word_starts[w]
            we = word_ends[w]
            positions = list(range(ws, we + 1))

            word_labels = [
                self.idx2p2g.get(int(p2g_preds[i]), "_CONT") for i in positions
            ]
            ortho_top1 = reconstruct_ortho(word_labels)
            ortho_results.append(ortho_top1)

            word_probs = [float(probs[i, p2g_preds[i]]) for i in positions]
            confiance = 1.0
            for p in word_probs:
                confiance *= p
            n_pos = len(positions)
            if n_pos > 0:
                confiance = confiance ** (1.0 / n_pos)
            confiance_results.append(confiance)

            alternatives: list[tuple[str, float]] = [(ortho_top1, confiance)]
            for pos_idx, i in enumerate(positions):
                if probs[i, p2g_preds[i]] >= 0.8:
                    continue
                top_k_indices = np.argsort(probs[i])[-k:][::-1]
                for rank, alt_idx in enumerate(top_k_indices):
                    if rank == 0:
                        continue
                    alt_prob = float(probs[i, alt_idx])
                    if alt_prob < 0.01:
                        break
                    alt_labels = list(word_labels)
                    alt_labels[pos_idx] = self.idx2p2g.get(int(alt_idx), "_CONT")
                    alt_ortho = reconstruct_ortho(alt_labels)
                    if alt_ortho and alt_ortho != ortho_top1:
                        alt_probs = list(word_probs)
                        alt_probs[pos_idx] = alt_prob
                        alt_conf = 1.0
                        for p in alt_probs:
                            alt_conf *= p
                        if n_pos > 0:
                            alt_conf = alt_conf ** (1.0 / n_pos)
                        alternatives.append((alt_ortho, alt_conf))

            seen: set[str] = set()
            unique_alts: list[tuple[str, float]] = []
            for ortho, score in sorted(alternatives, key=lambda x: -x[1]):
                if ortho not in seen:
                    seen.add(ortho)
                    unique_alts.append((ortho, score))
            alternatives_results.append(unique_alts[:k])

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
            "alternatives": alternatives_results,
            "confiance": confiance_results,
        }
