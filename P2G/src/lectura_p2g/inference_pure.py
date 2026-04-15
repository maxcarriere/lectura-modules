"""Inférence pur Python (zéro dépendance) pour le modèle unifié P2G.

N'utilise que la bibliothèque standard Python. Charge les poids depuis JSON.
Plus lent que NumPy/ONNX mais totalement portable.

Usage :
    engine = PureInferenceEngine("modeles/unifie_p2g_weights.json", "modeles/unifie_p2g_vocab.json")
    result = engine.analyser(["le", "ʃa"])
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from lectura_p2g.utils.p2g_labels import reconstruct_ortho

logger = logging.getLogger(__name__)


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

            fwd_out = []
            h_f = [0.0] * hs
            c_f = [0.0] * hs
            for t in range(seq_len):
                h_f, c_f = fwd_cell.forward(current[t], h_f, c_f)
                fwd_out.append(h_f[:])

            bwd_out = [[0.0] * hs for _ in range(seq_len)]
            h_b = [0.0] * hs
            c_b = [0.0] * hs
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = bwd_cell.forward(current[t], h_b, c_b)
                bwd_out[t] = h_b[:]

            current = [fwd_out[t] + bwd_out[t] for t in range(seq_len)]

        return current


class PureInferenceEngine:
    """Inférence pur Python pour le modèle unifié P2G."""

    def __init__(self, weights_path: str | Path, vocab_path: str | Path):
        logger.info("Loading P2G pure-Python model from %s", weights_path)
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
        self.char_embedding = self._get("char_embedding.weight")

        self.char_bilstm = self._build_lstm(
            "char_lstm",
            self.config["char_num_layers"],
            self.config["char_hidden_dim"],
        )

        self.p2g_weight = self._get("p2g_head.weight")
        self.p2g_bias = self._get("p2g_head.bias")

        self.word_bilstm = self._build_lstm(
            "word_lstm",
            self.config.get("word_num_layers", 1),
            self.config["word_hidden_dim"],
        )

        self.pos_weight = self._get("pos_head.weight")
        self.pos_bias = self._get("pos_head.bias")

        self.morpho_weights = {}
        self.morpho_biases = {}
        for feat_name in self.idx2morpho:
            self.morpho_weights[feat_name] = self._get(f"morpho_heads.{feat_name}.weight")
            self.morpho_biases[feat_name] = self._get(f"morpho_heads.{feat_name}.bias")

    def _linear(self, x: list[float], weight: list[list[float]], bias: list[float]) -> list[float]:
        return _add(_matvec(weight, x), bias)

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

        # Character embedding
        emb = [self.char_embedding[cid][:] for cid in char_ids]

        # Character BiLSTM
        char_out = self.char_bilstm.forward(emb)

        # P2G
        n_words = len(ipa_words)
        ortho_results = []
        for w in range(n_words):
            ws = word_starts[w]
            we = word_ends[w]
            word_labels = []
            for i in range(ws, we + 1):
                logits = self._linear(char_out[i], self.p2g_weight, self.p2g_bias)
                pred = _argmax(logits)
                label = self.idx2p2g.get(pred, "_CONT")
                word_labels.append(label)
            ortho_results.append(reconstruct_ortho(word_labels))

        # Word representations
        char_hidden = self.config["char_hidden_dim"]
        word_repr = []
        for w in range(n_words):
            fwd_at_end = char_out[word_ends[w]][:char_hidden]
            bwd_at_start = char_out[word_starts[w]][char_hidden:]
            word_repr.append(fwd_at_end + bwd_at_start)

        # Word BiLSTM
        word_out = self.word_bilstm.forward(word_repr)

        # POS
        pos_results = []
        for w in range(n_words):
            logits = self._linear(word_out[w], self.pos_weight, self.pos_bias)
            pos_results.append(self.idx2pos.get(_argmax(logits), "NOM"))

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

        # Character embedding + BiLSTM
        emb = [self.char_embedding[cid][:] for cid in char_ids]
        char_out = self.char_bilstm.forward(emb)

        n_words = len(ipa_words)
        ortho_results = []
        alternatives_results = []
        confiance_results = []

        for w in range(n_words):
            ws = word_starts[w]
            we = word_ends[w]
            positions = list(range(ws, we + 1))

            # Compute logits and softmax per position
            all_logits: list[list[float]] = []
            for i in positions:
                logits = self._linear(char_out[i], self.p2g_weight, self.p2g_bias)
                all_logits.append(logits)

            # Softmax per position
            all_probs: list[list[float]] = []
            for logits in all_logits:
                max_l = max(logits)
                exp_l = [math.exp(v - max_l) for v in logits]
                sum_exp = sum(exp_l)
                all_probs.append([e / sum_exp for e in exp_l])

            # Top-1
            word_labels = []
            word_preds = []
            word_top1_probs = []
            for pos_idx, logits in enumerate(all_logits):
                pred = _argmax(logits)
                word_preds.append(pred)
                word_labels.append(self.idx2p2g.get(pred, "_CONT"))
                word_top1_probs.append(all_probs[pos_idx][pred])

            ortho_top1 = reconstruct_ortho(word_labels)
            ortho_results.append(ortho_top1)

            # Confiance (moyenne géométrique)
            confiance = 1.0
            for p in word_top1_probs:
                confiance *= p
            n_pos = len(positions)
            if n_pos > 0:
                confiance = confiance ** (1.0 / n_pos)
            confiance_results.append(confiance)

            # Alternatives
            alternatives: list[tuple[str, float]] = [(ortho_top1, confiance)]
            for pos_idx in range(len(positions)):
                if word_top1_probs[pos_idx] >= 0.8:
                    continue
                # Top-K for this position
                prob_row = all_probs[pos_idx]
                indexed = sorted(enumerate(prob_row), key=lambda x: -x[1])
                for rank, (alt_idx, alt_prob) in enumerate(indexed[:k]):
                    if rank == 0:
                        continue
                    if alt_prob < 0.01:
                        break
                    alt_labels = list(word_labels)
                    alt_labels[pos_idx] = self.idx2p2g.get(alt_idx, "_CONT")
                    alt_ortho = reconstruct_ortho(alt_labels)
                    if alt_ortho and alt_ortho != ortho_top1:
                        alt_probs = list(word_top1_probs)
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

        # Word representations + Word BiLSTM
        char_hidden = self.config["char_hidden_dim"]
        word_repr = []
        for w in range(n_words):
            fwd_at_end = char_out[word_ends[w]][:char_hidden]
            bwd_at_start = char_out[word_starts[w]][char_hidden:]
            word_repr.append(fwd_at_end + bwd_at_start)
        word_out = self.word_bilstm.forward(word_repr)

        # POS
        pos_results = []
        for w in range(n_words):
            logits = self._linear(word_out[w], self.pos_weight, self.pos_bias)
            pos_results.append(self.idx2pos.get(_argmax(logits), "NOM"))

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
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
            "alternatives": alternatives_results,
            "confiance": confiance_results,
        }
