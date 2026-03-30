"""Inférence ONNX Runtime pour le modèle unifié P2G.

Dépendance : onnxruntime

Usage :
    engine = OnnxInferenceEngine("modeles/unifie_p2g.onnx", "modeles/unifie_p2g_vocab.json")
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


class OnnxInferenceEngine:
    """Inférence via ONNX Runtime."""

    def __init__(self, onnx_path: str | Path, vocab_path: str | Path):
        import onnxruntime as ort

        logger.info("Loading P2G ONNX model from %s", onnx_path)
        self.session = ort.InferenceSession(str(onnx_path))

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

    def _encode_sentence(
        self, ipa_words: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Encode IPA words into model inputs."""
        chars: list[str] = ["<BOS>"]
        word_start_positions: list[int] = []
        word_end_positions: list[int] = []

        for w_idx, word in enumerate(ipa_words):
            if w_idx > 0:
                chars.append("<SEP>")
            word_start = len(chars)
            for ch in word:
                chars.append(ch)
            word_end = len(chars) - 1
            word_start_positions.append(word_start)
            word_end_positions.append(word_end)

        chars.append("<EOS>")

        char_ids = np.array(
            [[self.char2idx.get(ch, 1) for ch in chars]], dtype=np.int64
        )
        word_starts = np.array([word_start_positions], dtype=np.int64)
        word_ends = np.array([word_end_positions], dtype=np.int64)

        return char_ids, word_starts, word_ends, chars

    def analyser(self, ipa_words: list[str]) -> dict[str, Any]:
        """Analyse une liste de mots IPA.

        Returns:
            {
                "ipa_words": [...],
                "ortho": [ortho_per_word],
                "pos": [pos_per_word],
                "morpho": {feat: [val_per_word]},
            }
        """
        logger.debug("analyser() called with %s IPA words", len(ipa_words))
        if not ipa_words:
            return {"ipa_words": [], "ortho": [], "pos": [], "morpho": {}}

        char_ids, word_starts, word_ends, chars = self._encode_sentence(ipa_words)

        outputs = self.session.run(
            None,
            {
                "char_ids": char_ids,
                "word_starts": word_starts,
                "word_ends": word_ends,
            },
        )

        output_names = [o.name for o in self.session.get_outputs()]
        output_dict = dict(zip(output_names, outputs))

        n_words = len(ipa_words)

        # P2G: reconstruct orthography per word
        p2g_logits = output_dict["p2g_logits"]
        p2g_preds = p2g_logits[0].argmax(axis=-1)

        ortho_results = []
        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            word_labels = [self.idx2p2g.get(int(p2g_preds[i]), "_CONT") for i in range(ws, we + 1)]
            ortho = reconstruct_ortho(word_labels)
            ortho_results.append(ortho)

        # POS
        pos_logits = output_dict.get("pos_logits")
        pos_results = []
        if pos_logits is not None:
            pos_preds = pos_logits[0].argmax(axis=-1)
            for w in range(n_words):
                pos_results.append(self.idx2pos.get(int(pos_preds[w]), "NOM"))

        # Morpho
        morpho_results: dict[str, list[str]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key in output_dict:
                feat_preds = output_dict[key][0].argmax(axis=-1)
                morpho_results[feat_name] = [
                    idx2label.get(int(feat_preds[w]), "_") for w in range(n_words)
                ]

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
        }

    def analyser_avec_alternatives(
        self, ipa_words: list[str], k: int = 5
    ) -> dict[str, Any]:
        """Comme analyser() mais retourne aussi les alternatives et confiances.

        Retour supplémentaire par rapport à analyser() :
            "alternatives": list[list[tuple[str, float]]]  # par mot, (ortho, score)
            "confiance": list[float]                        # confiance top-1 par mot
        """
        if not ipa_words:
            return {
                "ipa_words": [], "ortho": [], "pos": [], "morpho": {},
                "alternatives": [], "confiance": [],
            }

        char_ids, word_starts, word_ends, chars = self._encode_sentence(ipa_words)

        outputs = self.session.run(
            None,
            {
                "char_ids": char_ids,
                "word_starts": word_starts,
                "word_ends": word_ends,
            },
        )

        output_names = [o.name for o in self.session.get_outputs()]
        output_dict = dict(zip(output_names, outputs))

        n_words = len(ipa_words)
        p2g_logits = output_dict["p2g_logits"]  # (1, seq_len, n_labels)

        # Softmax sur la dimension labels
        logits_2d = p2g_logits[0]  # (seq_len, n_labels)
        exp_l = np.exp(logits_2d - logits_2d.max(axis=-1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

        p2g_preds = logits_2d.argmax(axis=-1)

        ortho_results = []
        alternatives_results = []
        confiance_results = []

        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            positions = list(range(ws, we + 1))

            # Top-1 : reconstruction standard
            word_labels = [
                self.idx2p2g.get(int(p2g_preds[i]), "_CONT") for i in positions
            ]
            ortho_top1 = reconstruct_ortho(word_labels)
            ortho_results.append(ortho_top1)

            # Confiance = produit des probabilités top-1 par position
            word_probs = [float(probs[i, p2g_preds[i]]) for i in positions]
            confiance = 1.0
            for p in word_probs:
                confiance *= p
            # Normaliser en racine n-ième (moyenne géométrique)
            n_pos = len(positions)
            if n_pos > 0:
                confiance = confiance ** (1.0 / n_pos)
            confiance_results.append(confiance)

            # Alternatives : identifier les positions incertaines (top-1 < 0.8)
            # et générer des variantes par substitution
            alternatives: list[tuple[str, float]] = [(ortho_top1, confiance)]

            for pos_idx, i in enumerate(positions):
                if probs[i, p2g_preds[i]] >= 0.8:
                    continue
                # Top-K pour cette position
                top_k_indices = np.argsort(probs[i])[-k:][::-1]
                for rank, alt_idx in enumerate(top_k_indices):
                    if rank == 0:
                        continue  # Skip top-1 (already used)
                    alt_prob = float(probs[i, alt_idx])
                    if alt_prob < 0.01:
                        break
                    # Substituer cette position et reconstruire
                    alt_labels = list(word_labels)
                    alt_labels[pos_idx] = self.idx2p2g.get(int(alt_idx), "_CONT")
                    alt_ortho = reconstruct_ortho(alt_labels)
                    if alt_ortho and alt_ortho != ortho_top1:
                        # Score = moyenne géométrique avec substitution
                        alt_probs = list(word_probs)
                        alt_probs[pos_idx] = alt_prob
                        alt_conf = 1.0
                        for p in alt_probs:
                            alt_conf *= p
                        if n_pos > 0:
                            alt_conf = alt_conf ** (1.0 / n_pos)
                        alternatives.append((alt_ortho, alt_conf))

            # Dédupliquer et trier
            seen: set[str] = set()
            unique_alts: list[tuple[str, float]] = []
            for ortho, score in sorted(alternatives, key=lambda x: -x[1]):
                if ortho not in seen:
                    seen.add(ortho)
                    unique_alts.append((ortho, score))
            alternatives_results.append(unique_alts[:k])

        # POS
        pos_logits = output_dict.get("pos_logits")
        pos_results = []
        if pos_logits is not None:
            pos_preds = pos_logits[0].argmax(axis=-1)
            for w in range(n_words):
                pos_results.append(self.idx2pos.get(int(pos_preds[w]), "NOM"))

        # Morpho
        morpho_results: dict[str, list[str]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key in output_dict:
                feat_preds = output_dict[key][0].argmax(axis=-1)
                morpho_results[feat_name] = [
                    idx2label.get(int(feat_preds[w]), "_") for w in range(n_words)
                ]

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
            "alternatives": alternatives_results,
            "confiance": confiance_results,
        }
