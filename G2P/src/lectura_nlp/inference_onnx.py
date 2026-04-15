"""Inférence ONNX Runtime pour le modèle unifié.

Dépendance : onnxruntime

Usage :
    engine = OnnxInferenceEngine("modeles/unifie.onnx", "modeles/unifie_vocab.json")
    result = engine.analyser("Les enfants sont arrivés")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OnnxInferenceEngine:
    """Inférence via ONNX Runtime."""

    def __init__(self, onnx_path: str | Path, vocab_path: str | Path):
        import onnxruntime as ort

        logger.info("Loading G2P ONNX model from %s", onnx_path)
        self.session = ort.InferenceSession(str(onnx_path))

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

    def _encode_sentence(
        self, tokens: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Encode tokens into model inputs."""
        chars: list[str] = ["<BOS>"]
        word_start_positions: list[int] = []
        word_end_positions: list[int] = []

        for w_idx, token in enumerate(tokens):
            if w_idx > 0:
                chars.append("<SEP>")
            word_start = len(chars)
            for ch in token.lower():
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

    def analyser(self, tokens: list[str]) -> dict[str, Any]:
        """Analyse une liste de tokens.

        Returns:
            {
                "tokens": [...],
                "g2p": [ipa_per_token],
                "pos": [pos_per_token],
                "liaison": [liaison_per_token],
                "morpho": {feat: [val_per_token]},
            }
        """
        logger.debug("analyser() called with %s tokens", len(tokens))
        if not tokens:
            return {"tokens": [], "g2p": [], "pos": [], "liaison": [], "morpho": {}}

        char_ids, word_starts, word_ends, chars = self._encode_sentence(tokens)

        outputs = self.session.run(
            None,
            {
                "char_ids": char_ids,
                "word_starts": word_starts,
                "word_ends": word_ends,
            },
        )

        # Parse outputs based on session output names
        output_names = [o.name for o in self.session.get_outputs()]
        output_dict = dict(zip(output_names, outputs))

        n_words = len(tokens)

        # G2P: reconstruct IPA per word
        g2p_logits = output_dict["g2p_logits"]
        g2p_preds = g2p_logits[0].argmax(axis=-1)

        g2p_results = []
        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            word_labels = [self.idx2g2p.get(int(g2p_preds[i]), "_CONT") for i in range(ws, we + 1)]
            ipa = "".join(lab for lab in word_labels if lab != "_CONT" and lab != "<PAD>")
            g2p_results.append(ipa)

        # POS
        pos_logits = output_dict.get("pos_logits")
        pos_results = []
        if pos_logits is not None:
            pos_preds = pos_logits[0].argmax(axis=-1)
            for w in range(n_words):
                pos_results.append(self.idx2pos.get(int(pos_preds[w]), "NOM"))

        # Liaison
        liaison_logits = output_dict.get("liaison_logits")
        liaison_results = []
        if liaison_logits is not None:
            liaison_preds = liaison_logits[0].argmax(axis=-1)
            for w in range(n_words):
                liaison_results.append(self.idx2liaison.get(int(liaison_preds[w]), "none"))

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
            "tokens": tokens,
            "g2p": g2p_results,
            "pos": pos_results,
            "liaison": liaison_results,
            "morpho": morpho_results,
        }
