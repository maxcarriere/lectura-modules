"""Inférence ONNX Runtime pour le modèle unifié P2G V2 (avec features lexicales).

V2 par rapport à V1 :
  - Supporte les lex_features (24d) pour améliorer POS/Morpho
  - API V1-compatible (analyser) + API V2 (analyser_v2 avec top-K POS/Morpho)
  - Conserve analyser_avec_alternatives() pour les alternatives P2G

Usage :
    engine = OnnxInferenceEngineV2(
        "modeles/unifie_p2g_v3.onnx", "modeles/unifie_p2g_v3_vocab.json",
        lexicon_path="path/to/lexique_pos_candidates.json",
    )
    result = engine.analyser(["le", "ʃa"])
    result = engine.analyser_v2(["le", "ʃa"], top_k=3)
    result = engine.analyser_avec_alternatives(["le", "ʃa"], k=5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lectura_p2g.utils.p2g_labels import reconstruct_ortho

logger = logging.getLogger(__name__)

ALL_LEX_POS = [
    "ADJ", "ADJ:dem", "ADJ:ind", "ADJ:int", "ADJ:num", "ADJ:pos",
    "ADV", "ART:def", "ART:ind", "AUX", "CON", "INTJ",
    "NOM", "PRE",
    "PRO:dem", "PRO:ind", "PRO:int", "PRO:per", "PRO:pos", "PRO:rel",
    "VER",
]

LEX_FEATURE_DIM = len(ALL_LEX_POS) + 3


def _build_lex_features(word: str, lexicon: dict[str, list[str]] | None) -> list[float]:
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


class OnnxInferenceEngineV2:
    """Inférence ONNX Runtime pour le modèle unifié P2G V2."""

    def __init__(
        self,
        onnx_path: str | Path,
        vocab_path: str | Path,
        lexicon_path: str | Path | None = None,
        lexicon: dict[str, list[str]] | None = None,
    ):
        import onnxruntime as ort

        logger.info("Loading P2G V2 ONNX model from %s", onnx_path)
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

        self.lex_feature_dim = self.config.get("lex_feature_dim", LEX_FEATURE_DIM)

        if lexicon is not None:
            self.lexicon = lexicon
        elif lexicon_path is not None:
            with open(lexicon_path, encoding="utf-8") as f:
                self.lexicon = json.load(f)
            logger.info("Loaded lexicon: %d words", len(self.lexicon))
        else:
            self.lexicon = None

    def _encode_sentence(
        self, ipa_words: list[str], ortho_words: list[str] | None = None,
        *, use_lex: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Encode IPA words en inputs ONNX.

        Args:
            ipa_words: Mots IPA.
            ortho_words: Mots orthographiques pour lookup lexique.
                         Si None, pas de lex features.
            use_lex: Si True, utilise les features lexicales.
        """
        chars: list[str] = ["<BOS>"]
        word_starts: list[int] = []
        word_ends: list[int] = []

        for w_idx, word in enumerate(ipa_words):
            if w_idx > 0:
                chars.append("<SEP>")
            word_start = len(chars)
            for ch in word:
                chars.append(ch)
            word_end = len(chars) - 1
            word_starts.append(word_start)
            word_ends.append(word_end)

        chars.append("<EOS>")

        char_ids = np.array(
            [[self.char2idx.get(ch, 1) for ch in chars]], dtype=np.int64
        )
        ws = np.array([word_starts], dtype=np.int64)
        we = np.array([word_ends], dtype=np.int64)

        # Build lex features from ortho_words (if available and use_lex)
        lex_feats = []
        for w_idx in range(len(ipa_words)):
            if use_lex and ortho_words and w_idx < len(ortho_words):
                lex_feats.append(_build_lex_features(ortho_words[w_idx], self.lexicon))
            else:
                lex_feats.append([0.0] * self.lex_feature_dim)

        lex_features = np.array([lex_feats], dtype=np.float32)

        return char_ids, ws, we, lex_features, chars

    def analyser(
        self, ipa_words: list[str], ortho_words: list[str] | None = None,
        *, use_lex: bool = True,
    ) -> dict[str, Any]:
        """API V1-compatible.

        Args:
            ipa_words: Mots IPA.
            ortho_words: Mots orthographiques pour lookup lexique (optionnel).
            use_lex: Si True (defaut), utilise les features lexicales.
        """
        if not ipa_words:
            return {"ipa_words": [], "ortho": [], "pos": [], "morpho": {}}

        char_ids, word_starts, word_ends, lex_features, chars = self._encode_sentence(
            ipa_words, ortho_words, use_lex=use_lex,
        )

        outputs = self.session.run(
            None,
            {
                "char_ids": char_ids,
                "word_starts": word_starts,
                "word_ends": word_ends,
                "lex_features": lex_features,
            },
        )

        output_names = [o.name for o in self.session.get_outputs()]
        output_dict = dict(zip(output_names, outputs))
        n_words = len(ipa_words)

        # P2G
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
        pos_results = []
        if "pos_logits" in output_dict:
            pos_preds = output_dict["pos_logits"][0].argmax(axis=-1)
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

    def analyser_v2(
        self,
        ipa_words: list[str],
        ortho_words: list[str] | None = None,
        top_k: int = 3,
        *,
        use_lex: bool = True,
    ) -> dict[str, Any]:
        """API V2 : comme analyser() + top-K POS/Morpho avec scores."""
        if not ipa_words:
            return {
                "ipa_words": [], "ortho": [], "pos": [], "morpho": {},
                "pos_scores": [], "morpho_scores": {}, "confiance_pos": [],
            }

        char_ids, word_starts, word_ends, lex_features, chars = self._encode_sentence(
            ipa_words, ortho_words, use_lex=use_lex,
        )

        outputs = self.session.run(
            None,
            {
                "char_ids": char_ids,
                "word_starts": word_starts,
                "word_ends": word_ends,
                "lex_features": lex_features,
            },
        )

        output_names = [o.name for o in self.session.get_outputs()]
        output_dict = dict(zip(output_names, outputs))
        n_words = len(ipa_words)

        # P2G (standard)
        p2g_logits = output_dict["p2g_logits"]
        p2g_preds = p2g_logits[0].argmax(axis=-1)
        ortho_results = []
        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            word_labels = [self.idx2p2g.get(int(p2g_preds[i]), "_CONT") for i in range(ws, we + 1)]
            ortho_results.append(reconstruct_ortho(word_labels))

        # POS with top-K
        pos_results = []
        pos_scores = []
        confiance_pos = []
        if "pos_logits" in output_dict:
            pos_logits_2d = output_dict["pos_logits"][0]
            exp_l = np.exp(pos_logits_2d - pos_logits_2d.max(axis=-1, keepdims=True))
            pos_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

            for w in range(n_words):
                probs = pos_probs[w]
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_k_pos = [
                    (self.idx2pos.get(int(idx), "_"), float(probs[idx]))
                    for idx in top_indices
                    if self.idx2pos.get(int(idx), "<PAD>") != "<PAD>" and probs[idx] > 0.001
                ]
                pos_results.append(top_k_pos[0][0] if top_k_pos else "NOM")
                pos_scores.append(top_k_pos)
                confiance_pos.append(top_k_pos[0][1] if top_k_pos else 0.0)

        # Morpho with top-K
        morpho_results: dict[str, list[str]] = {}
        morpho_scores: dict[str, list[list[tuple[str, float]]]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key not in output_dict:
                continue
            feat_logits = output_dict[key][0]
            exp_l = np.exp(feat_logits - feat_logits.max(axis=-1, keepdims=True))
            feat_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

            feat_results = []
            feat_scores_list = []
            for w in range(n_words):
                probs = feat_probs[w]
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_k_feat = [
                    (idx2label.get(int(idx), "_"), float(probs[idx]))
                    for idx in top_indices
                    if idx2label.get(int(idx), "<PAD>") != "<PAD>" and probs[idx] > 0.001
                ]
                feat_results.append(top_k_feat[0][0] if top_k_feat else "_")
                feat_scores_list.append(top_k_feat)

            morpho_results[feat_name] = feat_results
            morpho_scores[feat_name] = feat_scores_list

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
            "pos_scores": pos_scores,
            "morpho_scores": morpho_scores,
            "confiance_pos": confiance_pos,
        }

    def analyser_avec_alternatives(
        self,
        ipa_words: list[str],
        ortho_words: list[str] | None = None,
        k: int = 5,
        *,
        use_lex: bool = True,
    ) -> dict[str, Any]:
        """Retourne alternatives P2G + top-K POS/Morpho.

        Combine les alternatives orthographiques (comme V1) avec les
        scores POS/Morpho V2.
        """
        if not ipa_words:
            return {
                "ipa_words": [], "ortho": [], "pos": [], "morpho": {},
                "alternatives": [], "confiance": [],
                "pos_scores": [], "morpho_scores": [],
            }

        char_ids, word_starts, word_ends, lex_features, chars = self._encode_sentence(
            ipa_words, ortho_words, use_lex=use_lex,
        )

        outputs = self.session.run(
            None,
            {
                "char_ids": char_ids,
                "word_starts": word_starts,
                "word_ends": word_ends,
                "lex_features": lex_features,
            },
        )

        output_names = [o.name for o in self.session.get_outputs()]
        output_dict = dict(zip(output_names, outputs))
        n_words = len(ipa_words)
        p2g_logits = output_dict["p2g_logits"]

        # Softmax P2G
        logits_2d = p2g_logits[0]
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

            word_labels = [self.idx2p2g.get(int(p2g_preds[i]), "_CONT") for i in positions]
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

        # POS with top-K
        pos_results = []
        pos_scores = []
        if "pos_logits" in output_dict:
            pos_logits_2d = output_dict["pos_logits"][0]
            exp_l = np.exp(pos_logits_2d - pos_logits_2d.max(axis=-1, keepdims=True))
            pos_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
            for w in range(n_words):
                p = pos_probs[w]
                top_indices = np.argsort(p)[-k:][::-1]
                top_k_pos = [
                    (self.idx2pos.get(int(idx), "_"), float(p[idx]))
                    for idx in top_indices
                    if self.idx2pos.get(int(idx), "<PAD>") != "<PAD>" and p[idx] > 0.001
                ]
                pos_results.append(top_k_pos[0][0] if top_k_pos else "NOM")
                pos_scores.append(top_k_pos)

        # Morpho with top-K
        morpho_results: dict[str, list[str]] = {}
        morpho_scores: dict[str, list[list[tuple[str, float]]]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key not in output_dict:
                continue
            feat_logits = output_dict[key][0]
            exp_l = np.exp(feat_logits - feat_logits.max(axis=-1, keepdims=True))
            feat_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)
            feat_results = []
            feat_scores_list = []
            for w in range(n_words):
                p = feat_probs[w]
                top_indices = np.argsort(p)[-k:][::-1]
                top_k_feat = [
                    (idx2label.get(int(idx), "_"), float(p[idx]))
                    for idx in top_indices
                    if idx2label.get(int(idx), "<PAD>") != "<PAD>" and p[idx] > 0.001
                ]
                feat_results.append(top_k_feat[0][0] if top_k_feat else "_")
                feat_scores_list.append(top_k_feat)
            morpho_results[feat_name] = feat_results
            morpho_scores[feat_name] = feat_scores_list

        return {
            "ipa_words": ipa_words,
            "ortho": ortho_results,
            "pos": pos_results,
            "morpho": morpho_results,
            "alternatives": alternatives_results,
            "confiance": confiance_results,
            "pos_scores": pos_scores,
            "morpho_scores": morpho_scores,
        }
