"""Inférence ONNX Runtime pour le modèle unifié G2P V2 (avec features lexicales).

V2 par rapport à V1 :
  - Supporte les lex_features (24d) pour améliorer POS/Morpho/Liaison
  - API V1-compatible (analyser) + API V2 (analyser_v2 avec top-K POS/Morpho)
  - Sans lexique, se comporte identiquement à V1

Usage :
    engine = OnnxInferenceEngineV2(
        "modeles/unifie_v2.onnx", "modeles/unifie_v2_vocab.json",
        lexicon_path="modeles/lexique_pos_candidates.json",
    )
    # Mode V1 (compatible)
    result = engine.analyser(["Les", "enfants"])
    # Mode V2 (avec lex + top-K)
    result = engine.analyser_v2(["Les", "enfants"], top_k=3)
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


class OnnxInferenceEngineV2:
    """Inférence ONNX Runtime pour le modèle unifié G2P V2."""

    def __init__(
        self,
        onnx_path: str | Path,
        vocab_path: str | Path,
        lexicon_path: str | Path | None = None,
        lexicon: dict[str, list[str]] | None = None,
    ):
        import onnxruntime as ort

        logger.info("Loading G2P V2 ONNX model from %s", onnx_path)
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

    def _encode_sentence(
        self, tokens: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Encode tokens en inputs ONNX (char_ids, word_starts, word_ends, lex_features)."""
        chars: list[str] = ["<BOS>"]
        word_starts: list[int] = []
        word_ends: list[int] = []

        for w_idx, token in enumerate(tokens):
            if w_idx > 0:
                chars.append("<SEP>")
            word_start = len(chars)
            for ch in token.lower():
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

        # Build lex features
        lex_feats = []
        for token in tokens:
            lex_feats.append(_build_lex_features(token, self.lexicon))
        lex_features = np.array([lex_feats], dtype=np.float32)

        return char_ids, ws, we, lex_features, chars

    def analyser(self, tokens: list[str]) -> dict[str, Any]:
        """API V1-compatible. Retourne le résultat standard (argmax)."""
        if not tokens:
            return {"tokens": [], "g2p": [], "pos": [], "liaison": [], "morpho": {}}

        char_ids, word_starts, word_ends, lex_features, chars = self._encode_sentence(tokens)

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

        n_words = len(tokens)

        # G2P
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
        pos_results = []
        if "pos_logits" in output_dict:
            pos_preds = output_dict["pos_logits"][0].argmax(axis=-1)
            for w in range(n_words):
                pos_results.append(self.idx2pos.get(int(pos_preds[w]), "NOM"))

        # Liaison
        liaison_results = []
        if "liaison_logits" in output_dict:
            liaison_preds = output_dict["liaison_logits"][0].argmax(axis=-1)
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

    def analyser_v2(
        self,
        tokens: list[str],
        top_k: int = 3,
        candidates: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """API V2 : comme analyser() + top-K POS/Morpho avec scores de confiance.

        Args:
            tokens: Liste de tokens.
            top_k: Nombre de candidats POS/Morpho à retourner.
            candidates: Candidats POS par mot {mot: [pos1, pos2, ...]}.
                        Si None, utilise le lexique interne.

        Returns:
            Résultat standard + :
                "pos_scores": list[list[tuple[str, float]]]  # top-K POS par mot
                "morpho_scores": dict[str, list[list[tuple[str, float]]]]  # top-K par feat
                "confiance_pos": list[float]  # confiance top-1 POS par mot
        """
        if not tokens:
            return {
                "tokens": [], "g2p": [], "pos": [], "liaison": [], "morpho": {},
                "pos_scores": [], "morpho_scores": {}, "confiance_pos": [],
            }

        char_ids, word_starts, word_ends, lex_features, chars = self._encode_sentence(tokens)

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
        n_words = len(tokens)

        # G2P (standard)
        g2p_logits = output_dict["g2p_logits"]
        g2p_preds = g2p_logits[0].argmax(axis=-1)
        g2p_results = []
        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            word_labels = [self.idx2g2p.get(int(g2p_preds[i]), "_CONT") for i in range(ws, we + 1)]
            ipa = "".join(lab for lab in word_labels if lab != "_CONT" and lab != "<PAD>")
            g2p_results.append(ipa)

        # Liaison (standard)
        liaison_results = []
        if "liaison_logits" in output_dict:
            liaison_preds = output_dict["liaison_logits"][0].argmax(axis=-1)
            for w in range(n_words):
                liaison_results.append(self.idx2liaison.get(int(liaison_preds[w]), "none"))

        # POS with top-K scores
        pos_results = []
        pos_scores = []
        confiance_pos = []
        if "pos_logits" in output_dict:
            pos_logits_2d = output_dict["pos_logits"][0]  # (n_words, n_pos)
            exp_l = np.exp(pos_logits_2d - pos_logits_2d.max(axis=-1, keepdims=True))
            pos_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

            for w in range(n_words):
                probs = pos_probs[w]

                # Build allowed set from candidates
                allowed = None
                if candidates:
                    word_cands = candidates.get(tokens[w].lower())
                    if word_cands:
                        allowed = set()
                        for cand_pos in word_cands:
                            for idx, pos_label in self.idx2pos.items():
                                if pos_label == cand_pos or pos_label.startswith(cand_pos + ":"):
                                    allowed.add(idx)

                # Apply masking if allowed set exists (soft: set non-allowed to 0)
                if allowed:
                    masked_probs = np.zeros_like(probs)
                    for idx in allowed:
                        if idx < len(probs):
                            masked_probs[idx] = probs[idx]
                    total = masked_probs.sum()
                    if total > 0:
                        masked_probs /= total
                    else:
                        masked_probs = probs  # Fallback
                    probs = masked_probs

                # Top-K
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_k_pos = []
                for idx in top_indices:
                    label = self.idx2pos.get(int(idx), "_")
                    if label != "<PAD>" and probs[idx] > 0.001:
                        top_k_pos.append((label, float(probs[idx])))

                pos_results.append(top_k_pos[0][0] if top_k_pos else "NOM")
                pos_scores.append(top_k_pos)
                confiance_pos.append(top_k_pos[0][1] if top_k_pos else 0.0)

        # Morpho with top-K scores
        morpho_results: dict[str, list[str]] = {}
        morpho_scores: dict[str, list[list[tuple[str, float]]]] = {}
        for feat_name, idx2label in self.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key not in output_dict:
                continue

            feat_logits = output_dict[key][0]  # (n_words, n_labels)
            exp_l = np.exp(feat_logits - feat_logits.max(axis=-1, keepdims=True))
            feat_probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

            feat_results = []
            feat_scores = []
            for w in range(n_words):
                probs = feat_probs[w]
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_k_feat = []
                for idx in top_indices:
                    label = idx2label.get(int(idx), "_")
                    if label != "<PAD>" and probs[idx] > 0.001:
                        top_k_feat.append((label, float(probs[idx])))

                feat_results.append(top_k_feat[0][0] if top_k_feat else "_")
                feat_scores.append(top_k_feat)

            morpho_results[feat_name] = feat_results
            morpho_scores[feat_name] = feat_scores

        return {
            "tokens": tokens,
            "g2p": g2p_results,
            "pos": pos_results,
            "liaison": liaison_results,
            "morpho": morpho_results,
            "pos_scores": pos_scores,
            "morpho_scores": morpho_scores,
            "confiance_pos": confiance_pos,
        }
