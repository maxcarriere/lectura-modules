"""Adaptateur G2P Unifie V2 — satisfait TaggerProtocol + G2PProtocol.

Encapsule OnnxInferenceEngineV2 (lectura-modules-private/G2P) pour
fournir un tagger POS/Morpho et un G2P en un seul forward pass.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex tokenisation française (elisions)
_TOKEN_RE = re.compile(
    r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu|[Jj]usqu|[Qq]uelqu)"
    r"['\u2019]"
    r"|[\w]+(?:-[\w]+)*"
    r"|[^\s\w]+",
)


import math

# Mapping UD feature keys → FR feature keys (correcteur convention)
_UD_KEY_TO_FR: dict[str, str] = {
    "Number": "nombre",
    "Gender": "genre",
    "Tense": "temps",
    "Mood": "mode",
    "Person": "personne",
}

# Mapping UD feature values → FR feature values (lexique convention)
_UD_VAL_TO_FR: dict[str, str] = {
    # Gender
    "Masc": "m", "Fem": "f",
    # Number
    "Sing": "s", "Plur": "p",
    # Tense
    "Pres": "pre", "Past": "pas", "Imp": "imp", "Fut": "fut",
    # Mood
    "Ind": "ind", "Sub": "sub", "Cnd": "con",
    # Person (same)
    "1": "1", "2": "2", "3": "3",
}


def _softmax(scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Normalise en probabilites via softmax."""
    if not scores:
        return []
    max_v = max(s for _, s in scores)
    exp_vals = [(label, math.exp(s - max_v)) for label, s in scores]
    total = sum(v for _, v in exp_vals)
    if total <= 0:
        return scores
    return [(label, v / total) for label, v in exp_vals]


class G2PUnifieAdapter:
    """Adaptateur satisfaisant TaggerProtocol + G2PProtocol.

    Appelle OnnxInferenceEngineV2.analyser_v2() pour POS/Morpho/G2P
    en un seul forward pass ONNX (1.8 Mo int8).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def _analyser_cached(
        self, words: list[str], *, use_lex: bool = True,
    ) -> dict[str, Any]:
        """Appelle analyser_v2 avec cache par tuple de mots."""
        key = (tuple(words), use_lex)
        if key not in self._cache:
            try:
                self._cache[key] = self._engine.analyser_v2(
                    list(words), use_lex=use_lex,
                )
            except TypeError:
                # Engine doesn't support use_lex kwarg (older version)
                self._cache[key] = self._engine.analyser_v2(list(words))
        return self._cache[key]

    def _extraire_morpho_fr(self, result: dict, i: int) -> dict[str, str]:
        """Extrait les features morpho pour la position i, mappees en FR."""
        d: dict[str, str] = {}
        morpho = result.get("morpho", {})
        for ud_key, fr_key in _UD_KEY_TO_FR.items():
            vals = morpho.get(ud_key, [])
            if i < len(vals) and vals[i] != "_":
                mapped = _UD_VAL_TO_FR.get(vals[i], vals[i])
                d[fr_key] = mapped
        return d

    # -- TaggerProtocol --

    def tokenize(self, text: str) -> list[tuple[str, bool]]:
        """Tokenisation française (regex elisions)."""
        result: list[tuple[str, bool]] = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group()
            is_word = tok[0].isalpha() or tok[0] == "_"
            result.append((tok, is_word))
        return result

    def tag_words(self, words: list[str]) -> list[dict]:
        """Analyse POS/Morpho via le modele unifie V2.

        Retourne list[dict] avec pos, genre, nombre, temps, mode, personne
        (meme format que LexiqueTagger).
        """
        if not words:
            return []

        result = self._analyser_cached(words)
        n = len(words)
        tags: list[dict] = []

        for i in range(n):
            d: dict[str, str] = {}
            if i < len(result.get("pos", [])):
                d["pos"] = result["pos"][i]
            d.update(self._extraire_morpho_fr(result, i))
            tags.append(d)

        return tags

    def tag_words_rich(self, words: list[str]) -> list[dict]:
        """Analyse POS/Morpho enrichie avec scores de confiance.

        Retourne list[dict] avec en plus :
        - pos_scores: list[tuple[str, float]] (top-K POS avec probabilites)
        - confiance_pos: float (confiance du top-1)
        - g2p: str (IPA)
        """
        if not words:
            return []

        result = self._analyser_cached(words)
        n = len(words)
        tags: list[dict] = []

        raw_pos_scores = result.get("pos_scores", [])
        raw_confiance = result.get("confiance_pos", [])
        raw_g2p = result.get("g2p", [])

        for i in range(n):
            d: dict[str, Any] = {}
            if i < len(result.get("pos", [])):
                d["pos"] = result["pos"][i]
            d.update(self._extraire_morpho_fr(result, i))

            # POS scores (deja normalises par softmax dans le V2 engine)
            if i < len(raw_pos_scores):
                ps = raw_pos_scores[i]
                # Deja list[tuple[str, float]] normalise
                d["pos_scores"] = ps if ps else []
            else:
                d["pos_scores"] = []

            # Confiance POS
            if i < len(raw_confiance):
                d["confiance_pos"] = raw_confiance[i]
            else:
                d["confiance_pos"] = 1.0

            # G2P (IPA)
            if i < len(raw_g2p):
                d["g2p"] = raw_g2p[i]

            tags.append(d)

        return tags

    def tag_words_dual(self, words: list[str]) -> list[dict]:
        """Double tagging : blind (use_lex=False) + lex-attention (use_lex=True).

        Retourne list[dict] avec les champs habituels plus :
        - pos_blind: POS predit sans features lexicales
        - pos_scores_blind: top-K POS scores du mode blind
        - divergence_pos: True si pos_blind != pos (lex)
        """
        if not words:
            return []

        result_lex = self._analyser_cached(words, use_lex=True)
        result_blind = self._analyser_cached(words, use_lex=False)
        n = len(words)
        tags: list[dict] = []

        raw_pos_scores_lex = result_lex.get("pos_scores", [])
        raw_confiance_lex = result_lex.get("confiance_pos", [])
        raw_g2p = result_lex.get("g2p", [])
        raw_pos_scores_blind = result_blind.get("pos_scores", [])

        for i in range(n):
            d: dict[str, Any] = {}

            # POS from lex mode (primary)
            pos_lex = result_lex["pos"][i] if i < len(result_lex.get("pos", [])) else ""
            pos_blind = result_blind["pos"][i] if i < len(result_blind.get("pos", [])) else ""

            d["pos"] = pos_lex
            d["pos_blind"] = pos_blind
            d["divergence_pos"] = (pos_blind != pos_lex)

            # Morpho from lex mode
            d.update(self._extraire_morpho_fr(result_lex, i))

            # POS scores
            if i < len(raw_pos_scores_lex):
                d["pos_scores"] = raw_pos_scores_lex[i] or []
            else:
                d["pos_scores"] = []

            if i < len(raw_pos_scores_blind):
                d["pos_scores_blind"] = raw_pos_scores_blind[i] or []
            else:
                d["pos_scores_blind"] = []

            # Confiance POS
            if i < len(raw_confiance_lex):
                d["confiance_pos"] = raw_confiance_lex[i]
            else:
                d["confiance_pos"] = 1.0

            # G2P (IPA)
            if i < len(raw_g2p):
                d["g2p"] = raw_g2p[i]

            tags.append(d)

        return tags

    # -- G2PProtocol --

    def g2p(self, word: str) -> str:
        """Retourne l'IPA d'un mot via le modele unifie V2."""
        result = self._analyser_cached([word])
        g2p_list = result.get("g2p", [])
        return g2p_list[0] if g2p_list else ""

    def prononcer(self, mot: str) -> str | None:
        """Alias de g2p() pour satisfaire l'interface _candidats.py."""
        r = self.g2p(mot)
        return r if r else None


def creer_adapter_g2p_unifie(
    model_dir: str | Path | None = None,
) -> G2PUnifieAdapter | None:
    """Factory : cree un G2PUnifieAdapter si les fichiers sont disponibles.

    Tente d'abord d'utiliser le module lectura_nlp installe (qui a
    use_lex pour le double tagging). Si absent, fallback sur la copie
    locale dans Correcteur/data/g2p_v2/.

    Retourne None si onnxruntime n'est pas installe ou si les fichiers
    modele sont absents (degradation gracieuse).

    Args:
        model_dir: Repertoire contenant les fichiers G2P V2.
            Si None, essaie lectura_nlp puis Correcteur/data/g2p_v2/.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.info("onnxruntime non installe — G2P Unifie V2 indisponible")
        return None

    # 1. Essayer lectura_nlp installe (supporte use_lex)
    if model_dir is None:
        try:
            from lectura_nlp import creer_engine
            engine = creer_engine(mode="onnx")
            logger.info("G2P Unifie V2 charge via lectura_nlp")
            return G2PUnifieAdapter(engine)
        except Exception:
            logger.debug("lectura_nlp indisponible, fallback copie locale")

    # 2. Fallback : copie locale dans data/g2p_v2/
    if model_dir is None:
        model_dir = Path(__file__).resolve().parent / "data" / "g2p_v2"

    model_dir = Path(model_dir)

    onnx_path = model_dir / "modeles" / "unifie_v2_int8.onnx"
    vocab_path = model_dir / "modeles" / "unifie_v2_vocab.json"
    lexicon_path = model_dir / "lexique_pos_candidates.json"
    inference_dir = model_dir

    if not onnx_path.exists():
        logger.info("Modele ONNX absent: %s", onnx_path)
        return None
    if not vocab_path.exists():
        logger.info("Vocab absent: %s", vocab_path)
        return None

    # Import dynamique de inference_onnx_v2.py (pas un package installe)
    import sys
    inference_dir_str = str(inference_dir)
    if inference_dir_str not in sys.path:
        sys.path.insert(0, inference_dir_str)

    try:
        from inference_onnx_v2 import OnnxInferenceEngineV2
    except ImportError:
        logger.warning("Impossible d'importer inference_onnx_v2")
        return None
    finally:
        if inference_dir_str in sys.path:
            sys.path.remove(inference_dir_str)

    lexicon_arg = str(lexicon_path) if lexicon_path.exists() else None

    engine = OnnxInferenceEngineV2(
        str(onnx_path), str(vocab_path),
        lexicon_path=lexicon_arg,
    )

    logger.info("G2P Unifie V2 charge via copie locale")
    return G2PUnifieAdapter(engine)
