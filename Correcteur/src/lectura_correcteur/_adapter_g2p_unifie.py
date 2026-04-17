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


class G2PUnifieAdapter:
    """Adaptateur satisfaisant TaggerProtocol + G2PProtocol.

    Appelle OnnxInferenceEngineV2.analyser_v2() pour POS/Morpho/G2P
    en un seul forward pass ONNX (1.8 Mo int8).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def _analyser_cached(self, words: list[str]) -> dict[str, Any]:
        """Appelle analyser_v2 avec cache par tuple de mots."""
        key = tuple(words)
        if key not in self._cache:
            self._cache[key] = self._engine.analyser_v2(list(words))
        return self._cache[key]

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
        (meme format que MorphoTagger).
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
            morpho = result.get("morpho", {})
            for feat in ("genre", "nombre", "temps", "mode", "personne"):
                vals = morpho.get(feat, [])
                if i < len(vals) and vals[i] != "_":
                    d[feat] = vals[i]
            tags.append(d)

        return tags

    # -- G2PProtocol --

    def g2p(self, word: str) -> str:
        """Retourne l'IPA d'un mot via le modele unifie V2."""
        result = self._analyser_cached([word])
        g2p_list = result.get("g2p", [])
        return g2p_list[0] if g2p_list else ""


def creer_adapter_g2p_unifie(
    model_dir: str | Path | None = None,
) -> G2PUnifieAdapter | None:
    """Factory : cree un G2PUnifieAdapter si les fichiers sont disponibles.

    Retourne None si onnxruntime n'est pas installe ou si les fichiers
    modele sont absents (degradation gracieuse).

    Args:
        model_dir: Repertoire racine de lectura-modules-private.
            Par defaut : Modules/lectura-modules-private/ relatif au workspace.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.info("onnxruntime non installe — G2P Unifie V2 indisponible")
        return None

    if model_dir is None:
        # Remonter depuis ce fichier vers le workspace
        # Ce fichier : Modules/Correcteur/src/lectura_correcteur/_adapter_g2p_unifie.py
        # Workspace : 4 niveaux au-dessus
        workspace = Path(__file__).resolve().parents[4]
        model_dir = workspace / "Modules" / "lectura-modules-private"

    model_dir = Path(model_dir)

    onnx_path = model_dir / "G2P" / "modeles" / "unifie_v2_int8.onnx"
    vocab_path = model_dir / "G2P" / "modeles" / "unifie_v2_vocab.json"
    lexicon_path = model_dir / "donnees" / "lexique_pos_candidates.json"
    inference_dir = model_dir / "G2P"

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

    logger.info("G2P Unifie V2 charge avec succes")
    return G2PUnifieAdapter(engine)
