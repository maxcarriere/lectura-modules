"""Lectura NLP — Modele unifie G2P+POS+Morpho+Liaison pour le francais.

Architecture : BiLSTM char-level + multi-tete (1.75M params, ONNX INT8 = 1.8 Mo)

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md

Quatre backends d'inference au choix :
  - ONNX Runtime  (rapide, ~2ms/phrase)   — necessite modeles locaux
  - NumPy         (leger, ~50ms/phrase)    — necessite modeles locaux
  - Pure Python   (zero dependance)        — necessite modeles locaux
  - API           (serveur Lectura)        — mode par defaut (Niveau 1)

Exemple rapide (mode API, zero config)::

    from lectura_nlp import creer_engine
    engine = creer_engine()
    result = engine.analyser(["bonjour", "monde"])

Exemple avec backend local::

    from lectura_nlp import creer_engine
    engine = creer_engine(mode="local")
    result = engine.analyser(["bonjour", "monde"])
"""

from pathlib import Path

__version__ = "3.0.1"

_MODELES_DIR = Path(__file__).parent / "modeles"


def get_model_path(filename: str) -> Path:
    """Retourne le chemin absolu vers un fichier modele embarque."""
    return _MODELES_DIR / filename


def _modeles_locaux() -> bool:
    """Verifie si les modeles locaux sont disponibles."""
    return (_MODELES_DIR / "unifie_v2_int8.onnx").exists() or (
        _MODELES_DIR / "unifie_v2_vocab.json"
    ).exists()


def _resoudre_lexique(lexicon_path: str | Path | None) -> str | Path | dict | None:
    """Cascade de resolution du lexique pour les lex_features.

    1. Chemin explicite (si passe)
    2. Fichier JSON dans modeles/
    3. lectura_lexique installe -> generer le dict
    4. None (pas de lexique -> lex_features = zeros)
    """
    if lexicon_path is not None:
        return lexicon_path
    json_path = _MODELES_DIR / "lexique_pos_candidates.json"
    if json_path.exists():
        return json_path
    try:
        from lectura_lexique import Lexique
        lex = Lexique()
        result: dict[str, list[str]] = {}
        for mot, entrees in lex._index.items():
            pos_set: set[str] = set()
            for e in entrees:
                if hasattr(e, "pos") and e.pos:
                    pos_set.add(e.pos)
            if pos_set:
                result[mot] = sorted(pos_set)
        return result
    except (ImportError, Exception):
        pass
    return None


def creer_engine(
    mode: str = "auto",
    api_url: str | None = None,
    api_key: str | None = None,
    lexicon_path: str | Path | None = None,
):
    """Factory pour creer un engine d'inference G2P.

    Parameters
    ----------
    mode : str
        "auto" (defaut) : local si modeles presents, sinon API
        "local" : force le mode local (ONNX > NumPy > Pure)
        "api" : force le mode API
        "onnx", "numpy", "pure" : force un backend local specifique
    api_url : str | None
        URL du serveur Lectura (pour mode API)
    api_key : str | None
        Cle API (pour mode API)
    lexicon_path : str | Path | None
        Chemin vers le fichier lexique POS (JSON).
        Si None, resolution automatique (cascade).
    """
    if mode == "api":
        from lectura_nlp.inference_api import ApiInferenceEngine
        return ApiInferenceEngine(api_url=api_url, api_key=api_key)

    if mode == "auto" and not _modeles_locaux():
        from lectura_nlp.inference_api import ApiInferenceEngine
        return ApiInferenceEngine(api_url=api_url, api_key=api_key)

    # Mode local — essayer les backends dans l'ordre de preference
    model_onnx = get_model_path("unifie_v2_int8.onnx")
    model_vocab = get_model_path("unifie_v2_vocab.json")
    resolved_lexicon = _resoudre_lexique(lexicon_path)

    if mode in ("auto", "local", "onnx"):
        try:
            from lectura_nlp.inference_onnx_v2 import OnnxInferenceEngineV2
            if isinstance(resolved_lexicon, dict):
                return OnnxInferenceEngineV2(
                    str(model_onnx), str(model_vocab),
                    lexicon=resolved_lexicon,
                )
            return OnnxInferenceEngineV2(
                str(model_onnx), str(model_vocab),
                lexicon_path=str(resolved_lexicon) if resolved_lexicon else None,
            )
        except (ImportError, FileNotFoundError, Exception):
            if mode == "onnx":
                raise

    if mode in ("auto", "local", "numpy"):
        try:
            from lectura_nlp.inference_numpy import NumpyInferenceEngine
            weights_dir = Path(__file__).parent / "modeles_numpy"
            if not weights_dir.exists():
                weights_dir = _MODELES_DIR
            return NumpyInferenceEngine(
                str(weights_dir), str(model_vocab),
                lexicon_path=str(resolved_lexicon) if isinstance(resolved_lexicon, Path) else None,
                lexicon=resolved_lexicon if isinstance(resolved_lexicon, dict) else None,
            )
        except (ImportError, FileNotFoundError, Exception):
            if mode == "numpy":
                raise

    if mode in ("auto", "local", "pure"):
        try:
            from lectura_nlp.inference_pure import PureInferenceEngine
            weights_dir = Path(__file__).parent / "modeles_numpy"
            if not weights_dir.exists():
                weights_dir = _MODELES_DIR
            return PureInferenceEngine(
                str(weights_dir), str(model_vocab),
                lexicon_path=str(resolved_lexicon) if isinstance(resolved_lexicon, Path) else None,
                lexicon=resolved_lexicon if isinstance(resolved_lexicon, dict) else None,
            )
        except (ImportError, FileNotFoundError, Exception):
            if mode == "pure":
                raise

    raise RuntimeError(
        f"Aucun backend d'inference disponible (mode={mode!r}). "
        "Verifiez que les modeles sont installes ou utilisez mode='api'."
    )


# API publique
from lectura_nlp.tokeniseur import tokeniser, phrase_vers_chars
from lectura_nlp.posttraitement import (
    appliquer_liaison,
    appliquer_regles_g2p,
    charger_corrections,
    charger_homographes,
    corriger_g2p,
)
try:
    from lectura_nlp.pipeline_formules import (
        analyser_phrase_complete,
        ResultatPhraseG2P,
        MotAnalyseG2P,
    )
except ImportError:
    pass  # lectura_formules non disponible — pipeline_formules desactive
