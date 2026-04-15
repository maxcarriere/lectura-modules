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

__version__ = "2.0.0"

_MODELES_DIR = Path(__file__).parent / "modeles"


def get_model_path(filename: str) -> Path:
    """Retourne le chemin absolu vers un fichier modele embarque."""
    return _MODELES_DIR / filename


def _modeles_locaux() -> bool:
    """Verifie si les modeles locaux sont disponibles."""
    return (_MODELES_DIR / "unifie_int8.onnx").exists() or (
        _MODELES_DIR / "unifie_vocab.json"
    ).exists()


def creer_engine(
    mode: str = "auto",
    api_url: str | None = None,
    api_key: str | None = None,
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
    """
    if mode == "api":
        from lectura_nlp.inference_api import ApiInferenceEngine
        return ApiInferenceEngine(api_url=api_url, api_key=api_key)

    if mode == "auto" and not _modeles_locaux():
        from lectura_nlp.inference_api import ApiInferenceEngine
        return ApiInferenceEngine(api_url=api_url, api_key=api_key)

    # Mode local — essayer les backends dans l'ordre de preference
    model_onnx = get_model_path("unifie_int8.onnx")
    model_vocab = get_model_path("unifie_vocab.json")

    if mode in ("auto", "local", "onnx"):
        try:
            from lectura_nlp.inference_onnx import OnnxInferenceEngine
            return OnnxInferenceEngine(str(model_onnx), str(model_vocab))
        except (ImportError, FileNotFoundError, Exception):
            if mode == "onnx":
                raise

    if mode in ("auto", "local", "numpy"):
        try:
            from lectura_nlp.inference_numpy import NumpyInferenceEngine
            weights_dir = Path(__file__).parent / "modeles_numpy"
            if not weights_dir.exists():
                weights_dir = _MODELES_DIR
            return NumpyInferenceEngine(str(weights_dir))
        except (ImportError, FileNotFoundError, Exception):
            if mode == "numpy":
                raise

    if mode in ("auto", "local", "pure"):
        try:
            from lectura_nlp.inference_pure import PureInferenceEngine
            weights_dir = Path(__file__).parent / "modeles_numpy"
            if not weights_dir.exists():
                weights_dir = _MODELES_DIR
            return PureInferenceEngine(str(weights_dir))
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
