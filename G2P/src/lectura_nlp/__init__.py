"""Lectura NLP — Modèle unifié G2P+POS+Morpho+Liaison pour le français.

Architecture : BiLSTM char-level + multi-tête (1.75M params, ONNX INT8 = 1.8 Mo)

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md

Trois backends d'inférence au choix :
  - ONNX Runtime  (rapide, ~2ms/phrase)
  - NumPy         (léger, ~50ms/phrase)
  - Pure Python   (zéro dépendance, ~200ms/phrase)

Exemple rapide::

    from lectura_nlp import get_model_path
    from lectura_nlp.inference_onnx import OnnxInferenceEngine
    from lectura_nlp.tokeniseur import tokeniser

    engine = OnnxInferenceEngine(get_model_path("unifie_int8.onnx"),
                                  get_model_path("unifie_vocab.json"))
    result = engine.analyser(tokeniser("Les enfants jouent."))
"""

from pathlib import Path

__version__ = "1.1.0"

_MODELES_DIR = Path(__file__).parent / "modeles"


def get_model_path(filename: str) -> Path:
    """Retourne le chemin absolu vers un fichier modele embarque."""
    return _MODELES_DIR / filename


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
    pass  # lectura_formules non disponible — pipeline_formules désactivé
