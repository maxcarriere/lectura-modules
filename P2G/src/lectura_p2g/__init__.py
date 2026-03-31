"""Lectura P2G — Modèle unifié P2G+POS+Morpho pour le français (IPA → orthographe).

Architecture : BiLSTM char-level + word feedback multi-tête (2.56M params, ONNX INT8 = 2.6 Mo)

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md

Trois backends d'inférence au choix :
  - ONNX Runtime  (rapide, ~2ms/phrase)
  - NumPy         (léger, ~50ms/phrase)
  - Pure Python   (zéro dépendance, ~200ms/phrase)

Exemple rapide::

    from lectura_p2g import get_model_path
    from lectura_p2g.inference_onnx import OnnxInferenceEngine

    engine = OnnxInferenceEngine(get_model_path("unifie_p2g_v2_int8.onnx"),
                                  get_model_path("unifie_p2g_v2_vocab.json"))
    result = engine.analyser(["le", "ʃa", "ɛ", "bɔ̃"])
    print(result["ortho"])  # ['le', 'chat', 'est', 'bon']
"""

from pathlib import Path

__version__ = "1.0.0"

_MODELES_DIR = Path(__file__).parent / "modeles"


def get_model_path(filename: str) -> Path:
    """Retourne le chemin absolu vers un fichier modele embarque."""
    return _MODELES_DIR / filename


# API publique
from lectura_p2g.tokeniseur import tokeniser_ipa, ipa_phrase_vers_chars
from lectura_p2g.posttraitement import corriger_p2g, corriger_phrase_v2
