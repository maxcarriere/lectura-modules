#!/usr/bin/env python3
"""Exemple basique : G2P + POS sur une phrase.

Usage :
    python exemples/exemple_basique.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ajouter le package au path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from lectura_nlp.inference_numpy import NumpyInferenceEngine
from lectura_nlp.tokeniseur import tokeniser

# Chemins des fichiers modèle
MODELS_DIR = _ROOT / "modeles"

# Créer le moteur d'inférence NumPy
engine = NumpyInferenceEngine(
    MODELS_DIR / "unifie_weights.json",
    MODELS_DIR / "unifie_vocab.json",
)

# Analyser une phrase
phrase = "Les enfants jouent dans le jardin."
tokens = tokeniser(phrase)
result = engine.analyser(tokens)

# Afficher les résultats
print(f"Phrase : {phrase}")
print(f"Tokens : {tokens}")
print()
print(f"{'Mot':15s} {'IPA':15s} {'POS':10s}")
print("-" * 40)
for i, tok in enumerate(tokens):
    ipa = result["g2p"][i] if i < len(result["g2p"]) else ""
    pos = result["pos"][i] if i < len(result["pos"]) else ""
    print(f"{tok:15s} {ipa:15s} {pos:10s}")
