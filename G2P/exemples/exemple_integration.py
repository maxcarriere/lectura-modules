#!/usr/bin/env python3
"""Exemple d'intégration : pipeline complet avec liaisons et corrections.

Usage :
    python exemples/exemple_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ajouter le package au path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from lectura_nlp.inference_numpy import NumpyInferenceEngine
from lectura_nlp.tokeniseur import tokeniser
from lectura_nlp.posttraitement import (
    appliquer_liaison,
    charger_corrections,
    charger_homographes,
    corriger_g2p,
)

# Chemins
MODELS_DIR = _ROOT / "modeles"
DATA_DIR = _ROOT / "src" / "lectura_nlp" / "data"

# 1. Créer le moteur d'inférence
engine = NumpyInferenceEngine(
    MODELS_DIR / "unifie_weights.json",
    MODELS_DIR / "unifie_vocab.json",
)

# 2. Charger les corrections G2P (optionnel, améliore la précision)
charger_corrections(DATA_DIR / "g2p_corrections_unifie.json")

# 2b. Charger la table d'homographes (POS-aware, prioritaire sur corrections)
charger_homographes(DATA_DIR / "homographes.json")

# 3. Analyser plusieurs phrases
phrases = [
    "Les enfants sont arrivés à la maison.",
    "Un petit animal courait dans les bois.",
    "Ils ont été très heureux de vous revoir.",
]

for phrase in phrases:
    tokens = tokeniser(phrase)
    result = engine.analyser(tokens)

    # 4. Appliquer les corrections G2P (POS-aware)
    g2p = [
        corriger_g2p(tok, result["g2p"][i], result["pos"][i] if i < len(result.get("pos", [])) else None)
        for i, tok in enumerate(tokens)
        if i < len(result["g2p"])
    ]

    # 5. Appliquer les liaisons
    ipa_final = appliquer_liaison(tokens, g2p, result["liaison"])

    # 6. Afficher le résultat
    print(f"Phrase : {phrase}")
    print(f"IPA    : {' '.join(g2p)}")
    print(f"Liaison: {' '.join(ipa_final)}")
    print(f"POS    : {' '.join(result['pos'])}")

    # Morphologie
    morpho = result.get("morpho", {})
    for trait in sorted(morpho.keys()):
        vals = morpho[trait]
        non_pad = [(tokens[i], v) for i, v in enumerate(vals) if v != "_" and v != "PAD"]
        if non_pad:
            print(f"  {trait:10s}: {', '.join(f'{t}={v}' for t, v in non_pad)}")

    print()
