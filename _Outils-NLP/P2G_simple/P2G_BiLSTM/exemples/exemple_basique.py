#!/usr/bin/env python3
"""Exemple basique : utilisation de Lectura P2G (backend BiLSTM).

Fonctionne sans modele BiLSTM (table + regles uniquement).

Usage :
    PYTHONPATH=.. python exemples/exemple_basique.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lectura_p2g import LecturaP2G

# -- Initialisation (table embarquee, zero dependance) --
p2g = LecturaP2G()

# -- Prediction simple --
print("=== Prediction simple ===")
mots = ["b\u0254\u0303\u0292u\u0281", "m\u025bz\u0254\u0303",
        "\u0283a", "o", "wazo"]
for ipa in mots:
    ortho = p2g.predict(ipa)
    print(f"  /{ipa}/ \u2192 {ortho}")

# -- Prediction par syllabe --
print("\n=== Prediction par syllabe ===")
syllabes = ["k\u0251\u0303", "ty", "\u0283\u0251\u0303", "t\u0259"]
for syl in syllabes:
    ortho = p2g.predict_syllable(syl)
    print(f"  /{syl}/ \u2192 {ortho}")

# -- Verification du mode --
print(f"\nModele BiLSTM charge : {p2g.has_model}")
print(f"Backend : {p2g.backend}")
