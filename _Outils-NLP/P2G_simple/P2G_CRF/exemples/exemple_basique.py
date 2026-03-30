#!/usr/bin/env python3
"""Exemple basique : utilisation de Lectura P2G (backend CRF).

Fonctionne sans modele CRF (table + regles uniquement).

Usage :
    PYTHONPATH=.. python exemples/exemple_basique.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lectura_p2g import LecturaP2G

# ── Initialisation (table embarquee, zero dependance) ──
p2g = LecturaP2G()

# ── Prediction simple ──
print("=== Prediction simple ===")
mots = ["bɔ̃ʒuʁ", "mɛzɔ̃", "ʃa", "o", "wazo"]
for ipa in mots:
    ortho = p2g.predict(ipa)
    print(f"  /{ipa}/ → {ortho}")

# ── Prediction par syllabe ──
print("\n=== Prediction par syllabe ===")
syllabes = ["kɑ̃", "ty", "ʃɑ̃", "tə"]
for syl in syllabes:
    ortho = p2g.predict_syllable(syl)
    print(f"  /{syl}/ → {ortho}")

# ── Verification du mode ──
print(f"\nModele CRF charge : {p2g.has_model}")
print(f"Backend : {p2g.backend}")
