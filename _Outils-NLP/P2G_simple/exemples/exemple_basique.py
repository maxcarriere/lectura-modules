#!/usr/bin/env python3
"""Exemple basique : utilisation de Lectura P2G.

Fonctionne sans onnxruntime (table + regles uniquement).
Si un modele est present (CRF, BiLSTM, Seq2Seq), il sera utilise automatiquement.

Usage :
    PYTHONPATH=.. python exemples/exemple_basique.py
"""

import sys
from pathlib import Path

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lectura_p2g import LecturaP2G

# ── Initialisation (table embarquee, zero dependance) ──
p2g = LecturaP2G()
print(f"Backend: {p2g.backend}  |  Modele: {'oui' if p2g.has_model else 'non'}")

# ── Prediction simple ──
print("\n=== Prediction simple ===")
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

# ── Candidates (sans modele, retourne un seul resultat) ──
print("\n=== Candidates ===")
candidates = p2g.predict_candidates("pɛʃœʁ", k=5)
print(f"  /pɛʃœʁ/ → {candidates}")
if not p2g.has_model:
    print("  (Un seul candidat sans modele ML)")
