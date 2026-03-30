#!/usr/bin/env python3
"""Exemple d'integration : P2G avec modele Seq2Seq et beam search.

Necessite onnxruntime et un modele entraine.

Usage :
    PYTHONPATH=.. python exemples/exemple_integration.py
"""

import sys
from pathlib import Path

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lectura_p2g import LecturaP2G

MODEL_DIR = Path(__file__).resolve().parent.parent / "modele"

# ── Initialisation avec modele ──
p2g = LecturaP2G(model_dir=MODEL_DIR)

if p2g.has_model:
    print("Modele Seq2Seq charge avec succes !")
else:
    print("Modele Seq2Seq non disponible, utilisation table + regles.")
    print(f"  (Cherche dans: {MODEL_DIR})")
    print("  Installez onnxruntime et placez les fichiers ONNX dans modele/")

# ── Prediction simple ──
print("\n=== Prediction ===")
test = ["bɔ̃ʒuʁ", "mɛzɔ̃", "pɛʃœʁ", "kɔ̃stitysjɔ̃", "ɑ̃tikɔ̃stitysjɔnɛləmɑ̃"]
for ipa in test:
    ortho = p2g.predict(ipa)
    print(f"  /{ipa}/ → {ortho}")

# ── Beam search : candidates multiples ──
print("\n=== Beam search (top-5) ===")
ambiguous = [
    "pɛʃœʁ",   # pecheur / pecheurs / pecheure
    "vɛʁ",      # vert / verre / vers / ver
    "sɑ̃",       # sang / sans / cent / s'en
    "o",         # eau / au / aux / oh
    "ʃɑ̃",       # chant / champ / chants
]
for ipa in ambiguous:
    candidates = p2g.predict_candidates(ipa, k=5)
    print(f"\n  /{ipa}/")
    for word, prob in candidates:
        bar = "█" * int(prob * 30)
        print(f"    {word:<20s} {prob:>6.1%} {bar}")

# ── Utilisation comme dictionnaire de desambiguisation ──
print("\n=== Exemple de desambiguisation contextuelle ===")
print("  En combinant P2G avec un modele de langue,")
print("  on peut choisir le bon homophone en contexte :")
print()
candidates = p2g.predict_candidates("vɛʁ", k=5)
mots = [w for w, _ in candidates]
print(f"  Candidates pour /vɛʁ/ : {mots}")
print(f"  Contexte 'un _ de terre' → meilleur candidat parmi : ver, verre, vers...")
