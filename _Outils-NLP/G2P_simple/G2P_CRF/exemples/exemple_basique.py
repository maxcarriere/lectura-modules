"""Exemple basique — G2P Lectura (backend CRF).

Montre les differentes facons d'utiliser le convertisseur.
"""

from pathlib import Path
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lectura_g2p import LecturaG2P

HERE = Path(__file__).parent.parent / "modele"

# ── 1. Backend CRF (zero dependance) ──

print("=== Backend CRF ===")
g2p = LecturaG2P(HERE / "g2p_model_crf.json",
                  corrections_path=HERE / "g2p_corrections_crf.json")

mots = ["bonjour", "maison", "chat", "chien", "ordinateur"]
for mot in mots:
    phone = g2p.predict(mot)
    print(f"  {mot:15} → /{phone}/")

# ── 2. predict_batch() → liste de transcriptions ──

print("\n=== predict_batch() ===")
phrase = ["Le", "chat", "mange", "la", "souris"]
phones = g2p.predict_batch(phrase)
for mot, phone in zip(phrase, phones):
    print(f"  {mot:10} → /{phone}/")

# ── 3. predict() avec POS (desambiguisation) ──

print("\n=== predict() avec POS ===")
print(f"  plus (ADV)  → /{g2p.predict('plus', pos='ADV')}/")
print(f"  plus (NOM)  → /{g2p.predict('plus', pos='NOM')}/")

# ── 4. predict_formatted() → affichage lisible ──

print("\n=== predict_formatted() ===")
for mot in ["francais", "beautiful", "extraordinaire"]:
    print(f"  {g2p.predict_formatted(mot)}")
