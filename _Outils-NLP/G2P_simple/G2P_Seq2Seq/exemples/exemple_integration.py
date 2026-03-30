"""Exemple d'integration — G2P Lectura (backend Seq2Seq).

Montre comment integrer le G2P dans un pipeline existant :
  - Traitement par lot (batch)
  - Export JSON / CSV
  - Combinaison avec un POS tagger
"""

from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lectura_g2p import LecturaG2P

HERE = Path(__file__).parent.parent / "modele"

# ── Constructeur direct ──

g2p = LecturaG2P(HERE / "g2p_seq2seq_encoder_int8.onnx",
                  decoder_path=HERE / "g2p_seq2seq_decoder_int8.onnx",
                  vocab_path=HERE / "g2p_seq2seq_vocab.json",
                  corrections_path=HERE / "g2p_corrections_seq2seq.json")
print(f"Backend : {g2p.backend}\n")


# ── Traitement par lot ──

phrases = [
    "Le chat mange la souris",
    "Bonjour comment allez-vous",
    "Les enfants jouent dans la cour",
    "Il fait beau aujourd'hui",
]

print("=== Traitement par lot ===\n")
for phrase in phrases:
    mots = phrase.split()
    phones = g2p.predict_batch(mots)
    ipa = " ".join(phones)
    print(f"  {phrase}")
    print(f"    → /{ipa}/\n")

# ── Export JSON ──

print("=== Export JSON ===\n")
texte = "Le petit chat noir"
mots = texte.split()
export = [
    {"mot": mot, "ipa": g2p.predict(mot)}
    for mot in mots
]
json_output = json.dumps(export, ensure_ascii=False, indent=2)
print(json_output)
print()

# ── Export CSV ──

print("=== Export CSV (stdout) ===\n")
writer = csv.writer(sys.stdout)
writer.writerow(["mot", "ipa"])
for mot in "extraordinaire anticonstitutionnellement".split():
    writer.writerow([mot, g2p.predict(mot)])

# ── Combinaison avec POS tagger (si disponible) ──

print("\n=== G2P + POS (si POS tagger disponible) ===\n")

# Simuler des resultats POS (normalement viendraient d'un tagger)
tagged_words = [
    ("est", "AUX"),
    ("est", "NOM"),
    ("plus", "ADV"),
    ("plus", "NOM"),
]

for mot, pos in tagged_words:
    phone = g2p.predict(mot, pos=pos)
    print(f"  {mot:8} ({pos:4}) → /{phone}/")
