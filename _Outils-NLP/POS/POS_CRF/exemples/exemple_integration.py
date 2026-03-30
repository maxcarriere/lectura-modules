"""Exemple d'intégration — POS Tagger Lectura.

Montre comment intégrer le tagger dans un pipeline existant :
  - Traitement par lot (batch)
  - Filtrage par catégorie grammaticale
  - Export JSON / CSV
"""

from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lectura_pos import PosTagger

MODEL = Path(__file__).parent.parent / "modele" / "pos_model_crf.json"
LEXICON = Path(__file__).parent.parent / "modele" / "mini_lexique.json"

tagger = PosTagger(MODEL, lexicon_path=LEXICON)

# ── Traitement par lot ──

phrases = [
    "Le chat mange la souris.",
    "Je suis allé au marché.",
    "Les enfants jouent dans la cour de l'école.",
    "Il est parti sans dire un mot.",
]

print("=== Traitement par lot ===\n")
all_results = []
for phrase in phrases:
    tagged = tagger.tag(phrase)
    all_results.append({"phrase": phrase, "tokens": tagged})
    print(f"  {phrase}")
    print(f"    → {tagged}\n")

# ── Filtrage par catégorie ──

print("=== Extraction des noms ===\n")
texte = "Le petit chat noir mange la grosse souris grise dans le jardin"
tagged = tagger.tag(texte)
noms = [mot for mot, tag in tagged if tag == "NOM"]
print(f"  Texte : {texte}")
print(f"  Noms  : {noms}\n")

print("=== Extraction des verbes ===\n")
verbes = [mot for mot, tag in tagged if tag in ("VER", "AUX")]
print(f"  Verbes : {verbes}\n")

# ── Export JSON ──

print("=== Export JSON ===\n")
export = tagger.tag_detailed(texte)
json_output = json.dumps(export, ensure_ascii=False, indent=2)
print(json_output[:300] + "…\n")

# ── Export CSV ──

print("=== Export CSV (stdout) ===\n")
writer = csv.writer(sys.stdout)
writer.writerow(["mot", "tag", "description"])
for d in export:
    writer.writerow([d["mot"], d["tag"], d["description"]])
