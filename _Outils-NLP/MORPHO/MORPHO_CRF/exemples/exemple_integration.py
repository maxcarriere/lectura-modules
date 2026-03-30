"""Exemple d'intégration — Analyseur morphologique Lectura (CRF).

Montre comment intégrer le tagger morpho dans un pipeline existant :
  - Traitement par lot (batch)
  - Filtrage par catégorie grammaticale et traits
  - Vérification d'accords (genre/nombre)
  - Export JSON / CSV
"""

from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lectura_morpho import MorphoTagger

MODEL = Path(__file__).parent.parent / "modele" / "morpho_model_crf.json"
LEXICON = Path(__file__).parent.parent / "modele" / "glaff_lookup.json"
MINI_LEX = (
    Path(__file__).parent.parent.parent.parent / "POS" / "POS_CRF" / "modele" / "mini_lexique.json"
)

lexicon = LEXICON if LEXICON.exists() else None
mini_lex = MINI_LEX if MINI_LEX.exists() else None
tagger = MorphoTagger(MODEL, lexicon_path=lexicon, mini_lexicon_path=mini_lex)

# ── Traitement par lot ──

phrases = [
    "Le chat mange la souris.",
    "Je suis allé au marché.",
    "Les enfants jouent dans la cour de l'école.",
    "Il est parti sans dire un mot.",
]

print("=== Traitement par lot ===\n")
for phrase in phrases:
    tagged = tagger.tag(phrase)
    print(f"  {phrase}")
    for r in tagged:
        print(f"    {r['mot']:12} → {r['tag_complet']:20} lemme={r['lemme']}")
    print()

# ── Filtrage par catégorie ──

print("=== Extraction des noms et leurs traits ===\n")
texte = "Le petit chat noir mange la grosse souris grise dans le jardin"
tagged = tagger.tag(texte)
noms = [r for r in tagged if r["pos"] == "NOM"]
print(f"  Texte : {texte}")
for n in noms:
    print(f"    {n['mot']:12} genre={n['genre']!s:5} nombre={n['nombre']!s:5} lemme={n['lemme']}")
print()

# ── Extraction des verbes conjugués ──

print("=== Extraction des verbes conjugués ===\n")
verbes = [r for r in tagged if r["pos"] in ("VER", "AUX") and r["mode"] is not None]
for v in verbes:
    print(f"    {v['mot']:12} mode={v['mode']!s:5} temps={v['temps']!s:5} "
          f"pers={v['personne']!s:3} nb={v['nombre']!s:5} lemme={v['lemme']}")
print()

# ── Vérification d'accords (exemple simple) ──

print("=== Vérification d'accords (DET + NOM) ===\n")
texte2 = "Les chat mange les souris"
tagged2 = tagger.tag(texte2)
for i in range(len(tagged2) - 1):
    curr = tagged2[i]
    next_ = tagged2[i + 1]
    if curr["pos"].startswith("ART") and next_["pos"] == "NOM":
        if curr["nombre"] and next_["nombre"] and curr["nombre"] != next_["nombre"]:
            print(f"  ACCORD SUSPECT : '{curr['mot']}' ({curr['nombre']}) "
                  f"+ '{next_['mot']}' ({next_['nombre']})")
        else:
            print(f"  OK : '{curr['mot']}' + '{next_['mot']}' "
                  f"({curr['nombre'] or '?'}/{next_['nombre'] or '?'})")
print()

# ── Export JSON ──

print("=== Export JSON ===\n")
export = tagger.tag(texte)
json_output = json.dumps(export, ensure_ascii=False, indent=2)
print(json_output[:400] + "...\n")

# ── Export CSV ──

print("=== Export CSV (stdout) ===\n")
writer = csv.writer(sys.stdout)
writer.writerow(["mot", "pos", "tag_complet", "genre", "nombre", "temps", "mode", "personne", "lemme"])
for r in export:
    writer.writerow([r["mot"], r["pos"], r["tag_complet"],
                     r["genre"] or "", r["nombre"] or "",
                     r["temps"] or "", r["mode"] or "",
                     r["personne"] or "", r["lemme"]])
