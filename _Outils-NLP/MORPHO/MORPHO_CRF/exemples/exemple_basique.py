"""Exemple basique — Analyseur morphologique Lectura (CRF).

Montre les trois façons d'utiliser le tagger morphologique.
"""

from pathlib import Path
import sys

# Ajouter le dossier parent au path
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

# ── 1. tag() → liste de dicts ──

print("=== tag() ===")
result = tagger.tag("Le chat mange la souris")
for r in result:
    print(f"  {r['mot']:12} {r['tag_complet']:20} → {r['lemme']}")

# ── 2. tag_formatted() → texte lisible ──

print("\n=== tag_formatted() ===")
print(tagger.tag_formatted("Les enfants jouent dans la cour"))

# ── 3. tag_words() → mots déjà tokenisés ──

print("\n=== tag_words() ===")
words = ["Il", "est", "parti", "sans", "dire", "un", "mot"]
result = tagger.tag_words(words)
for r in result:
    traits = []
    if r["genre"]:
        traits.append(f"genre={r['genre']}")
    if r["nombre"]:
        traits.append(f"nombre={r['nombre']}")
    if r["mode"]:
        traits.append(f"mode={r['mode']}")
    if r["temps"]:
        traits.append(f"temps={r['temps']}")
    if r["personne"]:
        traits.append(f"personne={r['personne']}")
    traits_str = ", ".join(traits) if traits else "-"
    print(f"  {r['mot']:12} {r['pos']:8} {r['lemme']:12} ({traits_str})")
