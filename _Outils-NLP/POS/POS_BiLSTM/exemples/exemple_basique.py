"""Exemple basique — POS Tagger Lectura (BiLSTM).

Montre les trois façons d'utiliser le tagger.
"""

from pathlib import Path
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lectura_pos import PosTagger

MODEL = Path(__file__).parent.parent / "modele" / "pos_model_bilstm_int8.onnx"
VOCAB = Path(__file__).parent.parent / "modele" / "pos_vocab_bilstm.json"
LEXICON = Path(__file__).parent.parent / "modele" / "mini_lexique.json"

tagger = PosTagger(MODEL, vocab_path=VOCAB, lexicon_path=LEXICON)

# ── 1. tag() → liste de tuples ──

print("=== tag() ===")
result = tagger.tag("Le chat mange la souris")
for mot, pos in result:
    print(f"  {mot} → {pos}")

# ── 2. tag_detailed() → liste de dicts ──

print("\n=== tag_detailed() ===")
details = tagger.tag_detailed("Je suis allé au marché")
for d in details:
    print(f"  {d['mot']:12} {d['tag']:8} ({d['description']})")

# ── 3. tag_formatted() → texte lisible ──

print("\n=== tag_formatted() ===")
print(tagger.tag_formatted("Les enfants jouent dans la cour"))

# ── 4. tag_words() → mots déjà tokenisés ──

print("\n=== tag_words() ===")
words = ["Il", "est", "parti", "sans", "dire", "un", "mot"]
result = tagger.tag_words(words)
for mot, pos in result:
    print(f"  {mot} → {pos}")
