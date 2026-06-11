#!/usr/bin/env python3
"""Confirmer la theorie : POS-incoherence ART vs ART:ind."""
import os, sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Phonemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Graphemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Tokeniseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/G2P-Pipeline/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Formules/src")

LEXIQUE_DB = os.path.join(
    _PROJECT_ROOT, "src", "lectura_correcteur", "data", "lexique_correcteur.db",
)

from lectura_correcteur._lexique_lite import LexiqueLite
lexique = LexiqueLite(LEXIQUE_DB)

# Verifier les cgrams
for mot in ["des", "du", "les", "le", "la", "un", "une", "ces", "ses"]:
    infos = lexique.info(mot)
    cgrams = {e.get("cgram", "") for e in infos}
    print(f"  {mot:6s} cgrams={cgrams}")

print()

# Verifier le tagger
from lectura_correcteur._tagger_lexique import LexiqueTagger
tagger = LexiqueTagger(lexique)
tags = tagger.tag_words(["Elle", "envoyait", "des", "messages", "à", "Marc"])
for w, t in zip(["Elle", "envoyait", "des", "messages", "à", "Marc"], tags):
    print(f"  Tagger: {w:12s} -> pos={t.get('pos','?'):12s}")

print()

# Test de la condition POS-incoherence
for mot in ["des", "du", "les", "le", "un", "la", "ces", "ses"]:
    infos = lexique.info(mot)
    cgrams = {e.get("cgram", "") for e in infos} if infos else set()
    tag = tagger.tag_words([mot])
    pos = tag[0].get("pos", "")
    match = pos in cgrams
    print(f"  {mot:6s} pos={pos:12s} cgrams={cgrams:50s} MATCH={match}")
