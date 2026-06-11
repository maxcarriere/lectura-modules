#!/usr/bin/env python3
"""Diagnostic rapide : comprendre les FP accord.nombre_nom."""
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
from lectura_correcteur import CorrecteurV5, CorrecteurV5Config

lexique = LexiqueLite(LEXIQUE_DB)
cfg = CorrecteurV5Config()
correcteur = CorrecteurV5(lexique, config=cfg)

phrases = [
    "Elle envoyait des messages à Marc.",
    "Il répondait par des accusations.",
    "Il a cité des noms.",
    "Des trucs très précis.",
    "Elle avait cessé de chercher des explications.",
    "Des mots qui voulaient dire quelque chose.",
    "Des récits, des personnages, une intrigue.",
    "Les religions sont des récits.",
]

for phrase in phrases:
    print(f"\n{'='*70}")
    print(f"ORIGINAL: {phrase}")
    r = correcteur.corriger(phrase)
    print(f"CORRIGE : {r.phrase_corrigee}")

    # Toutes les corrections
    if r.corrections:
        for c in r.corrections:
            print(f"  [{c.regle}] {c.original!r} -> {c.corrige!r} | {c.explication}")

    # Mots analyses
    print("  MOTS:")
    for m in r.mots:
        pos = m.pos if hasattr(m, 'pos') else '?'
        morpho = ""
        if hasattr(m, 'nombre'):
            morpho += f" nb={m.nombre}"
        if hasattr(m, 'genre'):
            morpho += f" g={m.genre}"
        if hasattr(m, 'morpho') and m.morpho:
            morpho = f" morpho={m.morpho}"
        print(f"    {m.forme:20s} pos={pos:10s}{morpho}")
