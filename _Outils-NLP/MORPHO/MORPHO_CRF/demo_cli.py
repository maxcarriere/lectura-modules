#!/usr/bin/env python3
"""Démo en ligne de commande de l'analyseur morphologique Lectura (CRF).

Usage :
    python demo_cli.py "Les chats mangent les souris"
    python demo_cli.py                              # mode interactif
"""

from __future__ import annotations

import sys
from pathlib import Path

from lectura_morpho import MorphoTagger

MODEL_PATH = Path(__file__).parent / "modele" / "morpho_model_crf.json"
LEXICON_PATH = Path(__file__).parent / "modele" / "glaff_lookup.json"
MINI_LEX_PATH = (
    Path(__file__).parent.parent.parent / "POS" / "POS_CRF" / "modele" / "mini_lexique.json"
)


def main() -> None:
    lexicon = LEXICON_PATH if LEXICON_PATH.exists() else None
    mini_lex = MINI_LEX_PATH if MINI_LEX_PATH.exists() else None
    tagger = MorphoTagger(MODEL_PATH, lexicon_path=lexicon, mini_lexicon_path=mini_lex)

    # Mode argument
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(tagger.tag_formatted(text))
        return

    # Mode interactif
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Lectura Morpho Tagger (CRF) — Analyse morpho.        ║")
    print("║   Tapez une phrase, ou 'q' pour quitter.                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text or text.lower() in ("q", "quit", "exit"):
            break

        print()
        print(tagger.tag_formatted(text))
        print()


if __name__ == "__main__":
    main()
