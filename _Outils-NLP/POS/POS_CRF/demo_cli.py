#!/usr/bin/env python3
"""Démo en ligne de commande du POS Tagger Lectura.

Usage :
    python demo_cli.py "Le chat mange la souris"
    python demo_cli.py                              # mode interactif
"""

from __future__ import annotations

import sys
from pathlib import Path

from lectura_pos import PosTagger

MODEL_PATH = Path(__file__).parent / "modele" / "pos_model_crf.json"
LEXICON_PATH = Path(__file__).parent / "modele" / "mini_lexique.json"


def main() -> None:
    tagger = PosTagger(MODEL_PATH, lexicon_path=LEXICON_PATH)

    # Mode argument
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(tagger.tag_formatted(text))
        return

    # Mode interactif
    print("╔══════════════════════════════════════════════════╗")
    print("║   Lectura POS Tagger — Étiqueteur grammatical   ║")
    print("║   Tapez une phrase, ou 'q' pour quitter.        ║")
    print("╚══════════════════════════════════════════════════╝")
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
