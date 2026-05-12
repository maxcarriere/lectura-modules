#!/usr/bin/env python3
"""Demo en ligne de commande du Tokeniseur Lectura.

Usage :
    python demo_cli.py "L'enfant mange du chocolat."
    python demo_cli.py --words "L'enfant mange du chocolat."
    python demo_cli.py --normalize "L'enfant  mange...du  chocolat"
    python demo_cli.py                                  # mode interactif
"""

from __future__ import annotations

import argparse
import sys

from lectura_tokeniseur import LecturaTokeniseur


def main() -> None:
    parser = argparse.ArgumentParser(description="Lectura Tokeniseur — Demo CLI")
    parser.add_argument("text", nargs="*", help="Texte a analyser")
    parser.add_argument(
        "--words", action="store_true",
        help="Afficher uniquement la liste des mots",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Afficher uniquement le texte normalise",
    )
    args = parser.parse_args()

    tok = LecturaTokeniseur()

    # Mode argument
    if args.text:
        text = " ".join(args.text)

        if args.normalize:
            print(tok.normalize(text))
            return

        if args.words:
            print(" ".join(tok.extract_words(text)))
            return

        result = tok.analyze(text)
        print(f"Original   : {result.texte_original}")
        print(f"Normalisé  : {result.texte_normalise}")
        print(f"Mots       : {result.nb_mots}")
        print(f"Tokens     : {result.nb_tokens}")
        print()
        print(result.format_table())
        return

    # Mode interactif
    print("╔══════════════════════════════════════════════════╗")
    print("║   Lectura Tokeniseur — Normalisation + Tokens   ║")
    print("║   Tapez un texte, 'q' pour quitter.             ║")
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

        result = tok.analyze(text)
        print(f"  Normalisé : {result.texte_normalise}")
        print(f"  Mots ({result.nb_mots}) : {result.words()}")
        print()
        print(result.format_table())
        print()


if __name__ == "__main__":
    main()
