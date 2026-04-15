#!/usr/bin/env python3
"""Demo en ligne de commande du Tokeniseur Complet Lectura.

Usage :
    python demo_cli.py "L'enfant mange du chocolat."
    python demo_cli.py --words "L'enfant mange du chocolat."
    python demo_cli.py --normalize "L'enfant  mange...du  chocolat"
    python demo_cli.py --formules "Appeler le 06 12 34 56 78 le 15/03/2024."
    python demo_cli.py                                  # mode interactif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lectura_tokeniseur import LecturaTokeniseur


def main() -> None:
    parser = argparse.ArgumentParser(description="Lectura Tokeniseur Complet — Demo CLI")
    parser.add_argument("text", nargs="*", help="Texte a analyser")
    parser.add_argument(
        "--words", action="store_true",
        help="Afficher uniquement la liste des mots",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Afficher uniquement le texte normalise",
    )
    parser.add_argument(
        "--formules", action="store_true",
        help="Afficher uniquement les formules detectees",
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

        if args.formules:
            formules = tok.extract_formules(text)
            if not formules:
                print("Aucune formule detectee.")
            for f in formules:
                print(f"  {f.text:20s}  {f.formule_type.value:15s}  val={f.valeur!r}")
            return

        result = tok.analyze(text)
        print(f"Original   : {result.texte_original}")
        print(f"Normalise  : {result.texte_normalise}")
        print(f"Mots       : {result.nb_mots}")
        print(f"Formules   : {len(result.formules)}")
        print(f"Tokens     : {result.nb_tokens}")
        print()
        print(result.format_table())
        return

    # Mode interactif
    print("+" + "=" * 54 + "+")
    print("|   Lectura Tokeniseur Complet                         |")
    print("|   Normalisation + Tokens + Formules                  |")
    print("|   Tapez un texte, 'q' pour quitter.                  |")
    print("+" + "=" * 54 + "+")
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
        print(f"  Normalise : {result.texte_normalise}")
        print(f"  Mots ({result.nb_mots}) : {result.words()}")
        if result.formules:
            print(f"  Formules ({len(result.formules)}) :")
            for f in result.formules:
                print(f"    {f.text} -> {f.formule_type.value} (val={f.valeur!r})")
        print()
        print(result.format_table())
        print()


if __name__ == "__main__":
    main()
