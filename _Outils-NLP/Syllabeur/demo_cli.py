#!/usr/bin/env python3
"""Demo en ligne de commande du Syllabeur Lectura.

Usage :
    python demo_cli.py "chocolat"
    python demo_cli.py "Le chat mange la souris"
    python demo_cli.py                              # mode interactif
    python demo_cli.py --ipa "ʃɔkɔla"              # entrée IPA directe
"""

from __future__ import annotations

import argparse
import sys

from lectura_syllabeur import LecturaSyllabeur


def main() -> None:
    parser = argparse.ArgumentParser(description="Lectura Syllabeur — Demo CLI")
    parser.add_argument("text", nargs="*", help="Mot ou phrase a analyser")
    parser.add_argument(
        "--ipa", action="store_true",
        help="Traiter l'entree comme de l'IPA (syllabation directe)",
    )
    parser.add_argument(
        "--simple", action="store_true",
        help="Affichage simplifie (une ligne par mot)",
    )
    args = parser.parse_args()

    syl = LecturaSyllabeur()

    if args.ipa and args.text:
        phone = " ".join(args.text)
        sylls = syl.syllabify_ipa(phone)
        print(f"/{phone}/ → {'.'.join(sylls)} ({len(sylls)} syll.)")
        return

    # Mode argument
    if args.text:
        text = " ".join(args.text)
        results = syl.analyze_text(text)
        for r in results:
            if args.simple:
                print(r.format_simple())
            else:
                print(r.format_detail())
                print()
        return

    # Mode interactif
    print("╔══════════════════════════════════════════════════╗")
    print("║   Lectura Syllabeur — Analyse syllabique        ║")
    print("║   Tapez un mot ou une phrase, 'q' pour quitter. ║")
    print("║   Préfixez par / pour de l'IPA direct.          ║")
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

        if text.startswith("/"):
            # Mode IPA direct
            phone = text[1:].strip()
            sylls = syl.syllabify_ipa(phone)
            print(f"  /{phone}/ → {'.'.join(sylls)} ({len(sylls)} syll.)")
        else:
            results = syl.analyze_text(text)
            for r in results:
                print(r.format_detail())
        print()


if __name__ == "__main__":
    main()
