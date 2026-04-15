#!/usr/bin/env python3
"""Demo en ligne de commande du Syllabeur Complet Lectura.

Usage :
    python demo_cli.py "chocolat"
    python demo_cli.py "Le chat mange la souris"
    python demo_cli.py                              # mode interactif
    python demo_cli.py --ipa "ʃɔkɔla"              # entree IPA directe
    python demo_cli.py --simple "extraordinaire"    # affichage simplifie
    python demo_cli.py --groupes                    # demo groupes de lecture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lectura_aligneur import (
    LecturaSyllabeur, MotAnalyse, OptionsGroupes,
)


def _demo_groupes(syl: LecturaSyllabeur) -> None:
    """Demonstration des groupes de lecture avec exemples pre-programmes."""
    from dataclasses import dataclass

    @dataclass
    class FakeToken:
        text: str
        span: tuple

    exemples = [
        (
            "les enfants jouent (liaison)",
            [
                MotAnalyse(token=FakeToken("les", (0, 3)), phone="lez", liaison="Lz"),
                MotAnalyse(token=FakeToken("enfants", (4, 11)), phone="ɑ̃fɑ̃"),
                MotAnalyse(token=FakeToken("jouent", (12, 18)), phone="ʒu"),
            ],
        ),
        (
            "avec elle (enchainement)",
            [
                MotAnalyse(token=FakeToken("avec", (0, 4)), phone="avɛk"),
                MotAnalyse(token=FakeToken("elle", (5, 9)), phone="ɛl"),
            ],
        ),
        (
            "le chat dort (pas de liaison)",
            [
                MotAnalyse(token=FakeToken("le", (0, 2)), phone="lə"),
                MotAnalyse(token=FakeToken("chat", (3, 7)), phone="ʃa"),
                MotAnalyse(token=FakeToken("dort", (8, 12)), phone="dɔʁ"),
            ],
        ),
    ]

    for desc, mots in exemples:
        print(f"--- {desc} ---")
        r = syl.analyser_complet(mots)
        print(r.format_detail())
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lectura Syllabeur Complet — Demo CLI")
    parser.add_argument("text", nargs="*", help="Mot ou phrase a analyser")
    parser.add_argument(
        "--ipa", action="store_true",
        help="Traiter l'entree comme de l'IPA (syllabation directe)",
    )
    parser.add_argument(
        "--simple", action="store_true",
        help="Affichage simplifie (une ligne par mot)",
    )
    parser.add_argument(
        "--groupes", action="store_true",
        help="Demo des groupes de lecture (exemples pre-programmes)",
    )
    args = parser.parse_args()

    syl = LecturaSyllabeur()

    if args.groupes:
        _demo_groupes(syl)
        return

    if args.ipa and args.text:
        phone = " ".join(args.text)
        sylls = syl.syllabify_ipa(phone)
        print(f"/{phone}/ -> {'.'.join(sylls)} ({len(sylls)} syll.)")
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
    print("+" + "=" * 54 + "+")
    print("|   Lectura Syllabeur Complet                          |")
    print("|   Analyse syllabique + groupes de lecture             |")
    print("|   Tapez un mot ou une phrase, 'q' pour quitter.      |")
    print("|   Prefixez par / pour de l'IPA direct.               |")
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

        if text.startswith("/"):
            # Mode IPA direct
            phone = text[1:].strip()
            sylls = syl.syllabify_ipa(phone)
            print(f"  /{phone}/ -> {'.'.join(sylls)} ({len(sylls)} syll.)")
        else:
            results = syl.analyze_text(text)
            for r in results:
                print(r.format_detail())
        print()


if __name__ == "__main__":
    main()
