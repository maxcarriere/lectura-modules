#!/usr/bin/env python3
"""Demo en ligne de commande de Lectura Liaisons.

Usage :
    python demo_cli.py                              # exemples intégrés
    python demo_cli.py --check "haricot"            # vérifier h aspiré
    python demo_cli.py --interactive                 # mode interactif
"""

from __future__ import annotations

import argparse
import sys

from lectura_liaisons import (
    LecturaLiaisons,
    MotInfo,
    TokenMot,
    TokenSep,
    GroupeJonction,
    JonctionOptions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lectura Liaisons — Demo CLI")
    parser.add_argument(
        "--check", metavar="MOT",
        help="Vérifier si un mot a un h aspiré",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Mode interactif (saisir des paires de mots)",
    )
    args = parser.parse_args()

    lia = LecturaLiaisons()

    if args.check:
        word = args.check
        is_h = lia.is_h_aspire(word)
        print(f"  «{word}» → h aspiré : {'oui' if is_h else 'non'}")
        return

    if args.interactive:
        print("╔══════════════════════════════════════════════════╗")
        print("║   Lectura Liaisons — Mode interactif             ║")
        print("║   Format : mot1 pos1 phone1 | mot2 pos2 phone2  ║")
        print("║   Ex : les ART:def le | enfants NOM ɑ̃fɑ̃         ║")
        print("║   'q' pour quitter.                              ║")
        print("╚══════════════════════════════════════════════════╝")
        print()

        while True:
            try:
                line = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line or line.lower() in ("q", "quit", "exit"):
                break

            parts = line.split("|")
            if len(parts) != 2:
                print("  Format attendu : mot1 pos1 phone1 | mot2 pos2 phone2")
                continue

            try:
                p1 = parts[0].strip().split()
                p2 = parts[1].strip().split()
                w1 = MotInfo(ortho=p1[0], phone=p1[2] if len(p1) > 2 else "", pos=[p1[1]] if len(p1) > 1 else [])
                w2 = MotInfo(ortho=p2[0], phone=p2[2] if len(p2) > 2 else "", pos=[p2[1]] if len(p2) > 1 else [])
            except (IndexError, ValueError):
                print("  Format attendu : mot1 pos1 phone1 | mot2 pos2 phone2")
                continue

            print()
            print(lia.format_decision(w1, w2))
            print()

        return

    # Mode exemples
    print("╔══════════════════════════════════════════════════╗")
    print("║   Lectura Liaisons — Exemples de classification  ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    exemples = [
        ("Obligatoire (ART + NOM)",
         MotInfo("les", "le", ["ART:def"]), MotInfo("enfants", "ɑ̃fɑ̃", ["NOM"])),
        ("Obligatoire (ART:ind + NOM, dénasalisation)",
         MotInfo("un", "œ̃", ["ART:ind"]), MotInfo("ami", "ami", ["NOM"])),
        ("Obligatoire (ADJ + NOM)",
         MotInfo("petit", "pəti", ["ADJ"]), MotInfo("oiseau", "wazo", ["NOM"])),
        ("Obligatoire (PRO:per + AUX)",
         MotInfo("ils", "il", ["PRO:per"]), MotInfo("ont", "ɔ̃", ["AUX"])),
        ("Obligatoire (est + VER)",
         MotInfo("est", "ɛ", ["AUX"]), MotInfo("arrivé", "aʁive", ["VER"])),
        ("Interdite (et + ...)",
         MotInfo("et", "e", ["CON"]), MotInfo("alors", "alɔʁ", ["ADV"])),
        ("Bloquée (h aspiré : héros)",
         MotInfo("les", "le", ["ART:def"]), MotInfo("héros", "eʁo", ["NOM"])),
        ("Enchaînement (neuf heures → v)",
         MotInfo("neuf", "nœf", ["NUM"]), MotInfo("heures", "œʁ", ["NOM"])),
        ("Enchaînement (avec arrivé)",
         MotInfo("avec", "avɛk", ["PRE"]), MotInfo("elle", "ɛl", ["PRO:per"])),
    ]

    for label, w1, w2 in exemples:
        print(f"--- {label} ---")
        print(lia.format_decision(w1, w2))
        print()

    # ── Pipeline apply_jonctions ──
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   Pipeline apply_jonctions                       ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("Phrase : L'enfant est peut-être arrivé")
    print()

    tokens = [
        TokenMot("L'", "l", ["ART:def"], (0, 2)),
        TokenSep("'", "apostrophe", (1, 2)),
        TokenMot("enfant", "ɑ̃fɑ̃", ["NOM"], (2, 8)),
        TokenSep(" ", "space", (8, 9)),
        TokenMot("est", "ɛ", ["AUX"], (9, 12)),
        TokenSep(" ", "space", (12, 13)),
        TokenMot("peut", "pø", ["VER"], (13, 17)),
        TokenSep("-", "hyphen", (17, 18)),
        TokenMot("être", "ɛtʁ", ["VER"], (18, 22)),
        TokenSep(" ", "space", (22, 23)),
        TokenMot("arrivé", "aʁive", ["VER"], (23, 29)),
    ]

    groups = lia.apply_jonctions(tokens)

    for g in groups:
        parts = []
        for c in g.components:
            if isinstance(c, TokenMot):
                parts.append(c.ortho)
            elif isinstance(c, TokenSep):
                parts.append(c.text)
        label = "".join(parts)
        typ = g.jonction_type or "simple"
        print(f"  {label:20s}  /{g.phone}/  ({typ})")

    print()
    print("Phrase : Les enfants ont mangé un excellent repas")
    print()

    tokens2 = [
        TokenMot("Les", "le", ["ART:def"], (0, 3)),
        TokenSep(" ", "space", (3, 4)),
        TokenMot("enfants", "ɑ̃fɑ̃", ["NOM"], (4, 11)),
        TokenSep(" ", "space", (11, 12)),
        TokenMot("ont", "ɔ̃", ["AUX"], (12, 15)),
        TokenSep(" ", "space", (15, 16)),
        TokenMot("mangé", "mɑ̃ʒe", ["VER"], (16, 21)),
        TokenSep(" ", "space", (21, 22)),
        TokenMot("un", "œ̃", ["ART:ind"], (22, 24)),
        TokenSep(" ", "space", (24, 25)),
        TokenMot("excellent", "ɛksɛlɑ̃", ["ADJ"], (25, 34)),
        TokenSep(" ", "space", (34, 35)),
        TokenMot("repas", "ʁəpa", ["NOM"], (35, 40)),
    ]

    groups2 = lia.apply_jonctions(tokens2)

    for g in groups2:
        parts = []
        for c in g.components:
            if isinstance(c, TokenMot):
                parts.append(c.ortho)
            elif isinstance(c, TokenSep):
                parts.append(c.text)
        label = "".join(parts)
        typ = g.jonction_type or "simple"
        print(f"  {label:30s}  /{g.phone}/  ({typ})")


if __name__ == "__main__":
    main()
