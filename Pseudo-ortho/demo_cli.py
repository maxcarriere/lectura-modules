#!/usr/bin/env python3
"""Demo en ligne de commande de Lectura Pseudo-Ortho.

Usage :
    python demo_cli.py "kɑ̃"
    python demo_cli.py "tʁa" "bɔ̃"
    python demo_cli.py                    # mode interactif
"""

from __future__ import annotations

import argparse
import sys

from lectura_pseudo_ortho import LecturaPseudoOrtho


def main() -> None:
    parser = argparse.ArgumentParser(description="Lectura Pseudo-Ortho — Demo CLI")
    parser.add_argument("ipa", nargs="*", help="Syllabe(s) IPA a convertir")
    args = parser.parse_args()

    p2g = LecturaPseudoOrtho()

    # Mode argument
    if args.ipa:
        for syl in args.ipa:
            result = p2g.predict(syl)
            print(f"  /{syl}/ → {result}")
        if len(args.ipa) > 1:
            mot = p2g.predict_word(args.ipa)
            print(f"  mot : {mot}")
        return

    # Mode interactif
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Lectura Pseudo-Ortho — IPA → pseudo-orthographe  ║")
    print("║   Tapez une syllabe IPA, 'q' pour quitter.         ║")
    print("║   Separez par espaces pour un mot multi-syllabes.  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text or text.lower() in ("q", "quit", "exit"):
            break

        parts = text.split()
        if len(parts) == 1:
            result = p2g.predict(parts[0])
            print(f"  /{parts[0]}/ → {result}")
        else:
            for syl in parts:
                result = p2g.predict(syl)
                print(f"  /{syl}/ → {result}")
            mot = p2g.predict_word(parts)
            print(f"  mot : {mot}")
        print()


if __name__ == "__main__":
    main()
