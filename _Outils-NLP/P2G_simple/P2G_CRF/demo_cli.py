#!/usr/bin/env python3
"""Demo en ligne de commande du P2G Lectura (backend CRF).

Usage :
    python demo_cli.py bɔ̃ʒuʁ pɛʃœʁ
    python demo_cli.py                       # mode demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lectura_p2g import LecturaP2G

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Lectura P2G — Demo CLI (CRF)")
    parser.add_argument("ipa", nargs="*", help="Chaine(s) IPA a convertir")
    args = parser.parse_args()

    model_path = HERE / "modele" / "p2g_model_crf.json"

    if model_path.exists():
        p2g = LecturaP2G(model_path)
    else:
        print(f"NOTE : modele CRF non trouve ({model_path}), utilisation table + regles",
              file=sys.stderr)
        p2g = LecturaP2G()

    # Mode argument
    if args.ipa:
        for ipa in args.ipa:
            candidates = p2g.predict_candidates(ipa, k=5)
            print(f"/{ipa}/")
            for word, prob in candidates:
                bar = "█" * int(prob * 20)
                print(f"  {word:<25s} {prob:>6.1%} {bar}")
            print()
        return

    # Mode demo
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║   Lectura P2G — Phoneme-Grapheme (CRF)         ║")
    print(f"║   Mode: {'CRF' if p2g.has_model else 'table + regles':30s}     ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    test_words = ["bɔ̃ʒuʁ", "mɛzɔ̃", "ʃa", "o", "pɛʃœʁ", "ɑ̃fɑ̃", "vɛʁ"]
    for ipa in test_words:
        ortho = p2g.predict(ipa)
        print(f"  /{ipa}/ → {ortho}")


if __name__ == "__main__":
    main()
