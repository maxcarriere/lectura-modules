#!/usr/bin/env python3
"""Demo en ligne de commande du P2G Lectura (backend BiLSTM).

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
    parser = argparse.ArgumentParser(description="Lectura P2G — Demo CLI (BiLSTM)")
    parser.add_argument("ipa", nargs="*", help="Chaine(s) IPA a convertir")
    args = parser.parse_args()

    model_path = HERE / "modele" / "p2g_bilstm_int8.onnx"
    vocab_path = HERE / "modele" / "p2g_vocab.json"

    if model_path.exists() and vocab_path.exists():
        p2g = LecturaP2G(model_path, vocab_path=vocab_path)
    else:
        print(f"NOTE : modele BiLSTM non trouve ({model_path}), utilisation table + regles",
              file=sys.stderr)
        p2g = LecturaP2G()

    # Mode argument
    if args.ipa:
        for ipa in args.ipa:
            candidates = p2g.predict_candidates(ipa, k=5)
            print(f"/{ipa}/")
            for word, prob in candidates:
                bar = "\u2588" * int(prob * 20)
                print(f"  {word:<25s} {prob:>6.1%} {bar}")
            print()
        return

    # Mode demo
    print(f"\u2554{'':═<50s}\u2557")
    print(f"\u2551   Lectura P2G — Phoneme-Grapheme (BiLSTM)      \u2551")
    print(f"\u2551   Mode: {'BiLSTM' if p2g.has_model else 'table + regles':30s}     \u2551")
    print(f"\u255a{'':═<50s}\u255d")
    print()

    test_words = ["\u0062\u0254\u0303\u0292\u0075\u0281", "m\u025bz\u0254\u0303",
                  "\u0283a", "o", "p\u025b\u0283\u0153\u0281",
                  "\u0251\u0303f\u0251\u0303", "v\u025b\u0281"]
    for ipa in test_words:
        ortho = p2g.predict(ipa)
        print(f"  /{ipa}/ \u2192 {ortho}")


if __name__ == "__main__":
    main()
