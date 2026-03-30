#!/usr/bin/env python3
"""Demo en ligne de commande du G2P Lectura (backend Seq2Seq).

Usage :
    python demo_cli.py "Bonjour le monde"
    python demo_cli.py                       # mode interactif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lectura_g2p import LecturaG2P

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Lectura G2P — Demo CLI (Seq2Seq)")
    parser.add_argument("text", nargs="*", help="Texte a transcrire")
    args = parser.parse_args()

    encoder_path = HERE / "modele" / "g2p_seq2seq_encoder_int8.onnx"
    decoder_path = HERE / "modele" / "g2p_seq2seq_decoder_int8.onnx"
    vocab_path = HERE / "modele" / "g2p_seq2seq_vocab.json"
    corrections_path = HERE / "modele" / "g2p_corrections_seq2seq.json"

    if not encoder_path.exists():
        print(f"ERREUR : modele non trouve : {encoder_path}", file=sys.stderr)
        sys.exit(1)

    corr = corrections_path if corrections_path.exists() else None
    g2p = LecturaG2P(
        encoder_path, decoder_path=decoder_path,
        vocab_path=vocab_path, corrections_path=corr,
    )

    # Mode argument
    if args.text:
        text = " ".join(args.text)
        for word in text.split():
            print(g2p.predict_formatted(word))
        return

    # Mode interactif
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║   Lectura G2P — Grapheme-Phoneme (Seq2Seq)     ║")
    print(f"║   Tapez un mot ou une phrase, 'q' pour quitter. ║")
    print(f"╚══════════════════════════════════════════════════╝")
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
        for word in text.split():
            print(g2p.predict_formatted(word))
        print()


if __name__ == "__main__":
    main()
