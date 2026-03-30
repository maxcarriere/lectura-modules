#!/usr/bin/env python3
"""Demo CLI pour Lectura P2G (facade multi-backend).

Usage :
    # Table + regles (zero dependance)
    python demo_cli.py

    # Avec backend specifique
    python demo_cli.py --crf bɔ̃ʒuʁ pɛʃœʁ
    python demo_cli.py --bilstm bɔ̃ʒuʁ
    python demo_cli.py --seq2seq bɔ̃ʒuʁ

    # Comparer les 3 backends
    python demo_cli.py --compare bɔ̃ʒuʁ pɛʃœʁ

    # Mot(s) specifique(s)
    python demo_cli.py bɔ̃ʒuʁ pɛʃœʁ ɑ̃tikɔ̃stitysjɔnɛləmɑ̃

    # Mode interactif
    python demo_cli.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lectura_p2g import LecturaP2G, _load_backend


HERE = Path(__file__).parent


def _make_p2g(args) -> LecturaP2G:
    """Cree une instance LecturaP2G selon les arguments."""
    if args.backend:
        return _load_backend(HERE, args.backend)
    if args.model:
        return LecturaP2G(model_path=args.model, table_path=args.table)
    return LecturaP2G(table_path=args.table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lectura P2G — Convertisseur phoneme → grapheme",
    )
    parser.add_argument("ipa", nargs="*", help="Chaine(s) IPA a convertir")
    parser.add_argument("--crf", dest="backend", action="store_const", const="crf",
                        help="Backend CRF")
    parser.add_argument("--bilstm", dest="backend", action="store_const", const="bilstm",
                        help="Backend BiLSTM (ONNX)")
    parser.add_argument("--seq2seq", dest="backend", action="store_const", const="seq2seq",
                        help="Backend Seq2Seq (ONNX)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare les 3 backends cote a cote")
    parser.add_argument("--model", type=str, default=None,
                        help="Chemin vers un fichier modele (auto-detect backend)")
    parser.add_argument("--table", type=str, default=None,
                        help="Fichier p2g_table.json externe")
    parser.add_argument("-k", "--top-k", type=int, default=5,
                        help="Nombre de candidats (defaut: 5)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Mode interactif (saisie continue)")
    args = parser.parse_args()

    if args.compare:
        _compare(args)
        return

    p2g = _make_p2g(args)
    print(f"Lectura P2G — Backend: {p2g.backend} (modele: {'oui' if p2g.has_model else 'non'})")
    print()

    if args.interactive:
        _interactive(p2g, args.top_k)
    elif args.ipa:
        _process_words(p2g, args.ipa, args.top_k)
    else:
        _demo(p2g, args.top_k)


def _process_words(p2g: LecturaP2G, words: list[str], k: int) -> None:
    """Traite une liste de mots IPA."""
    for ipa in words:
        candidates = p2g.predict_candidates(ipa, k=k)
        print(f"/{ipa}/")
        for word, prob in candidates:
            bar = "█" * int(prob * 20)
            print(f"  {word:<25s} {prob:>6.1%} {bar}")
        print()


def _interactive(p2g: LecturaP2G, k: int) -> None:
    """Mode interactif."""
    print("Entrez une transcription IPA (Ctrl+D pour quitter) :")
    print()
    try:
        while True:
            try:
                ipa = input("IPA> ").strip()
            except EOFError:
                break
            if not ipa:
                continue
            _process_words(p2g, [ipa], k)
    except KeyboardInterrupt:
        pass
    print("\nAu revoir !")


def _compare(args) -> None:
    """Compare les 3 backends cote a cote."""
    backends = {}
    for name in ("crf", "bilstm", "seq2seq"):
        try:
            backends[name] = _load_backend(HERE, name)
        except Exception:
            print(f"  WARN: {name} non disponible", file=sys.stderr)

    if not backends:
        print("ERREUR : aucun backend disponible", file=sys.stderr)
        sys.exit(1)

    ipa_list = args.ipa or [
        "bɔ̃ʒuʁ", "mɛzɔ̃", "ʃa", "o", "pɛʃœʁ", "ɑ̃fɑ̃",
    ]

    header = f"  {'IPA':25}"
    for name in backends:
        header += f" {name:>22}"
    print(header)
    print("  " + "-" * (25 + 22 * len(backends)))

    for ipa in ipa_list:
        line = f"  /{ipa:<23}/"
        for name, p2g in backends.items():
            pred = p2g.predict(ipa)
            line += f" {pred:>22}"
        print(line)


def _demo(p2g: LecturaP2G, k: int) -> None:
    """Demo avec des mots courants."""
    test_words = [
        ("bɔ̃ʒuʁ", "bonjour"),
        ("mɛzɔ̃", "maison"),
        ("ʃa", "chat"),
        ("o", "eau"),
        ("wazo", "oiseau"),
        ("pɛʃœʁ", "pecheur"),
        ("faʁmasi", "pharmacie"),
        ("ʃɑ̃", "chant"),
        ("fɛʁ", "fer"),
        ("ʒɑ̃", "jean"),
        ("kɔ̃stitysjɔ̃", "constitution"),
        ("ɑ̃fɑ̃", "enfant"),
        ("pʁɔfɛsœʁ", "professeur"),
        ("ɑ̃tikɔ̃stitysjɔnɛləmɑ̃", "anticonstitutionnellement"),
    ]

    print(f"--- Demo P2G ({p2g.backend}) ---")
    print(f"{'IPA':<30s} {'Top-1':<20s} {'Reference':<20s}")
    print("-" * 70)

    for ipa, ref in test_words:
        pred = p2g.predict(ipa)
        mark = "✓" if pred == ref else "~"
        print(f"/{ipa:<28s}/ {pred:<20s} {ref:<20s} {mark}")

    if p2g.has_model:
        print("\n--- Candidates ---")
        for ipa, ref in test_words[:5]:
            candidates = p2g.predict_candidates(ipa, k=k)
            top_words = ", ".join(f"{w} ({p:.0%})" for w, p in candidates[:3])
            in_top = "✓" if ref in [w for w, _ in candidates] else "✗"
            print(f"  /{ipa}/ → {top_words}  [{in_top} in top-{k}]")


if __name__ == "__main__":
    main()
