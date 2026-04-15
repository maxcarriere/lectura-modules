#!/usr/bin/env python3
"""Démonstration interactive du modèle unifié P2G.

Usage :
    python demo_cli.py --backend onnx
    python demo_cli.py --backend numpy
    python demo_cli.py --backend pure
    python demo_cli.py --backend onnx --phrase "le ʃa ɛ bɔ̃"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))


def create_engine(backend: str, models_dir: Path):
    """Crée le moteur d'inférence pour le backend choisi."""
    vocab_path = models_dir / "unifie_p2g_v2_vocab.json"
    weights_path = models_dir / "unifie_p2g_v2_weights.json"
    onnx_path = models_dir / "unifie_p2g_v2_int8.onnx"

    if backend == "onnx":
        from lectura_p2g.inference_onnx import OnnxInferenceEngine
        return OnnxInferenceEngine(onnx_path, vocab_path)
    elif backend == "numpy":
        from lectura_p2g.inference_numpy import NumpyInferenceEngine
        return NumpyInferenceEngine(weights_path, vocab_path)
    elif backend == "pure":
        from lectura_p2g.inference_pure import PureInferenceEngine
        return PureInferenceEngine(weights_path, vocab_path)
    else:
        raise ValueError(f"Backend inconnu : {backend}")


def format_result(result: dict) -> str:
    """Formate le résultat d'analyse pour l'affichage."""
    lines = []
    ipa_words = result.get("ipa_words", [])
    ortho = result.get("ortho", [])
    pos = result.get("pos", [])
    morpho = result.get("morpho", {})

    # En-tête du tableau
    header = f"{'IPA':15s} {'Ortho':15s} {'POS':10s}"
    morpho_feats = sorted(morpho.keys())
    for feat in morpho_feats:
        header += f" {feat:8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for i in range(len(ipa_words)):
        ipa = ipa_words[i] if i < len(ipa_words) else ""
        ort = ortho[i] if i < len(ortho) else ""
        p = pos[i] if i < len(pos) else ""
        row = f"{ipa:15s} {ort:15s} {p:10s}"
        for feat in morpho_feats:
            val = morpho.get(feat, [])[i] if i < len(morpho.get(feat, [])) else "_"
            row += f" {val:8s}"
        lines.append(row)

    # Phrase reconstruite
    if ortho:
        lines.append("")
        lines.append("Orthographe : " + " ".join(ortho))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Démonstration du modèle unifié P2G+POS+Morpho"
    )
    parser.add_argument(
        "--backend", default="onnx",
        choices=["onnx", "numpy", "pure"],
        help="Backend d'inférence (défaut: onnx)",
    )
    parser.add_argument(
        "--modeles", type=Path, default=_ROOT / "modeles",
        help="Répertoire des fichiers modèle",
    )
    parser.add_argument(
        "--phrase", type=str, default=None,
        help="Phrase IPA à analyser (mode non-interactif)",
    )
    args = parser.parse_args()

    print(f"Chargement du modèle (backend={args.backend})...")
    engine = create_engine(args.backend, args.modeles)
    print("Prêt.\n")

    if args.phrase:
        ipa_words = args.phrase.strip().split()
        result = engine.analyser(ipa_words)
        print(format_result(result))
        return

    # Mode interactif
    print("Entrez une phrase IPA (mots séparés par des espaces), ou 'q' pour quitter :")
    print("Exemple : le ʃa ɛ bɔ̃")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if text.lower() in ("q", "quit", "exit"):
            print("Au revoir.")
            break

        if not text:
            continue

        ipa_words = text.split()
        if not ipa_words:
            print("  (aucun mot)")
            continue

        result = engine.analyser(ipa_words)
        print(format_result(result))


if __name__ == "__main__":
    main()
