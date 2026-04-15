#!/usr/bin/env python3
"""Démonstration interactive du modèle unifié.

Usage :
    python demo_cli.py --backend numpy
    python demo_cli.py --backend onnx
    python demo_cli.py --backend pure
    python demo_cli.py --backend numpy --phrase "Les enfants jouent"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from lectura_nlp.tokeniseur import tokeniser
from lectura_nlp.posttraitement import (
    appliquer_liaison,
    charger_corrections,
    charger_homographes,
    corriger_g2p,
)


def create_engine(backend: str, models_dir: Path):
    """Crée le moteur d'inférence pour le backend choisi."""
    vocab_path = models_dir / "unifie_vocab.json"
    weights_path = models_dir / "unifie_weights.json"
    onnx_path = models_dir / "unifie_int8.onnx"

    if backend == "onnx":
        from lectura_nlp.inference_onnx import OnnxInferenceEngine
        return OnnxInferenceEngine(onnx_path, vocab_path)
    elif backend == "numpy":
        from lectura_nlp.inference_numpy import NumpyInferenceEngine
        return NumpyInferenceEngine(weights_path, vocab_path)
    elif backend == "pure":
        from lectura_nlp.inference_pure import PureInferenceEngine
        return PureInferenceEngine(weights_path, vocab_path)
    else:
        raise ValueError(f"Backend inconnu : {backend}")


def format_result(result: dict) -> str:
    """Formate le résultat d'analyse pour l'affichage."""
    lines = []
    tokens = result["tokens"]
    g2p = result.get("g2p", [])
    pos = result.get("pos", [])
    liaison = result.get("liaison", [])
    morpho = result.get("morpho", {})

    # En-tête du tableau
    header = f"{'Mot':15s} {'IPA':15s} {'POS':10s} {'Liaison':8s}"
    morpho_feats = sorted(morpho.keys())
    for feat in morpho_feats:
        header += f" {feat:8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, tok in enumerate(tokens):
        ipa = g2p[i] if i < len(g2p) else ""
        p = pos[i] if i < len(pos) else ""
        lia = liaison[i] if i < len(liaison) else ""
        row = f"{tok:15s} {ipa:15s} {p:10s} {lia:8s}"
        for feat in morpho_feats:
            val = morpho.get(feat, [])[i] if i < len(morpho.get(feat, [])) else "_"
            row += f" {val:8s}"
        lines.append(row)

    # IPA avec liaisons appliquées
    if g2p and liaison:
        ipa_with_liaison = appliquer_liaison(tokens, g2p, liaison)
        lines.append("")
        lines.append("IPA avec liaisons : " + " ".join(ipa_with_liaison))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Démonstration du modèle unifié G2P+POS+Morpho+Liaison"
    )
    parser.add_argument(
        "--backend", default="numpy",
        choices=["onnx", "numpy", "pure"],
        help="Backend d'inférence (défaut: numpy)",
    )
    parser.add_argument(
        "--modeles", type=Path, default=_ROOT / "modeles",
        help="Répertoire des fichiers modèle",
    )
    parser.add_argument(
        "--phrase", type=str, default=None,
        help="Phrase à analyser (mode non-interactif)",
    )
    args = parser.parse_args()

    print(f"Chargement du modèle (backend={args.backend})...")
    engine = create_engine(args.backend, args.modeles)

    # Charger les corrections G2P
    corrections_path = args.modeles / "g2p_corrections_unifie.json"
    if corrections_path.exists():
        charger_corrections(corrections_path)
        print(f"Corrections G2P chargées ({corrections_path.name})")

    # Charger la table d'homographes (POS-aware, prioritaire sur corrections)
    homographes_path = args.modeles / "homographes.json"
    if homographes_path.exists():
        charger_homographes(homographes_path)
        print(f"Homographes chargés ({homographes_path.name})")

    print("Prêt.\n")

    if args.phrase:
        tokens = tokeniser(args.phrase)
        result = engine.analyser(tokens)
        for i, tok in enumerate(tokens):
            if i < len(result["g2p"]):
                pos_tag = result["pos"][i] if i < len(result.get("pos", [])) else None
                result["g2p"][i] = corriger_g2p(tok, result["g2p"][i], pos_tag)
        print(format_result(result))
        return

    # Mode interactif
    print("Entrez une phrase (ou 'q' pour quitter) :")
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

        tokens = tokeniser(text)
        if not tokens:
            print("  (aucun token)")
            continue

        result = engine.analyser(tokens)

        # Appliquer les corrections G2P (POS-aware)
        for i, tok in enumerate(tokens):
            if i < len(result["g2p"]):
                pos_tag = result["pos"][i] if i < len(result.get("pos", [])) else None
                result["g2p"][i] = corriger_g2p(tok, result["g2p"][i], pos_tag)

        print(format_result(result))


if __name__ == "__main__":
    main()
