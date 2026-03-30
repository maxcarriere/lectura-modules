#!/usr/bin/env python3
"""Entraine un modele CRF P2G et l'exporte en JSON.

Architecture : features IPA → CRF → export JSON → Viterbi.

Pre-requis :
    pip install sklearn-crfsuite

Usage :
    python entrainer_crf.py --train train_p2g.csv --eval eval_p2g.csv
    python entrainer_crf.py --train train_p2g.csv --output modele.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path

# Ajouter le parent pour importer les features
sys.path.insert(0, str(Path(__file__).parent.parent))

from lectura_p2g import _extract_ipa_features, iter_phonemes

_CONT = "_CONT"


def load_dataset(path: Path) -> list[tuple[str, list[str]]]:
    """Charge un CSV (ipa, ortho, aligned_labels) → [(ipa, labels)]."""
    data: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ipa = row.get("ipa", "").strip()
            labels_str = row.get("aligned_labels", "").strip()
            if not ipa or not labels_str:
                continue
            labels = labels_str.split(",")
            phonemes = iter_phonemes(ipa)
            if len(labels) == len(phonemes):
                data.append((ipa, labels))
    return data


def extract_features(ipa: str) -> list[dict]:
    """Extrait les features pour une chaine IPA."""
    ipa_chars = iter_phonemes(ipa)
    return [_extract_ipa_features(ipa_chars, i) for i in range(len(ipa_chars))]


def train_crf(train_data, max_iter: int = 100):
    """Entraine un modele CRF (Passive Aggressive, online)."""
    try:
        import sklearn_crfsuite
    except ImportError:
        print("ERREUR : sklearn-crfsuite non installe.", file=sys.stderr)
        print("  pip install sklearn-crfsuite", file=sys.stderr)
        sys.exit(1)

    X_train = []
    y_train = []

    for ipa, labels in train_data:
        feats = extract_features(ipa)
        str_feats = [{k: str(v) for k, v in f.items()} for f in feats]
        X_train.append(str_feats)
        y_train.append(labels)

    print(f"  Sequences : {len(X_train)}")

    crf = sklearn_crfsuite.CRF(
        algorithm="pa",
        max_iterations=max_iter,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    return crf


def export_to_json(crf, output_path: Path) -> None:
    """Exporte le modele CRF en JSON (format Viterbi)."""
    state_features: dict[str, dict[str, float]] = {}
    for (attr, label), weight in crf.state_features_.items():
        if abs(weight) < 1e-6:
            continue
        if attr not in state_features:
            state_features[attr] = {}
        state_features[attr][label] = weight

    transitions: dict[str, dict[str, float]] = {}
    for (label_from, label_to), weight in crf.transition_features_.items():
        if abs(weight) < 1e-6:
            continue
        if label_from not in transitions:
            transitions[label_from] = {}
        transitions[label_from][label_to] = weight

    model_data = {
        "state_features": state_features,
        "transitions": transitions,
        "tags": list(crf.classes_),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Modele sauvegarde : {output_path} ({size_kb:.0f} KB)")
    print(f"  Tags : {len(model_data['tags'])}")
    print(f"  State features : {len(state_features)}")
    print(f"  Transitions : {sum(len(v) for v in transitions.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement CRF P2G")
    parser.add_argument("--train", required=True, help="CSV d'entrainement")
    parser.add_argument("--eval", default=None, help="CSV d'evaluation")
    parser.add_argument("--output", default="../modele/p2g_model_crf.json",
                        help="Fichier JSON de sortie")
    parser.add_argument("--max-iter", type=int, default=100, help="Iterations max")
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"ERREUR : {train_path} non trouve", file=sys.stderr)
        sys.exit(1)

    print("Chargement des donnees...")
    train_data = load_dataset(train_path)
    print(f"  {len(train_data)} mots charges")

    print("Entrainement CRF...")
    crf = train_crf(train_data, max_iter=args.max_iter)

    print("Export JSON...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_json(crf, output_path)

    if args.eval:
        eval_path = Path(args.eval)
        if eval_path.exists():
            print("\nEvaluation...")
            eval_data = load_dataset(eval_path)
            correct = 0
            total = 0
            for ipa, gold_labels in eval_data:
                feats = extract_features(ipa)
                str_feats = [{k: str(v) for k, v in f.items()} for f in feats]
                pred_labels = crf.predict_single(str_feats)
                gold_ortho = "".join(l for l in gold_labels if l != _CONT)
                pred_ortho = "".join(l for l in pred_labels if l != _CONT)
                total += 1
                if pred_ortho == gold_ortho:
                    correct += 1
            print(f"  Word accuracy : {correct / total:.2%} ({correct}/{total})")

    print("\nTermine.")


if __name__ == "__main__":
    main()
