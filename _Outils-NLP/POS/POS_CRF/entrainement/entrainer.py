#!/usr/bin/env python3
"""Entraîne un modèle CRF POS-tagging — Script autonome.

Utilise sklearn-crfsuite pour l'entraînement et exporte au format JSON
compatible avec lectura_pos.py.

Pré-requis :
    pip install sklearn-crfsuite

Usage :
    # Entraînement standard (avec les données fournies)
    python entrainer.py

    # Entraînement personnalisé
    python entrainer.py --corpus mon_corpus.conllu --dev mon_dev.conllu

    # Fine-tuning : ajouter vos données à celles existantes
    python entrainer.py --corpus donnees/pos_train_merged.conllu --extra mon_corpus.conllu

    # Ajuster la régularisation
    python entrainer.py --c1 0.05 --c2 0.05 --max-iter 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Mapping UD → tags projet ────────────────────────────────────────────────

_UPOS_MAP: dict[str, str] = {
    "NOUN": "NOM", "PROPN": "NOM", "VERB": "VER", "AUX": "AUX",
    "ADJ": "ADJ", "ADV": "ADV", "ADP": "PRE", "CCONJ": "CON",
    "SCONJ": "CON", "INTJ": "INTJ", "NUM": "NOM", "SYM": "NOM", "X": "NOM",
}

_FEATURE_MAP: list[tuple[str, str, str, str]] = [
    ("DET", "Definite", "Def", "ART:def"),
    ("DET", "Definite", "Ind", "ART:ind"),
    ("DET", "PronType", "Art", "ART:def"),
    ("DET", "Poss", "Yes", "ADJ:pos"),
    ("DET", "PronType", "Dem", "ADJ:dem"),
    ("DET", "PronType", "Int", "ADJ:int"),
    ("PRON", "PronType", "Prs", "PRO:per"),
    ("PRON", "PronType", "Rel", "PRO:rel"),
    ("PRON", "PronType", "Dem", "PRO:dem"),
    ("PRON", "Poss", "Yes", "PRO:pos"),
    ("PRON", "PronType", "Int", "PRO:int"),
    ("PRON", "PronType", "Ind", "PRO:ind"),
]

_UPOS_FALLBACK: dict[str, str] = {"DET": "ART:ind", "PRON": "PRO:per"}
_IGNORE_UPOS = {"PUNCT", "SPACE"}

_LEX_TAGS = ("VER", "AUX", "NOM", "ADJ", "ADV", "ART:def", "ART:ind",
             "PRO:per", "PRE", "CON")


def _parse_ud_feats(feat_string: str) -> dict[str, str]:
    if not feat_string or feat_string == "_":
        return {}
    result: dict[str, str] = {}
    for part in feat_string.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def _ud_to_project_tag(upos: str, feats: dict[str, str] | None = None) -> str | None:
    if upos in _IGNORE_UPOS:
        return None
    if feats:
        for rule_upos, feat_key, feat_val, tag in _FEATURE_MAP:
            if upos == rule_upos and feats.get(feat_key) == feat_val:
                return tag
    if upos in _UPOS_FALLBACK:
        return _UPOS_FALLBACK[upos]
    return _UPOS_MAP.get(upos, "NOM")


# ── Parsing CoNLL-U ─────────────────────────────────────────────────────────

def parse_conllu(path: str) -> list[list[tuple[str, str, dict[str, str]]]]:
    """Parse un fichier CoNLL-U → [(word, upos, feats), ...] par phrase."""
    sentences: list[list[tuple[str, str, dict[str, str]]]] = []
    current: list[tuple[str, str, dict[str, str]]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            if "-" in fields[0] or "." in fields[0]:
                continue
            word = fields[1]
            upos = fields[3]
            feats = _parse_ud_feats(fields[5])
            current.append((word, upos, feats))

    if current:
        sentences.append(current)
    return sentences


def convert_sentences(
    raw_sentences: list[list[tuple[str, str, dict[str, str]]]],
) -> list[tuple[list[str], list[str]]]:
    """Convertit en (words, tags) avec tags projet."""
    result = []
    for sent in raw_sentences:
        words: list[str] = []
        tags: list[str] = []
        for word, upos, feats in sent:
            tag = _ud_to_project_tag(upos, feats)
            if tag is None:
                continue
            words.append(word)
            tags.append(tag)
        if words:
            result.append((words, tags))
    return result


# ── Extraction de features ──────────────────────────────────────────────────

def extract_word_features(
    words: list[str], idx: int,
) -> dict[str, str]:
    """Extrait les features pour le mot à la position idx."""
    word = words[idx]
    low = word.lower()
    feats: dict[str, str] = {
        "bias": "1.0",
        "word": low,
        "len": str(float(len(low))),
        "suf3": low[-3:] if len(low) >= 3 else low,
        "suf2": low[-2:] if len(low) >= 2 else low,
        "pre2": low[:2] if len(low) >= 2 else low,
        "pre3": low[:3] if len(low) >= 3 else low,
        "is_upper": "1.0" if word.isupper() else "0.0",
        "is_title": "1.0" if word.istitle() else "0.0",
        "is_digit": "1.0" if word.isdigit() else "0.0",
    }

    # Sans lexique → features lex.* à 0
    for tag in _LEX_TAGS:
        feats[f"lex.{tag}"] = "0.0"

    feats["BOS"] = "1.0" if idx == 0 else "0.0"
    feats["EOS"] = "1.0" if idx == len(words) - 1 else "0.0"
    feats["w-1"] = words[idx - 1].lower() if idx > 0 else "__BOS__"
    feats["w+1"] = words[idx + 1].lower() if idx < len(words) - 1 else "__EOS__"

    return feats


def sentences_to_crfsuite_data(
    sentences: list[tuple[list[str], list[str]]],
) -> tuple[list[list[dict]], list[list[str]]]:
    """Convertit en format sklearn-crfsuite (X, y)."""
    X: list[list[dict]] = []
    y: list[list[str]] = []

    for words, tags in sentences:
        feats = [extract_word_features(words, i) for i in range(len(words))]
        X.append(feats)
        y.append(tags)

    return X, y


# ── Export du modèle ─────────────────────────────────────────────────────────

def export_crfsuite_model(crf_model, tags: list[str], output_path: str) -> None:
    """Exporte un modèle sklearn-crfsuite vers le format JSON Lectura."""
    state_features: dict[str, dict[str, float]] = {}
    for (attr, label), weight in crf_model.state_features_.items():
        if abs(weight) < 1e-6:
            continue
        if attr not in state_features:
            state_features[attr] = {}
        state_features[attr][label] = weight

    transitions: dict[str, dict[str, float]] = {}
    for (label_from, label_to), weight in crf_model.transition_features_.items():
        if abs(weight) < 1e-6:
            continue
        if label_from not in transitions:
            transitions[label_from] = {}
        transitions[label_from][label_to] = weight

    model_data = {
        "state_features": state_features,
        "transitions": transitions,
        "tags": tags,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=1)

    size_kb = output.stat().st_size / 1024
    print(f"Modèle sauvegardé : {output} ({size_kb:.0f} Ko)")


# ── Évaluation ───────────────────────────────────────────────────────────────

def evaluate(crf_model, X_test, y_test) -> dict[str, float]:
    """Évalue le modèle sur un jeu de test."""
    y_pred = crf_model.predict(X_test)

    total = 0
    correct = 0
    tag_stats: dict[str, dict[str, int]] = {}

    for pred_seq, gold_seq in zip(y_pred, y_test):
        for p, g in zip(pred_seq, gold_seq):
            total += 1
            if p == g:
                correct += 1
            if g not in tag_stats:
                tag_stats[g] = {"total": 0, "correct": 0}
            tag_stats[g]["total"] += 1
            if p == g:
                tag_stats[g]["correct"] += 1

    acc = correct / total if total else 0
    print(f"  Accuracy : {acc:.2%} ({correct}/{total})")

    sorted_tags = sorted(tag_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    print("  Par tag :")
    for tag, stats in sorted_tags:
        tag_acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"    {tag:10s} : {tag_acc:.1%} ({stats['correct']}/{stats['total']})")

    return {"accuracy": acc, "total": total}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Entraîne un modèle CRF POS-tagging pour le français"
    )
    parser.add_argument(
        "--corpus",
        default=str(here / "donnees" / "pos_train_merged.conllu"),
        help="Fichier CoNLL-U d'entraînement",
    )
    parser.add_argument(
        "--dev",
        default=str(here / "donnees" / "pos_dev_merged.conllu"),
        help="Fichier CoNLL-U de validation",
    )
    parser.add_argument(
        "--extra",
        help="Fichier CoNLL-U additionnel (fine-tuning : ajouté au corpus principal)",
    )
    parser.add_argument(
        "--output",
        default=str(here / ".." / "modele" / "pos_model_crf.json"),
        help="Fichier de sortie du modèle JSON",
    )
    parser.add_argument("--c1", type=float, default=0.1, help="Régularisation L1 (défaut: 0.1)")
    parser.add_argument("--c2", type=float, default=0.1, help="Régularisation L2 (défaut: 0.1)")
    parser.add_argument("--max-iter", type=int, default=100, help="Itérations max (défaut: 100)")
    args = parser.parse_args()

    try:
        import sklearn_crfsuite
    except ImportError:
        print("ERREUR : sklearn-crfsuite requis.")
        print("  pip install sklearn-crfsuite")
        sys.exit(1)

    # Parsing du corpus principal
    print(f"Parsing du corpus : {args.corpus}")
    raw_train = parse_conllu(args.corpus)
    train_data = convert_sentences(raw_train)
    print(f"  {len(train_data)} phrases")

    # Corpus additionnel (fine-tuning)
    if args.extra:
        extra_path = Path(args.extra)
        if extra_path.exists():
            print(f"Parsing du corpus additionnel : {args.extra}")
            raw_extra = parse_conllu(args.extra)
            extra_data = convert_sentences(raw_extra)
            print(f"  {len(extra_data)} phrases ajoutées")
            train_data.extend(extra_data)
        else:
            print(f"ATTENTION : corpus additionnel introuvable : {args.extra}")

    X_train, y_train = sentences_to_crfsuite_data(train_data)

    all_tags = sorted(set(tag for tags in y_train for tag in tags))
    print(f"  Total : {len(train_data)} phrases, {len(all_tags)} tags")
    print(f"  Tags : {all_tags}")

    # Entraînement
    print(f"\nEntraînement CRF (c1={args.c1}, c2={args.c2}, max_iter={args.max_iter})...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iter,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    print("\n--- Évaluation sur train ---")
    evaluate(crf, X_train, y_train)

    # Évaluation sur dev
    dev_path = Path(args.dev)
    if dev_path.exists():
        print(f"\nParsing dev : {args.dev}")
        raw_dev = parse_conllu(args.dev)
        dev_data = convert_sentences(raw_dev)
        X_dev, y_dev = sentences_to_crfsuite_data(dev_data)
        print(f"  {len(dev_data)} phrases")
        print("\n--- Évaluation sur dev ---")
        evaluate(crf, X_dev, y_dev)

    # Export
    print(f"\nExport vers {args.output}...")
    export_crfsuite_model(crf, all_tags, args.output)
    print("Terminé.")


if __name__ == "__main__":
    main()
