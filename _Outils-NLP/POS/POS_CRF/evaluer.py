#!/usr/bin/env python3
"""Évaluation du POS Tagger Lectura sur les corpus Universal Dependencies.

Compare deux configurations :
  1. CRF seul (sans mini-lexique)
  2. CRF + mini-lexique (post-traitement)

Usage :
    python evaluer.py --test chemin/vers/pos_test_merged.conllu
    python evaluer.py --test chemin/vers/fr_gsd-ud-test.conllu
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from lectura_pos import PosTagger, _apply_lexicon, _load_lexicon

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

def parse_conllu(path: Path) -> list[list[tuple[str, str]]]:
    """Parse un fichier CoNLL-U → [(mot, tag_projet), ...] par phrase."""
    sentences: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

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
            tag = _ud_to_project_tag(upos, feats)
            if tag is None:
                continue
            current.append((word, tag))
    if current:
        sentences.append(current)
    return sentences


# ── Métriques ────────────────────────────────────────────────────────────────

def evaluate(
    pred_all: list[list[str]],
    gold_all: list[list[str]],
) -> dict:
    """Calcule accuracy globale, P/R/F1 par tag, matrice de confusion."""
    total = 0
    correct = 0
    tag_stats: dict[str, dict[str, int]] = {}
    confusion: Counter[tuple[str, str]] = Counter()

    for pred_seq, gold_seq in zip(pred_all, gold_all):
        for pred, gold in zip(pred_seq, gold_seq):
            total += 1
            if gold == pred:
                correct += 1

            for t in (gold, pred):
                if t not in tag_stats:
                    tag_stats[t] = {"tp": 0, "fp": 0, "fn": 0}

            if gold == pred:
                tag_stats[gold]["tp"] += 1
            else:
                tag_stats[gold]["fn"] += 1
                tag_stats[pred]["fp"] += 1
                confusion[(gold, pred)] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "tag_stats": tag_stats,
        "confusion": confusion,
    }


def print_results(name: str, results: dict) -> None:
    """Affiche les résultats d'évaluation."""
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print(f"\n  Accuracy globale : {results['accuracy']:.2%}"
          f"  ({results['correct']}/{results['total']})\n")

    # Per-tag table
    tag_stats = results["tag_stats"]
    tags_sorted = sorted(tag_stats.keys(),
                         key=lambda t: tag_stats[t]["tp"] + tag_stats[t]["fn"],
                         reverse=True)

    print(f"  {'Tag':<10} {'Prec':>7} {'Rappel':>7} {'F1':>7}"
          f"  {'Support':>8}")
    print(f"  {'-' * 47}")

    for tag in tags_sorted:
        s = tag_stats[tag]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        support = tp + fn
        print(f"  {tag:<10} {prec:>7.1%} {rec:>7.1%} {f1:>7.1%}"
              f"  {support:>8}")

    # Top erreurs
    confusion = results["confusion"]
    if confusion:
        print(f"\n  Top 15 confusions :")
        print(f"  {'Vrai':<10} {'Prédit':<10} {'Nb':>6}")
        print(f"  {'-' * 30}")
        for (gold, pred), count in confusion.most_common(15):
            print(f"  {gold:<10} {pred:<10} {count:>6}")


def print_comparison(r1: dict, r2: dict) -> None:
    """Affiche un tableau comparatif entre deux configurations."""
    print(f"\n{'=' * 72}")
    print(f"  COMPARAISON : CRF seul vs CRF + mini-lexique")
    print(f"{'=' * 72}\n")

    diff = r2["correct"] - r1["correct"]
    print(f"  {'Métrique':<25} {'CRF seul':>12} {'CRF + lexique':>14} {'Diff':>8}")
    print(f"  {'-' * 62}")
    print(f"  {'Accuracy':<25} {r1['accuracy']:>11.2%} {r2['accuracy']:>13.2%}"
          f"  {'+' if diff >= 0 else ''}{diff:>5}")
    print(f"  {'Tokens corrects':<25} {r1['correct']:>12} {r2['correct']:>14}")
    print(f"  {'Tokens total':<25} {r1['total']:>12} {r2['total']:>14}")

    # Per-tag comparison (only tags where there's a difference)
    all_tags = set(r1["tag_stats"].keys()) | set(r2["tag_stats"].keys())
    diffs = []
    for tag in all_tags:
        s1 = r1["tag_stats"].get(tag, {"tp": 0, "fp": 0, "fn": 0})
        s2 = r2["tag_stats"].get(tag, {"tp": 0, "fp": 0, "fn": 0})
        tp_diff = s2["tp"] - s1["tp"]
        if tp_diff != 0:
            sup = s1["tp"] + s1["fn"]
            f1_1 = _f1(s1)
            f1_2 = _f1(s2)
            diffs.append((tag, sup, f1_1, f1_2, tp_diff))

    if diffs:
        diffs.sort(key=lambda x: abs(x[4]), reverse=True)
        print(f"\n  Tags avec changement :")
        print(f"  {'Tag':<10} {'Support':>8} {'F1 avant':>10} {'F1 après':>10}"
              f" {'TP diff':>8}")
        print(f"  {'-' * 50}")
        for tag, sup, f1_1, f1_2, tp_diff in diffs:
            sign = "+" if tp_diff > 0 else ""
            print(f"  {tag:<10} {sup:>8} {f1_1:>9.1%} {f1_2:>9.1%}"
                  f"  {sign}{tp_diff:>6}")
    else:
        print("\n  Aucune différence par tag.")


def _f1(stats: dict[str, int]) -> float:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0


# ── Export JSON ──────────────────────────────────────────────────────────────

def export_results(results: dict, name: str, path: Path) -> None:
    """Exporte les résultats en JSON."""
    tag_stats = results["tag_stats"]
    export = {
        "name": name,
        "accuracy": round(results["accuracy"], 4),
        "total_tokens": results["total"],
        "correct_tokens": results["correct"],
        "per_tag": {},
    }
    for tag, s in sorted(tag_stats.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        export["per_tag"][tag] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"\n  Résultats exportés : {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Évaluation du POS Tagger Lectura")
    parser.add_argument(
        "--test", required=True,
        help="Fichier CoNLL-U de test (ex: pos_test_merged.conllu)",
    )
    parser.add_argument(
        "--model", default=str(here / "modele" / "pos_model_crf.json"),
        help="Chemin du modèle CRF",
    )
    parser.add_argument(
        "--lexicon", default=str(here / "modele" / "mini_lexique.json"),
        help="Chemin du mini-lexique",
    )
    parser.add_argument(
        "--export", default=str(here / "EVALUATION.json"),
        help="Fichier JSON de sortie pour les résultats",
    )
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"ERREUR : fichier test introuvable : {test_path}", file=sys.stderr)
        sys.exit(1)

    # Charger le test set
    print("Chargement du test set...")
    sentences = parse_conllu(test_path)
    n_tokens = sum(len(s) for s in sentences)
    print(f"  {len(sentences)} phrases, {n_tokens} tokens (hors PUNCT/SPACE)")

    # Préparer les données gold
    gold_all = [[tag for _, tag in sent] for sent in sentences]
    words_all = [[word for word, _ in sent] for sent in sentences]

    # ── Config 1 : CRF seul ──
    print("\nChargement du modèle CRF...")
    tagger_base = PosTagger(args.model)

    print("Évaluation CRF seul...")
    pred_base = [
        [tag for _, tag in tagger_base.tag_words(words)]
        for words in words_all
    ]
    results_base = evaluate(pred_base, gold_all)
    print_results("CRF seul (sans mini-lexique)", results_base)

    # ── Config 2 : CRF + mini-lexique ──
    lexicon_path = Path(args.lexicon)
    if lexicon_path.exists():
        print("\nChargement du mini-lexique...")
        tagger_lex = PosTagger(args.model, lexicon_path=args.lexicon)
        n_entries = len(tagger_lex.lexicon)
        print(f"  {n_entries} entrées chargées")

        print("Évaluation CRF + mini-lexique...")
        pred_lex = [
            [tag for _, tag in tagger_lex.tag_words(words)]
            for words in words_all
        ]
        results_lex = evaluate(pred_lex, gold_all)
        print_results("CRF + mini-lexique", results_lex)

        # Comparaison
        print_comparison(results_base, results_lex)

        # Export de la meilleure config
        export_results(results_lex, "CRF + mini-lexique", Path(args.export))
    else:
        print(f"\n  (mini-lexique non trouvé : {lexicon_path}, évaluation simple)")
        export_results(results_base, "CRF seul", Path(args.export))


if __name__ == "__main__":
    main()
