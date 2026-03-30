#!/usr/bin/env python3
"""Evaluation du G2P Lectura (backend BiLSTM) sur un fichier gold (dico.csv).

Compare deux configurations :
  1. Modele seul (sans corrections)
  2. Modele + corrections

Usage :
    python evaluer.py --gold chemin/vers/dico.csv
    python evaluer.py --gold dico.csv --export EVALUATION.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

from lectura_g2p import (
    G2pCorrections,
    LecturaG2P,
    iter_phonemes,
    postprocess,
)

HERE = Path(__file__).parent


# ── Normalisation tolerante ──────────────────────────────────────────────────

def _normalize_o(ipa: str) -> str:
    """Normalise ɔ→o. Respecte combining marks."""
    phonemes = iter_phonemes(ipa)
    result: list[str] = []
    for ph in phonemes:
        base = ph[0]
        rest = ph[1:]
        if rest:
            result.append(ph)
        elif base == "ɔ":
            result.append("o")
        else:
            result.append(ph)
    return "".join(result)


def _normalize_mid_vowels(ipa: str) -> str:
    """Normalise ɔ→o ET ɛ→e. Respecte combining marks."""
    phonemes = iter_phonemes(ipa)
    result: list[str] = []
    for ph in phonemes:
        base = ph[0]
        rest = ph[1:]
        if rest:
            result.append(ph)
        elif base == "ɔ":
            result.append("o")
        elif base == "ɛ":
            result.append("e")
        else:
            result.append(ph)
    return "".join(result)


# ── Chargement gold ──────────────────────────────────────────────────────────

def load_dico(path: Path) -> dict[str, set[str]]:
    """Charge dico.csv → dict[ortho] → set[phones]."""
    word_phones: dict[str, set[str]] = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()
            if not ortho or not phone or " " in ortho:
                continue
            if ortho not in word_phones:
                word_phones[ortho] = set()
            word_phones[ortho].add(phone)
    return word_phones


# ── Evaluation ───────────────────────────────────────────────────────────────

def _levenshtein(a: list[str], b: list[str]) -> int:
    """Distance de Levenshtein entre deux listes de phonemes."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def evaluate(
    g2p: LecturaG2P,
    dico: dict[str, set[str]],
    *,
    use_corrections: bool = True,
) -> dict:
    """Evalue le G2P sur toutes les entrees du dico gold."""
    total = 0
    correct = 0
    correct_o = 0
    correct_oe = 0
    errors: list[tuple[str, str, str, set[str]]] = []

    for ortho, phones_set in dico.items():
        total += 1

        pred = g2p.predict(ortho) if use_corrections else ""
        if not use_corrections:
            pred = g2p._model.predict(ortho)
            if pred:
                pred = postprocess(ortho, pred)

        is_correct = pred in phones_set
        pred_o = _normalize_o(pred)
        is_correct_o = any(_normalize_o(g) == pred_o for g in phones_set)
        pred_oe = _normalize_mid_vowels(pred)
        is_correct_oe = any(_normalize_mid_vowels(g) == pred_oe for g in phones_set)

        if is_correct:
            correct += 1
        if is_correct_o:
            correct_o += 1
        if is_correct_oe:
            correct_oe += 1

        if not is_correct:
            pred_ph = iter_phonemes(pred) if pred else []
            best_gold = min(
                phones_set,
                key=lambda g: _levenshtein(pred_ph, iter_phonemes(g)),
            )
            errors.append((ortho, pred, best_gold, phones_set))

    return {
        "total": total,
        "correct": correct,
        "correct_o": correct_o,
        "correct_oe": correct_oe,
        "accuracy": correct / total if total else 0,
        "accuracy_o": correct_o / total if total else 0,
        "accuracy_oe": correct_oe / total if total else 0,
        "errors": errors,
    }


def print_results(name: str, results: dict) -> None:
    """Affiche les resultats d'evaluation."""
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print(f"\n  Accuracy exacte   : {results['accuracy']:.2%}"
          f"  ({results['correct']}/{results['total']})")
    print(f"  Accuracy (o/ɔ)    : {results['accuracy_o']:.2%}")
    print(f"  Accuracy (o/ɔ+e/ɛ): {results['accuracy_oe']:.2%}")
    print(f"  Erreurs           : {len(results['errors'])}")


def print_top_errors(results: dict, n: int = 30) -> None:
    """Affiche les N premieres erreurs."""
    errors = results["errors"][:n]
    if not errors:
        print("\n  Aucune erreur.")
        return

    print(f"\n  Top {n} erreurs :")
    print(f"  {'Mot':<20} {'Prediction':<15} {'Gold':<15}")
    print(f"  {'-' * 50}")
    for ortho, pred, best_gold, _ in errors:
        print(f"  {ortho:<20} {pred:<15} {best_gold:<15}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation du G2P Lectura (BiLSTM)")
    parser.add_argument(
        "--gold", required=True,
        help="Fichier gold (dico.csv : ortho,phone)",
    )
    parser.add_argument(
        "--export", default=None,
        help="Fichier JSON de sortie pour les resultats",
    )
    parser.add_argument(
        "--top-errors", type=int, default=30,
        help="Nombre d'erreurs a afficher (defaut: 30)",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"ERREUR : fichier gold non trouve : {gold_path}", file=sys.stderr)
        sys.exit(1)

    model_path = HERE / "modele" / "g2p_model_bilstm_int8.onnx"
    vocab_path = HERE / "modele" / "g2p_vocab.json"
    corrections_path = HERE / "modele" / "g2p_corrections_bilstm.json"
    model_name = "BiLSTM INT8"

    if not model_path.exists():
        print(f"ERREUR : modele non trouve : {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Chargement du gold ({gold_path})...")
    dico = load_dico(gold_path)
    print(f"  {len(dico):,} mots")

    # Config 1 : modele seul
    print(f"\nChargement du modele {model_name}...")
    g2p_base = LecturaG2P(model_path, vocab_path=vocab_path)

    t0 = time.time()
    print("Evaluation modele seul...")
    results_base = evaluate(g2p_base, dico, use_corrections=False)
    print_results(f"{model_name} seul (sans corrections)", results_base)
    print_top_errors(results_base, args.top_errors)

    # Config 2 : modele + corrections
    corr_path = corrections_path if corrections_path.exists() else None
    if corr_path:
        print(f"\nChargement des corrections ({corr_path.name})...")
        g2p_corr = LecturaG2P(
            model_path, vocab_path=vocab_path, corrections_path=corr_path,
        )
        n_entries = g2p_corr._corrections.n_entries if g2p_corr._corrections else 0
        print(f"  {n_entries} entrees")

        print("Evaluation modele + corrections...")
        results_corr = evaluate(g2p_corr, dico, use_corrections=True)
        print_results(f"{model_name} + corrections", results_corr)
        print_top_errors(results_corr, args.top_errors)

        # Comparaison
        diff = results_corr["correct"] - results_base["correct"]
        print(f"\n{'=' * 72}")
        print(f"  COMPARAISON")
        print(f"{'=' * 72}")
        print(f"  {'Metrique':<25} {'Seul':>12} {'+ Corrections':>14} {'Diff':>8}")
        print(f"  {'-' * 62}")
        print(f"  {'Accuracy exacte':<25} {results_base['accuracy']:>11.2%}"
              f" {results_corr['accuracy']:>13.2%}"
              f"  {'+' if diff >= 0 else ''}{diff:>5}")
        best = results_corr
    else:
        print("\n  (corrections non trouvees, evaluation simple)")
        best = results_base

    elapsed = time.time() - t0

    if args.export:
        export = {
            "model": model_name,
            "total": best["total"],
            "accuracy": round(best["accuracy"], 4),
            "accuracy_o": round(best["accuracy_o"], 4),
            "accuracy_oe": round(best["accuracy_oe"], 4),
            "n_errors": len(best["errors"]),
        }
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        print(f"\n  Resultats exportes : {args.export}")

    print(f"\n  Temps : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
