#!/usr/bin/env python3
"""Evaluation du P2G Lectura (backend BiLSTM) sur un fichier gold (dico.csv).

Usage :
    python evaluer.py --gold chemin/vers/dico.csv
    python evaluer.py --gold dico.csv --export EVALUATION.json --top-errors 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from lectura_p2g import LecturaP2G

HERE = Path(__file__).parent


# ── Levenshtein ──────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Distance de Levenshtein entre deux chaines."""
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


# ── Chargement gold ──────────────────────────────────────────────────────────

def load_dico(path: Path) -> dict[str, set[str]]:
    """Charge dico.csv → dict[phone] → set[ortho]."""
    phone_orthos: dict[str, set[str]] = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()
            if not ortho or not phone or " " in ortho:
                continue
            if phone not in phone_orthos:
                phone_orthos[phone] = set()
            phone_orthos[phone].add(ortho)
    return phone_orthos


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    p2g: LecturaP2G,
    dico: dict[str, set[str]],
    *,
    top_k: int = 5,
) -> dict:
    """Evalue le P2G sur toutes les entrees du dico gold."""
    total = 0
    correct = 0
    correct_topk = 0
    cer_sum = 0.0
    errors: list[tuple[str, str, str, set[str]]] = []

    for phone, ortho_set in dico.items():
        total += 1

        pred = p2g.predict(phone)

        is_correct = pred in ortho_set
        if is_correct:
            correct += 1

        # Top-K
        candidates = p2g.predict_candidates(phone, k=top_k)
        pred_words = {w for w, _ in candidates}
        if ortho_set & pred_words:
            correct_topk += 1

        # CER (vs best gold)
        best_gold = min(
            ortho_set,
            key=lambda g: _levenshtein(pred, g),
        )
        cer = _levenshtein(pred, best_gold) / max(len(best_gold), 1)
        cer_sum += cer

        if not is_correct:
            errors.append((phone, pred, best_gold, ortho_set))

    # Trier erreurs par CER decroissant
    errors.sort(key=lambda e: _levenshtein(e[1], e[2]) / max(len(e[2]), 1), reverse=True)

    return {
        "total": total,
        "correct": correct,
        "correct_topk": correct_topk,
        "top_k": top_k,
        "accuracy": correct / total if total else 0,
        "accuracy_topk": correct_topk / total if total else 0,
        "cer": cer_sum / total if total else 0,
        "errors": errors,
    }


def print_results(name: str, results: dict) -> None:
    """Affiche les resultats d'evaluation."""
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print(f"\n  Accuracy exacte    : {results['accuracy']:.2%}"
          f"  ({results['correct']}/{results['total']})")
    print(f"  Accuracy top-{results['top_k']}     : {results['accuracy_topk']:.2%}"
          f"  ({results['correct_topk']}/{results['total']})")
    print(f"  CER moyen          : {results['cer']:.2%}")
    print(f"  Erreurs            : {len(results['errors'])}")


def print_top_errors(results: dict, n: int = 30) -> None:
    """Affiche les N premieres erreurs."""
    errors = results["errors"][:n]
    if not errors:
        print("\n  Aucune erreur.")
        return

    print(f"\n  Top {n} erreurs :")
    print(f"  {'IPA':<25} {'Prediction':<20} {'Gold':<20}")
    print(f"  {'-' * 65}")
    for phone, pred, best_gold, _ in errors:
        print(f"  {phone:<25} {pred:<20} {best_gold:<20}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation du P2G Lectura (BiLSTM)")
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
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Nombre de candidats pour accuracy top-K (defaut: 5)",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"ERREUR : fichier gold non trouve : {gold_path}", file=sys.stderr)
        sys.exit(1)

    model_path = HERE / "modele" / "p2g_bilstm_int8.onnx"
    vocab_path = HERE / "modele" / "p2g_vocab.json"
    model_name = "BiLSTM"

    print(f"Chargement du gold ({gold_path})...")
    dico = load_dico(gold_path)
    print(f"  {len(dico):,} entrees (phone → ortho)")

    if model_path.exists() and vocab_path.exists():
        print(f"\nChargement du modele {model_name}...")
        p2g = LecturaP2G(model_path, vocab_path=vocab_path)
    else:
        print(f"\n  Modele BiLSTM non trouve ({model_path}), utilisation table + regles")
        p2g = LecturaP2G()

    t0 = time.time()
    print("Evaluation...")
    results = evaluate(p2g, dico, top_k=args.top_k)
    elapsed = time.time() - t0

    print_results(f"P2G {model_name} ({'modele' if p2g.has_model else 'table+regles'})", results)
    print_top_errors(results, args.top_errors)

    if args.export:
        export = {
            "model": model_name,
            "has_model": p2g.has_model,
            "total": results["total"],
            "accuracy": round(results["accuracy"], 4),
            "accuracy_topk": round(results["accuracy_topk"], 4),
            "top_k": results["top_k"],
            "cer": round(results["cer"], 4),
            "n_errors": len(results["errors"]),
            "time_s": round(elapsed, 1),
        }
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        print(f"\n  Resultats exportes : {args.export}")

    print(f"\n  Temps : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
