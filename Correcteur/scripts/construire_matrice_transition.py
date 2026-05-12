#!/usr/bin/env python3
"""Construit une matrice de transition bigram POS a partir du corpus.

Lit corpus_edit.jsonl (270k entrees avec POS gold), extrait les sequences
POS, compte les bigrams, applique un smoothing Laplace, et sauvegarde
en log-probabilites JSON.

Usage :
    python scripts/construire_matrice_transition.py
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(_PROJECT_ROOT, "data", "corpus", "corpus_edit.jsonl")
OUTPUT_PATH = os.path.join(
    _PROJECT_ROOT, "src", "lectura_correcteur", "data",
    "transition_matrix_bigram.json",
)

# Smoothing Laplace
ALPHA = 0.01

# Symboles speciaux
BOS = "<BOS>"
EOS = "<EOS>"


def main() -> None:
    # 1. Compter les bigrams
    bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pos_set: set[str] = set()
    n_phrases = 0

    print(f"Lecture de {CORPUS_PATH}...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            morpho = entry.get("morpho", [])
            if not morpho:
                continue
            # Extraire la sequence POS
            pos_seq = [m.get("pos", "") for m in morpho]
            pos_seq = [p for p in pos_seq if p]  # skip empty
            if not pos_seq:
                continue

            pos_set.update(pos_seq)
            n_phrases += 1

            # Compter BOS -> premier, inter-mots, dernier -> EOS
            bigram_counts[BOS][pos_seq[0]] += 1
            for i in range(len(pos_seq) - 1):
                bigram_counts[pos_seq[i]][pos_seq[i + 1]] += 1
            bigram_counts[pos_seq[-1]][EOS] += 1

    print(f"  {n_phrases} phrases lues")
    print(f"  {len(pos_set)} labels POS distincts: {sorted(pos_set)}")

    # 2. Construire la liste de labels (avec BOS/EOS)
    pos_labels = sorted(pos_set)
    all_labels = pos_labels + [EOS]  # transitions possibles vers

    # 3. Calculer log-probabilites avec smoothing Laplace
    log_probs: dict[str, dict[str, float]] = {}
    contexts = [BOS] + pos_labels  # transitions depuis

    for ctx in contexts:
        counts = bigram_counts[ctx]
        total = sum(counts.values()) + ALPHA * len(all_labels)
        row: dict[str, float] = {}
        for label in all_labels:
            count = counts[label] + ALPHA
            row[label] = math.log(count / total)
        log_probs[ctx] = row

    # 4. Sauvegarder
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = {
        "description": "Matrice de transition bigram POS (log-probabilites, Laplace alpha=0.01)",
        "n_phrases": n_phrases,
        "alpha": ALPHA,
        "pos_labels": pos_labels,
        "log_probs": log_probs,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Stats
    n_entries = sum(len(row) for row in log_probs.values())
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\n  Matrice sauvegardee: {OUTPUT_PATH}")
    print(f"  Taille: {file_size / 1024:.1f} KB")
    print(f"  Dimensions: {len(contexts)} x {len(all_labels)} = {n_entries} entrees")

    # Verification: probabilites sommant a ~1 par ligne
    for ctx in contexts[:3]:
        row = log_probs[ctx]
        total_prob = sum(math.exp(v) for v in row.values())
        print(f"  Verif {ctx}: sum(exp(log_p)) = {total_prob:.6f}")


if __name__ == "__main__":
    main()
