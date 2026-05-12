#!/usr/bin/env python3
"""Analyse detaillee des FP sur le corpus negatif.

Identifie les regles et patterns responsables des faux positifs
sur les phrases correctes.

Usage :
    python scripts/analyser_fp_negatif.py
    python scripts/analyser_fp_negatif.py --max 200
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=500)
    parser.add_argument("--corpus", default=os.path.join(_PROJECT_ROOT, "data", "negatif_wicopaco.tsv"))
    args = parser.parse_args()

    # Charger corpus
    paires = []
    with open(args.corpus, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            paires.append((row[0].strip(), row[1].strip(), row[2].strip()))
    if args.max > 0:
        paires = paires[:args.max]

    print(f"Corpus: {len(paires)} phrases correctes")

    # Charger correcteur
    sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    lexique = Lexique(LEXIQUE_DB)
    correcteur = Correcteur(lexique)

    # Evaluer et collecter les FP
    fp_list = []
    ok_count = 0
    skip_count = 0

    for _, phrase, _ in paires:
        try:
            resultat = correcteur.corriger(phrase)
        except Exception as e:
            skip_count += 1
            continue

        # Filtrer maj/ponctuation
        grammar_corrections = [
            c for c in resultat.corrections
            if c.regle not in ("syntaxe.majuscule",)
            and not c.regle.startswith("ponctuation")
        ]
        if grammar_corrections:
            for c in grammar_corrections:
                fp_list.append({
                    "regle": c.regle,
                    "original": c.original,
                    "corrige": c.corrige,
                    "explication": c.explication,
                    "phrase": phrase[:100],
                })
        else:
            ok_count += 1

    fp_phrases = len(set(d["phrase"] for d in fp_list))
    print(f"\nOK: {ok_count}, FP phrases: {fp_phrases}, Skip: {skip_count}")
    print(f"Total corrections FP: {len(fp_list)}")

    # Par regle
    print(f"\n{'='*70}")
    print("FP PAR REGLE")
    print(f"{'='*70}")
    regle_counter = Counter(d["regle"] for d in fp_list)
    for regle, count in regle_counter.most_common(30):
        print(f"\n  {count:>4d}  {regle}")
        # Montrer exemples
        exemples = [d for d in fp_list if d["regle"] == regle][:5]
        for ex in exemples:
            print(f"        {ex['original']} → {ex['corrige']}")
            print(f"        | {ex['phrase'][:80]}")

    # Par pattern original→corrige
    print(f"\n{'='*70}")
    print("FP PAR PATTERN (original → corrige)")
    print(f"{'='*70}")
    pattern_counter = Counter(
        f"{d['original'].lower()} → {d['corrige'].lower()}" for d in fp_list
    )
    for pattern, count in pattern_counter.most_common(30):
        print(f"  {count:>4d}  {pattern}")


if __name__ == "__main__":
    main()
