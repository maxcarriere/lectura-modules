#!/usr/bin/env python3
"""Benchmark du correcteur sur les corpus reels.

Charge les deux corpus (corpus_evaluation 152 cas, corpus_benchmark 120 cas)
et evalue les performances.

Usage :
    python scripts/benchmark/run_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

# S'assurer que le projet est dans le PYTHONPATH
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Chemins en dur
LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura.db"
NGRAM_DB = "/data/work/projets/lectura/data/wikipedia/ngram_3.db"


@dataclass
class Cas:
    entree: str
    attendu: list[str]  # accepter plusieurs formes attendues
    categorie: str


def charger_corpus_evaluation() -> list[Cas]:
    """Charge tests/corpus_evaluation.py."""
    from tests.corpus_evaluation import CORPUS
    return [
        Cas(
            entree=c.erronee,
            attendu=[c.attendue],
            categorie=c.categories[0] if c.categories else "AUTRE",
        )
        for c in CORPUS
        if c.implementee
    ]


def charger_corpus_benchmark() -> list[Cas]:
    """Charge scripts/benchmark/corpus_benchmark.py."""
    from scripts.benchmark.corpus_benchmark import (
        ORTH_ACCENT, ORTH_PHONETIQUE, ORTH_TYPO, ORTH_DOUBLE, ORTH_INVARIABLE,
        ORTH_GRAPHIE, ORTH_SEGMENTATION,
        ACCORD_DET_NOM, ACCORD_SV, ACCORD_ADJ, ACCORD_PP_ETRE, ACCORD_ATTR,
        CONJ_PRESENT, CONJ_IMPARFAIT, CONJ_FUTUR, CONJ_PP_INF, CONJ_IRREGULIER,
        HOMO_ET_EST, HOMO_A_A, HOMO_SON_SONT, HOMO_ON_ONT, HOMO_AUTRES,
        AUTRE_NEGATION, AUTRE_ORDRE, AUTRE_SEMANTIQUE, AUTRE_PONCTUATION,
        OK_PIEGE_HOMO, OK_PIEGE_ACCORD, OK_COMPLEXE, OK_SIMPLE,
    )

    all_groups = [
        ORTH_ACCENT, ORTH_PHONETIQUE, ORTH_TYPO, ORTH_DOUBLE, ORTH_INVARIABLE,
        ORTH_GRAPHIE, ORTH_SEGMENTATION,
        ACCORD_DET_NOM, ACCORD_SV, ACCORD_ADJ, ACCORD_PP_ETRE, ACCORD_ATTR,
        CONJ_PRESENT, CONJ_IMPARFAIT, CONJ_FUTUR, CONJ_PP_INF, CONJ_IRREGULIER,
        HOMO_ET_EST, HOMO_A_A, HOMO_SON_SONT, HOMO_ON_ONT, HOMO_AUTRES,
        AUTRE_NEGATION, AUTRE_ORDRE, AUTRE_SEMANTIQUE, AUTRE_PONCTUATION,
        OK_PIEGE_HOMO, OK_PIEGE_ACCORD, OK_COMPLEXE, OK_SIMPLE,
    ]
    result = []
    for group in all_groups:
        for c in group:
            result.append(Cas(
                entree=c.erronee,
                attendu=c.attendue if isinstance(c.attendue, list) else [c.attendue],
                categorie=c.categorie,
            ))
    return result


def evaluer(correcteur, corpus: list[Cas], nom: str):
    """Evalue et affiche les resultats."""
    n_total = len(corpus)
    n_ok = 0
    par_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ok": 0},
    )
    ecarts = []

    t0 = time.perf_counter()
    for cas in corpus:
        result = correcteur.corriger(cas.entree)
        sortie = result.phrase_corrigee
        cat = cas.categorie
        par_cat[cat]["total"] += 1

        # Accepter plusieurs formes attendues
        match = any(
            sortie.lower().strip().rstrip(".") == att.lower().strip().rstrip(".")
            for att in cas.attendu
        )
        if match:
            n_ok += 1
            par_cat[cat]["ok"] += 1
        else:
            ecarts.append((cat, cas.entree, cas.attendu[0], sortie))

    elapsed = time.perf_counter() - t0
    ms = (elapsed / n_total) * 1000 if n_total else 0

    print(f"\n{'=' * 70}")
    print(f"  {nom} — {n_ok}/{n_total} corrects ({100*n_ok/n_total:.1f}%)  "
          f"  {ms:.0f} ms/phrase")
    print(f"{'=' * 70}")

    # Par categorie
    for cat in sorted(par_cat):
        s = par_cat[cat]
        t, o = s["total"], s["ok"]
        pct = 100 * o / t if t else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cat:<15} {o:>3}/{t:<3} {bar} {pct:5.1f}%")

    # Premiers ecarts
    if ecarts:
        print(f"\n  Premiers ecarts ({min(10, len(ecarts))}/{len(ecarts)}) :")
        for cat, entree, attendu, sortie in ecarts[:10]:
            print(f"    [{cat}]")
            print(f"      IN:  {entree}")
            print(f"      ATT: {attendu}")
            print(f"      OUT: {sortie}")

    return n_ok, n_total, elapsed


def main():
    print("Chargement du lexique...", file=sys.stderr)
    try:
        from lectura_lexique import Lexique
        lexique = Lexique(LEXIQUE_DB)
    except Exception as e:
        print(f"ERREUR lexique: {e}", file=sys.stderr)
        sys.exit(1)

    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig

    # Charger les corpus
    print("Chargement des corpus...", file=sys.stderr)
    try:
        corpus_eval = charger_corpus_evaluation()
        print(f"  corpus_evaluation: {len(corpus_eval)} cas", file=sys.stderr)
    except Exception as e:
        print(f"  corpus_evaluation: ERREUR {e}", file=sys.stderr)
        corpus_eval = []

    try:
        corpus_bench = charger_corpus_benchmark()
        print(f"  corpus_benchmark: {len(corpus_bench)} cas", file=sys.stderr)
    except Exception as e:
        print(f"  corpus_benchmark: ERREUR {e}", file=sys.stderr)
        corpus_bench = []

    corpus_all = corpus_eval + corpus_bench
    if not corpus_all:
        print("Aucun corpus charge!", file=sys.stderr)
        sys.exit(1)

    # --- Correcteur ---
    print("\nConstruction du correcteur...", file=sys.stderr)
    config = CorrecteurConfig(activer_scoring=True)
    correcteur = Correcteur(lexique, config=config)

    # Evaluer sur corpus_evaluation
    if corpus_eval:
        print("\n" + "━" * 70)
        print("  CORPUS EVALUATION")
        print("━" * 70)
        evaluer(correcteur, corpus_eval, "correcteur")

    # Evaluer sur corpus_benchmark
    if corpus_bench:
        print("\n" + "━" * 70)
        print("  CORPUS BENCHMARK")
        print("━" * 70)
        evaluer(correcteur, corpus_bench, "correcteur")

    # Evaluer combine
    if corpus_eval and corpus_bench:
        print("\n" + "━" * 70)
        print("  COMBINE (TOTAL)")
        print("━" * 70)
        evaluer(correcteur, corpus_all, "correcteur")


if __name__ == "__main__":
    main()
