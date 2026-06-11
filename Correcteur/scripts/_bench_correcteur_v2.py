#!/usr/bin/env python3
"""Benchmark du CorrecteurV2 sur WiCoPaCo.

Charge les paires (phrase_erronee, phrase_corrigee) de WiCoPaCo et du
corpus negatif, execute le pipeline v2 et calcule les metriques :
- Precision/Recall des corrections
- FP sur phrases correctes
- Detail par categorie (accord, conjugaison, homophone)

Usage :
    python3 scripts/_bench_correcteur_v2.py [--max N] [--expand]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Ajouter le src au path
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))


# -----------------------------------------------------------------------
# Chargement WiCoPaCo
# -----------------------------------------------------------------------

def charger_wicopaco(path: str, max_n: int = 0) -> list[tuple[str, str, str]]:
    """Charge (type_erreur, phrase_erronee, phrase_corrigee) depuis le TSV."""
    paires: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            cat = row[0].strip()
            err = row[1].strip()
            cor = row[2].strip()
            if err and cor:
                paires.append((cat, err, cor))
                if max_n and len(paires) >= max_n:
                    break
    return paires


def charger_negatif(path: str, max_n: int = 0) -> list[str]:
    """Charge les phrases correctes du corpus negatif."""
    phrases: list[str] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            phrase = row[1].strip()
            if phrase:
                phrases.append(phrase)
                if max_n and len(phrases) >= max_n:
                    break
    return phrases


# -----------------------------------------------------------------------
# Comparaison mot a mot
# -----------------------------------------------------------------------

def _tokenize_simple(phrase: str) -> list[str]:
    """Tokenise en mots simples (split sur espaces + ponctuation)."""
    import re
    return re.findall(r"[\w]+(?:[-']\w+)*", phrase.lower())


def comparer_corrections(
    phrase_err: str,
    phrase_cor: str,
    phrase_v2: str,
) -> dict:
    """Compare les corrections v2 vs gold via diff-based alignment.

    Utilise SequenceMatcher pour aligner err/cor (gold diff) et err/v2
    (corrections appliquees). Evite les faux positifs dus aux changements
    de tokenisation (elisions, espacement).

    Returns:
        dict avec VP, FP, FP_cible (mauvaise correction cible),
        FP_collateral (correction non-cible), FN
    """
    from difflib import SequenceMatcher

    mots_err = _tokenize_simple(phrase_err)
    mots_cor = _tokenize_simple(phrase_cor)
    mots_v2 = _tokenize_simple(phrase_v2)

    # 1. Trouver les positions erronees (err != cor)
    # Aligner err <-> cor via SequenceMatcher
    gold_errors: set[int] = set()  # positions dans err qui sont erronees
    gold_fixes: dict[int, str] = {}  # err_pos -> forme corrigee
    sm_gold = SequenceMatcher(None, mots_err, mots_cor)
    for op, i1, i2, j1, j2 in sm_gold.get_opcodes():
        if op == "replace":
            for k in range(i1, i2):
                gold_errors.add(k)
                # Meilleur effort : mapper position par position
                offset = k - i1
                if j1 + offset < j2:
                    gold_fixes[k] = mots_cor[j1 + offset]
        elif op == "delete":
            for k in range(i1, i2):
                gold_errors.add(k)
        # 'insert' : mots manquants dans err — ignores

    # 2. Trouver les positions modifiees par v2 (err != v2)
    v2_changes: dict[int, str] = {}  # err_pos -> forme v2
    sm_v2 = SequenceMatcher(None, mots_err, mots_v2)
    for op, i1, i2, j1, j2 in sm_v2.get_opcodes():
        if op == "replace":
            for k in range(i1, i2):
                offset = k - i1
                if j1 + offset < j2:
                    v2_changes[k] = mots_v2[j1 + offset]
        elif op == "delete":
            for k in range(i1, i2):
                v2_changes[k] = ""  # supprime

    # 3. Compter VP/FP/FN
    vp = 0
    fp = 0
    fp_cible = 0       # mauvaise correction sur le mot cible
    fp_collateral = 0   # correction sur un mot non-cible
    fn = 0

    for pos in gold_errors:
        if pos in v2_changes:
            if pos in gold_fixes and v2_changes[pos] == gold_fixes[pos]:
                vp += 1  # correction correcte
            else:
                fp += 1  # mauvaise correction
                fp_cible += 1
        else:
            fn += 1  # erreur non corrigee

    # FP sur mots corrects (v2 modifie un mot qui n'est pas errone)
    for pos in v2_changes:
        if pos not in gold_errors:
            fp += 1
            fp_collateral += 1

    return {
        "VP": vp, "FP": fp, "FN": fn,
        "FP_cible": fp_cible, "FP_collateral": fp_collateral,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark CorrecteurV2 sur WiCoPaCo")
    parser.add_argument("--max", type=int, default=0, help="Max paires a traiter (0=toutes)")
    parser.add_argument("--expand", action="store_true", help="Afficher le detail par phrase")
    parser.add_argument("--negatif", action="store_true", help="Inclure le corpus negatif (FP)")
    parser.add_argument("--cat", type=str, default="", help="Filtrer par categorie (accord, conjugaison, homophone)")
    args = parser.parse_args()

    # Chemins des donnees
    corpus_dir = Path(__file__).resolve().parent.parent.parent.parent / "Corpus" / "Correcteur"
    wicopaco_path = corpus_dir / "grammaire_wicopaco.tsv"
    negatif_path = corpus_dir / "negatif_wicopaco.tsv"

    if not wicopaco_path.exists():
        print(f"ERREUR: {wicopaco_path} introuvable")
        sys.exit(1)

    # Charger les donnees
    paires = charger_wicopaco(str(wicopaco_path), args.max)
    if args.cat:
        paires = [(c, e, g) for c, e, g in paires if c == args.cat]
    print(f"WiCoPaCo: {len(paires)} paires chargees")

    negatif: list[str] = []
    if args.negatif and negatif_path.exists():
        negatif = charger_negatif(str(negatif_path), args.max)
        print(f"Negatif: {len(negatif)} phrases correctes chargees")

    # Creer le CorrecteurV2
    from lectura_correcteur._lexique_lite import LexiqueLite
    from lectura_correcteur.correcteur_v2 import CorrecteurV2

    db_path = Path(__file__).resolve().parent.parent / "src" / "lectura_correcteur" / "data" / "lexique_correcteur.db"
    if not db_path.exists():
        print(f"ERREUR: {db_path} introuvable")
        sys.exit(1)

    print("Chargement du lexique et des modeles...")
    lexique = LexiqueLite(db_path)
    v2 = CorrecteurV2(lexique)
    print("CorrecteurV2 pret.")

    # --- Benchmark sur WiCoPaCo ---
    totals: Counter = Counter()
    by_cat: dict[str, Counter] = defaultdict(Counter)
    by_rule: Counter = Counter()  # corrections par regle
    t0 = time.time()

    for idx, (cat, err, cor) in enumerate(paires):
        result = v2.corriger(err)
        stats = comparer_corrections(err, cor, result.phrase_corrigee)

        totals["VP"] += stats["VP"]
        totals["FP"] += stats["FP"]
        totals["FP_cible"] += stats["FP_cible"]
        totals["FP_collateral"] += stats["FP_collateral"]
        totals["FN"] += stats["FN"]
        totals["n_phrases"] += 1

        by_cat[cat]["VP"] += stats["VP"]
        by_cat[cat]["FP"] += stats["FP"]
        by_cat[cat]["FP_cible"] += stats["FP_cible"]
        by_cat[cat]["FP_collateral"] += stats["FP_collateral"]
        by_cat[cat]["FN"] += stats["FN"]
        by_cat[cat]["n"] += 1

        # Comptage des regles appliquees
        for corr in result.corrections:
            by_rule[corr.regle] += 1

        if args.expand and (stats["VP"] > 0 or stats["FP"] > 0):
            status = "OK" if stats["VP"] > 0 and stats["FP"] == 0 else "ERR"
            corrections_desc = []
            for corr in result.corrections:
                corrections_desc.append(f"    [{corr.regle}] {corr.original} -> {corr.corrige}")
            print(f"[{status}] [{cat}]")
            print(f"  ERR: {err}")
            print(f"  COR: {cor}")
            print(f"  V2:  {result.phrase_corrigee}")
            print(f"  VP={stats['VP']} FP={stats['FP']} (cible={stats['FP_cible']} collateral={stats['FP_collateral']}) FN={stats['FN']}")
            if corrections_desc:
                print(f"  Corrections:")
                for cd in corrections_desc:
                    print(cd)
            print()

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  ... {idx + 1}/{len(paires)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # --- Benchmark negatif (FP sur phrases correctes) ---
    fp_negatif = 0
    fp_mots_negatif = 0
    if negatif:
        t1 = time.time()
        for phrase in negatif:
            result = v2.corriger(phrase)
            if result.phrase_corrigee.lower() != phrase.lower():
                fp_negatif += 1
                fp_mots_negatif += result.n_corrections
        elapsed_neg = time.time() - t1
        print(f"\nNegatif: {elapsed_neg:.1f}s pour {len(negatif)} phrases")

    # --- Resultats ---
    print("\n" + "=" * 60)
    print("RESULTATS CORRECTEUR V2")
    print("=" * 60)

    vp = totals["VP"]
    fp = totals["FP"]
    fn = totals["FN"]

    precision = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    fp_cible = totals["FP_cible"]
    fp_collateral = totals["FP_collateral"]

    # Precision "stricte" : tous les FP comptent
    precision_s = vp / (vp + fp) if (vp + fp) > 0 else 0
    # Precision "cible" : seuls les FP sur le mot cible comptent
    precision_c = vp / (vp + fp_cible) if (vp + fp_cible) > 0 else 0

    print(f"\nGlobal ({totals['n_phrases']} phrases, {elapsed:.1f}s):")
    print(f"  VP (corrections correctes)       = {vp}")
    print(f"  FP total                         = {fp}")
    print(f"    FP cible (mauvaise correction) = {fp_cible}")
    print(f"    FP collateral (non-cible)      = {fp_collateral}")
    print(f"  FN (erreurs non corrigees)        = {fn}")
    print(f"  Precision stricte  = {precision_s:.1%}")
    print(f"  Precision cible    = {precision_c:.1%}")
    print(f"  Recall             = {recall:.1%}")
    print(f"  F1 (strict)        = {f1:.1%}")

    if by_rule:
        print(f"\nCorrections par regle:")
        for rule, count in sorted(by_rule.items(), key=lambda x: -x[1]):
            print(f"  {rule:<20} {count:>5}")

    if negatif:
        print(f"\nNegatif ({len(negatif)} phrases correctes):")
        print(f"  Phrases modifiees (FP) = {fp_negatif}")
        print(f"  Mots modifies (FP)     = {fp_mots_negatif}")

    print(f"\nPar categorie:")
    print(f"  {'Categorie':<15} {'N':>5} {'VP':>5} {'FP':>5} {'FPc':>5} {'FPx':>5} {'FN':>5} {'Prec':>7} {'PrecC':>7} {'Rec':>7}")
    print(f"  {'-'*15} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
    for cat in sorted(by_cat.keys()):
        c = by_cat[cat]
        p = c["VP"] / (c["VP"] + c["FP"]) if (c["VP"] + c["FP"]) > 0 else 0
        pc = c["VP"] / (c["VP"] + c["FP_cible"]) if (c["VP"] + c["FP_cible"]) > 0 else 0
        r = c["VP"] / (c["VP"] + c["FN"]) if (c["VP"] + c["FN"]) > 0 else 0
        print(f"  {cat:<15} {c['n']:>5} {c['VP']:>5} {c['FP']:>5} {c['FP_cible']:>5} {c['FP_collateral']:>5} {c['FN']:>5} {p:>6.1%} {pc:>6.1%} {r:>6.1%}")

    # Cibles
    print(f"\nCibles:")
    print(f"  Precision cible > 50%  : {'OK' if precision_c > 0.50 else 'FAIL'} ({precision_c:.1%})")
    print(f"  Precision strict > 50% : {'OK' if precision_s > 0.50 else 'FAIL'} ({precision_s:.1%})")
    if negatif:
        print(f"  FP negatif < 200       : {'OK' if fp_mots_negatif < 200 else 'FAIL'} ({fp_mots_negatif})")


if __name__ == "__main__":
    main()
