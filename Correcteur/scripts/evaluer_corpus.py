#!/usr/bin/env python3
"""Evaluation iterative du correcteur sur le corpus reel.

Charge un echantillon de phrases de corpus_10000.jsonl,
evalue le correcteur avec le vrai lexique, et calcule les metriques
token-level (Precision, Recall, F1, F0.5, taux FP).

Usage :
    python scripts/evaluer_corpus.py [--n 200] [--offset 0] [--seed 42] [--all-previous]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
CORPUS_PATH = os.path.join(_PROJECT_ROOT, "data", "corpus", "corpus_10000.jsonl")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "benchmark", "iterations")


# ---------------------------------------------------------------------------
# Metriques mot-a-mot
# ---------------------------------------------------------------------------

def _normaliser(texte: str) -> str:
    return " ".join(texte.strip().split())


def _build_word_map(src: list[str], dst: list[str]) -> dict[int, str | None]:
    # Align using lowercase to avoid case-sensitivity breaking the alignment
    sm = SequenceMatcher(None, [w.lower() for w in src], [w.lower() for w in dst])
    word_map: dict[int, str | None] = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                word_map[i1 + k] = src[i1 + k]
        elif op == "replace":
            n_src, n_dst = i2 - i1, j2 - j1
            for k in range(n_src):
                word_map[i1 + k] = dst[j1 + k] if k < n_dst else None
        elif op == "delete":
            for k in range(i2 - i1):
                word_map[i1 + k] = None
    return word_map


def _compter_insertions(src: list[str], dst: list[str]) -> list[str]:
    sm = SequenceMatcher(None, [w.lower() for w in src], [w.lower() for w in dst])
    insertions: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            insertions.extend(dst[j1:j2])
        elif op == "replace":
            n_extra = (j2 - j1) - (i2 - i1)
            if n_extra > 0:
                insertions.extend(dst[j2 - n_extra : j2])
    return insertions


def _norm_mot(mot: str) -> str:
    """Normalise un mot pour comparaison (minuscule, strip ponctuation terminale)."""
    return mot.lower().rstrip(".,;:!?")


def calculer_metriques_mots(
    original: str, attendu: str, obtenu: str,
) -> tuple[int, int, int, int]:
    orig = _normaliser(original).split()
    gold = _normaliser(attendu).split()
    syst = _normaliser(obtenu).split()

    gold_map = _build_word_map(orig, gold)
    sys_map = _build_word_map(orig, syst)

    tp = fp = fn = tn = 0
    for i, mot_orig in enumerate(orig):
        mot_gold = gold_map.get(i, mot_orig)
        mot_sys = sys_map.get(i, mot_orig)

        # Normaliser pour comparaison (case-insensitive, ignore ponctuation ajoutee)
        no = _norm_mot(mot_orig) if mot_orig else ""
        ng = _norm_mot(mot_gold) if mot_gold else ""
        ns = _norm_mot(mot_sys) if mot_sys else ""

        needs_change = (ng != no)
        was_changed = (ns != no)
        correct_change = (ns == ng)

        if needs_change and was_changed and correct_change:
            tp += 1
        elif needs_change:
            fn += 1
        elif not needs_change and was_changed:
            fp += 1
        else:
            tn += 1

    gold_ins = _compter_insertions(orig, gold)
    sys_ins = _compter_insertions(orig, syst)
    for mot in gold_ins:
        if mot in sys_ins:
            tp += 1
            sys_ins.remove(mot)
        else:
            fn += 1
    fp += len(sys_ins)

    return tp, fp, fn, tn


def _f_score(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


# ---------------------------------------------------------------------------
# Chargement du corpus
# ---------------------------------------------------------------------------

def charger_corpus(path: str, n: int, offset: int) -> list[dict]:
    """Charge n phrases a partir de offset."""
    phrases = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(phrases) >= n:
                break
            phrases.append(json.loads(line))
    return phrases


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class ResultatPhrase:
    idx: int
    fautif: str
    correct: str
    obtenu: str
    exact_match: bool
    tp: int
    fp: int
    fn: int
    tn: int
    types_erreur: list[str]
    erreurs_detail: list[dict]
    temps_ms: float


def evaluer_echantillon(
    correcteur, phrases: list[dict], offset: int = 0,
) -> list[ResultatPhrase]:
    resultats = []
    for i, p in enumerate(phrases):
        fautif = p["fautif"]
        correct = p["correct"]
        types = [e["type"] for e in p.get("erreurs", [])]

        t0 = time.perf_counter()
        r = correcteur.corriger(fautif)
        dt = (time.perf_counter() - t0) * 1000

        obtenu = r.phrase_corrigee
        obtenu_n = _normaliser(obtenu).lower()
        correct_n = _normaliser(correct).lower()

        tp, fp, fn, tn = calculer_metriques_mots(fautif, correct, obtenu)

        # Detail des erreurs non corrigees
        erreurs_detail = []
        if not (obtenu_n == correct_n):
            orig_mots = _normaliser(fautif).split()
            gold_mots = _normaliser(correct).split()
            syst_mots = _normaliser(obtenu).split()
            gold_map = _build_word_map(orig_mots, gold_mots)
            sys_map = _build_word_map(orig_mots, syst_mots)
            for j, mot_orig in enumerate(orig_mots):
                mot_gold = gold_map.get(j, mot_orig)
                mot_sys = sys_map.get(j, mot_orig)
                no = _norm_mot(mot_orig) if mot_orig else ""
                ng = _norm_mot(mot_gold) if mot_gold else ""
                ns = _norm_mot(mot_sys) if mot_sys else ""
                if ng != no and ns != ng:
                    # Trouver le type d'erreur pour cette position
                    err_type = "?"
                    for e in p.get("erreurs", []):
                        if e.get("position") == j:
                            err_type = e["type"]
                            break
                    erreurs_detail.append({
                        "pos": j, "orig": mot_orig,
                        "gold": mot_gold, "sys": mot_sys,
                        "type": err_type,
                    })
                elif ng == no and ns != no:
                    erreurs_detail.append({
                        "pos": j, "orig": mot_orig,
                        "gold": mot_gold, "sys": mot_sys,
                        "type": "FP",
                    })

        resultats.append(ResultatPhrase(
            idx=offset + i,
            fautif=fautif,
            correct=correct,
            obtenu=obtenu,
            exact_match=(obtenu_n == correct_n),
            tp=tp, fp=fp, fn=fn, tn=tn,
            types_erreur=types,
            erreurs_detail=erreurs_detail,
            temps_ms=dt,
        ))
    return resultats


def afficher_resultats(resultats: list[ResultatPhrase], label: str = ""):
    n = len(resultats)
    n_exact = sum(1 for r in resultats if r.exact_match)

    total_tp = sum(r.tp for r in resultats)
    total_fp = sum(r.fp for r in resultats)
    total_fn = sum(r.fn for r in resultats)
    total_tn = sum(r.tn for r in resultats)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = _f_score(precision, recall, 1.0)
    f05 = _f_score(precision, recall, 0.5)

    mots_corrects = total_tn + total_fp
    taux_fp = total_fp / mots_corrects if mots_corrects else 0.0

    temps_moy = sum(r.temps_ms for r in resultats) / n if n else 0

    print(f"\n{'=' * 70}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 70}")
    print(f"  Phrases     : {n}")
    print(f"  Exact match : {n_exact}/{n} ({100 * n_exact / n:.1f}%)")
    print(f"  Temps moyen : {temps_moy:.1f} ms/phrase")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}  TN={total_tn}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  F0.5        : {f05:.4f}")
    print(f"  Taux FP     : {taux_fp:.4f} ({total_fp}/{mots_corrects} mots corrects)")

    # Par type d'erreur
    par_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0, "exact": 0, "n": 0}
    )
    for r in resultats:
        for t in set(r.types_erreur):
            par_type[t]["n"] += 1
            par_type[t]["tp"] += r.tp
            par_type[t]["fn"] += r.fn
            par_type[t]["fp"] += r.fp
            if r.exact_match:
                par_type[t]["exact"] += 1

    if par_type:
        print(f"\n  {'Type':<10} {'Exact':>8} {'P':>8} {'R':>8} {'F1':>8}")
        print(f"  {'-' * 42}")
        for t in sorted(par_type, key=lambda x: par_type[x]["n"], reverse=True):
            s = par_type[t]
            p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0
            rc = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0
            f = _f_score(p, rc, 1.0)
            print(f"  {t:<10} {s['exact']:>3}/{s['n']:<3}  {p:>7.3f}  {rc:>7.3f}  {f:>7.3f}")

    # Top erreurs (FN et FP)
    fn_details: dict[str, int] = defaultdict(int)
    fp_details: dict[str, int] = defaultdict(int)
    for r in resultats:
        for e in r.erreurs_detail:
            if e["type"] == "FP":
                fp_details[f"{e['orig']!r} -> {e['sys']!r}"] += 1
            else:
                fn_details[f"[{e['type']}] {e['orig']!r} gold={e['gold']!r} got={e['sys']!r}"] += 1

    if fn_details:
        print(f"\n  Top 20 erreurs non corrigees (FN) :")
        for desc, count in sorted(fn_details.items(), key=lambda x: -x[1])[:20]:
            print(f"    {count:>3}x  {desc}")

    if fp_details:
        print(f"\n  Top 20 faux positifs (FP) :")
        for desc, count in sorted(fp_details.items(), key=lambda x: -x[1])[:20]:
            print(f"    {count:>3}x  {desc}")


def sauvegarder_resultats(resultats: list[ResultatPhrase], iteration: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{iteration}.json")
    data = [asdict(r) for r in resultats]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  Resultats sauvegardes: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluation iterative du correcteur")
    parser.add_argument("--n", type=int, default=200, help="Nombre de phrases")
    parser.add_argument("--offset", type=int, default=0, help="Offset dans le corpus")
    parser.add_argument("--label", type=str, default="", help="Label de l'iteration")
    parser.add_argument("--all-previous", action="store_true",
                        help="Re-evaluer aussi les echantillons precedents (non-regression)")
    parser.add_argument("--scoring", action="store_true", help="Activer le scoring multi-facteurs")
    parser.add_argument("--g2p", action="store_true", help="Injecter le G2P Unifie V2 comme phonetiseur")
    args = parser.parse_args()

    print("Chargement du lexique...", file=sys.stderr)
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig

    lexique = Lexique(LEXIQUE_DB)
    config = CorrecteurConfig(
        activer_scoring=args.scoring,
        activer_negation=False,  # Bug connu : desactive par defaut
    )

    g2p = None
    if args.g2p:
        print("Chargement du G2P Unifie V2...", file=sys.stderr)
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
        g2p = creer_adapter_g2p_unifie()
        if g2p is None:
            print("ERREUR: G2P Unifie V2 non disponible", file=sys.stderr)
            sys.exit(1)
        print("  G2P Unifie V2 charge", file=sys.stderr)

    correcteur = Correcteur(lexique, config=config, g2p=g2p)

    # Warmup
    correcteur.corriger("test")

    # Evaluer echantillon courant
    print(f"Chargement du corpus (offset={args.offset}, n={args.n})...", file=sys.stderr)
    phrases = charger_corpus(CORPUS_PATH, args.n, args.offset)
    print(f"  {len(phrases)} phrases chargees", file=sys.stderr)

    print("Evaluation en cours...", file=sys.stderr)
    resultats = evaluer_echantillon(correcteur, phrases, args.offset)

    label = args.label or f"offset{args.offset}_n{args.n}"
    afficher_resultats(resultats, label)
    sauvegarder_resultats(resultats, label)

    # Non-regression sur echantillons precedents
    if args.all_previous and os.path.exists(RESULTS_DIR):
        prev_files = sorted(Path(RESULTS_DIR).glob("*.json"))
        for pf in prev_files:
            if pf.stem == label:
                continue
            print(f"\n  Non-regression: {pf.stem}")
            with open(pf) as f:
                prev_data = json.load(f)
            # Re-evaluer les memes phrases
            prev_phrases = []
            for rd in prev_data:
                prev_phrases.append({
                    "fautif": rd["fautif"],
                    "correct": rd["correct"],
                    "erreurs": [],
                    "types_erreur": rd.get("types_erreur", []),
                })
            prev_resultats = evaluer_echantillon(correcteur, prev_phrases)
            # Comparer
            old_exact = sum(1 for rd in prev_data if rd["exact_match"])
            new_exact = sum(1 for r in prev_resultats if r.exact_match)
            old_fp = sum(rd["fp"] for rd in prev_data)
            new_fp = sum(r.fp for r in prev_resultats)
            delta_exact = new_exact - old_exact
            delta_fp = new_fp - old_fp
            status = "OK" if delta_exact >= 0 and delta_fp <= 0 else "REGRESSION"
            print(f"    Exact: {old_exact} -> {new_exact} ({'+' if delta_exact >= 0 else ''}{delta_exact})")
            print(f"    FP:    {old_fp} -> {new_fp} ({'+' if delta_fp >= 0 else ''}{delta_fp})")
            print(f"    Status: {status}")


if __name__ == "__main__":
    main()
