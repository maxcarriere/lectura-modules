#!/usr/bin/env python3
"""Benchmark unifie multi-corpus pour le correcteur Lectura.

Lance tous les benchmarks disponibles et produit un rapport consolide.

Corpus :
    1. FLE synthetique (data/grammaire_fle.tsv) — erreurs d'apprenants
    2. WiCoPaCo grammaire (data/grammaire_wicopaco.tsv) — corrections Wikipedia
    3. Negatif (data/negatif_wicopaco.tsv) — phrases correctes (FP pur)
    4. Built-in grammaire (22 cas integres) — tests de regression rapides

Usage :
    python scripts/benchmark_unifie.py
    python scripts/benchmark_unifie.py --max-wicopaco 200
    python scripts/benchmark_unifie.py --skip-wicopaco --skip-negatif
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


def _extraire_mots(texte: str) -> list[str]:
    return [m.group().lower() for m in _MOT_RE.finditer(texte)]


def _trouver_mot_cible(erronee: str, attendue: str) -> tuple[str, str, int] | None:
    mots_err = _extraire_mots(erronee)
    mots_att = _extraire_mots(attendue)
    if len(mots_err) != len(mots_att):
        return None
    for i, (a, b) in enumerate(zip(mots_err, mots_att)):
        if a != b:
            return (a, b, i)
    return None


def _tronquer_contexte(erronee: str, attendue: str, fenetre: int = 12):
    tokens_err = erronee.split()
    tokens_att = attendue.split()
    if len(tokens_err) != len(tokens_att):
        return None
    idx = None
    for i, (a, b) in enumerate(zip(tokens_err, tokens_att)):
        if a != b:
            idx = i
            break
    if idx is None:
        return None
    start = max(0, idx - fenetre)
    end = min(len(tokens_err), idx + fenetre + 1)
    return (
        " ".join(tokens_err[start:end]),
        " ".join(tokens_att[start:end]),
    )


def _chercher_mot(mots_obtenu, idx, mot_att, mot_err):
    for offset in (0, -1, 1, -2, 2, -3, 3):
        j = idx + offset
        if 0 <= j < len(mots_obtenu):
            if mots_obtenu[j] == mot_att or mots_obtenu[j] == mot_err:
                return mots_obtenu[j]
    return None


def _normaliser(texte: str) -> str:
    return " ".join(texte.strip().lower().split())


@dataclass
class Resultats:
    total: int = 0
    ok: int = 0
    fn: int = 0
    wrong: int = 0
    fp: int = 0
    skip: int = 0

    @property
    def precision(self) -> float:
        tp = self.ok
        fp_tot = self.wrong + self.fp
        return tp / (tp + fp_tot) if (tp + fp_tot) > 0 else 1.0

    @property
    def recall(self) -> float:
        tp = self.ok
        return tp / (tp + self.fn) if (tp + self.fn) > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def f05(self) -> float:
        p, r = self.precision, self.recall
        return (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0


def charger_tsv(path: str) -> list[tuple[str, str, str]]:
    paires = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            paires.append((row[0].strip(), row[1].strip(), row[2].strip()))
    return paires


# ---------------------------------------------------------------------------
# Evaluateurs
# ---------------------------------------------------------------------------

def evaluer_phrase(correcteur, paires, label=""):
    """Evaluation en mode phrase (pour corpus courts type FLE)."""
    res = Resultats()
    par_cat: dict[str, Resultats] = {}
    for cat, erronee, attendue in paires:
        if cat not in par_cat:
            par_cat[cat] = Resultats()
        r = par_cat[cat]
        res.total += 1
        r.total += 1

        obtenu = correcteur.corriger(erronee).phrase_corrigee
        obtenu_n = _normaliser(obtenu)
        attendu_n = _normaliser(attendue)
        erronee_n = _normaliser(erronee)

        correction_attendue = (attendu_n != erronee_n)
        correction_faite = (obtenu_n != erronee_n)

        if obtenu_n == attendu_n:
            res.ok += 1
            r.ok += 1
        elif correction_attendue and not correction_faite:
            res.fn += 1
            r.fn += 1
        elif not correction_attendue and correction_faite:
            res.fp += 1
            r.fp += 1
        else:
            res.wrong += 1
            r.wrong += 1

    par_cat["TOTAL"] = res
    return par_cat


def evaluer_mot(correcteur, paires, label=""):
    """Evaluation en mode mot (pour corpus longs type WiCoPaCo)."""
    res = Resultats()
    par_cat: dict[str, Resultats] = {}
    for cat, erronee, attendue in paires:
        if cat not in par_cat:
            par_cat[cat] = Resultats()
        r = par_cat[cat]
        res.total += 1
        r.total += 1

        tronque = _tronquer_contexte(erronee, attendue)
        if tronque is None:
            res.skip += 1
            r.skip += 1
            continue
        err_ctx, att_ctx = tronque
        cible = _trouver_mot_cible(err_ctx, att_ctx)
        if cible is None:
            res.skip += 1
            r.skip += 1
            continue

        mot_err, mot_att, idx = cible
        try:
            obtenu = correcteur.corriger(err_ctx).phrase_corrigee
        except Exception:
            res.skip += 1
            r.skip += 1
            continue

        mots_obtenu = _extraire_mots(obtenu)
        mot_trouve = _chercher_mot(mots_obtenu, idx, mot_att, mot_err)

        if mot_trouve == mot_att:
            res.ok += 1
            r.ok += 1
        elif mot_trouve == mot_err:
            res.fn += 1
            r.fn += 1
        else:
            res.wrong += 1
            r.wrong += 1

    par_cat["TOTAL"] = res
    return par_cat


def evaluer_negatif(correcteur, paires, label=""):
    """Evaluation en mode negatif (phrases correctes, mesure FP)."""
    res = Resultats()
    for _, phrase, _ in paires:
        res.total += 1
        try:
            resultat = correcteur.corriger(phrase)
        except Exception:
            res.skip += 1
            continue
        # Filtrer les corrections de majuscule/ponctuation (pas grammar)
        grammar_corrections = [
            c for c in resultat.corrections
            if c.regle not in ("syntaxe.majuscule",)
            and not c.regle.startswith("ponctuation")
        ]
        if grammar_corrections:
            res.fp += 1
        else:
            res.ok += 1
    return {"TOTAL": res}


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def afficher_resultats(label: str, par_cat: dict[str, Resultats], elapsed: float):
    categories = sorted(k for k in par_cat if k != "TOTAL")
    if categories:
        categories.append("TOTAL")
    else:
        categories = ["TOTAL"]

    header = (f"{'Categorie':15s} {'OK':>5s} {'FN':>5s} {'Wrong':>5s}"
              f" {'FP':>5s} {'Total':>5s}"
              f" | {'P':>5s} {'R':>5s} {'F1':>5s} {'F0.5':>5s}")
    print(f"\n--- {label} ({elapsed:.1f}s) ---")
    print(header)
    print("-" * len(header))
    for cat in categories:
        r = par_cat[cat]
        print(f"{cat:15s} {r.ok:>5d} {r.fn:>5d} {r.wrong:>5d}"
              f" {r.fp:>5d} {r.total:>5d}"
              f" | {r.precision:>.3f} {r.recall:>.3f}"
              f" {r.f1:>.3f} {r.f05:>.3f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark unifie")
    parser.add_argument("--max-wicopaco", type=int, default=200,
                        help="Max paires par cat WiCoPaCo (0=illimite)")
    parser.add_argument("--max-negatif", type=int, default=500,
                        help="Max phrases negatives (0=illimite)")
    parser.add_argument("--skip-fle", action="store_true")
    parser.add_argument("--skip-wicopaco", action="store_true")
    parser.add_argument("--skip-negatif", action="store_true")
    parser.add_argument("--skip-builtin", action="store_true")
    args = parser.parse_args()

    # Charger correcteur
    sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    lexique = Lexique(LEXIQUE_DB)
    correcteur = Correcteur(lexique)

    data_dir = os.path.join(_PROJECT_ROOT, "data")
    all_results: dict[str, dict[str, Resultats]] = {}

    print(f"{'='*70}")
    print("BENCHMARK UNIFIE — Correcteur Lectura")
    print(f"{'='*70}")

    # 1. Built-in grammaire
    if not args.skip_builtin:
        from scripts.evaluer_corpus_grammaire import (
            _CAS_CONJUGAISON, _CAS_HOMOPHONES, _CAS_ACCORDS,
        )
        builtin = []
        for err, att in _CAS_CONJUGAISON:
            builtin.append(("conjugaison", err, att))
        for err, att in _CAS_HOMOPHONES:
            builtin.append(("homophone", err, att))
        for err, att in _CAS_ACCORDS:
            builtin.append(("accord", err, att))
        t0 = time.time()
        res = evaluer_phrase(correcteur, builtin, "Built-in")
        all_results["Built-in"] = res
        afficher_resultats("Built-in grammaire", res, time.time() - t0)

    # 2. FLE synthetique
    fle_path = os.path.join(data_dir, "grammaire_fle.tsv")
    if not args.skip_fle and os.path.exists(fle_path):
        paires = charger_tsv(fle_path)
        t0 = time.time()
        res = evaluer_phrase(correcteur, paires, "FLE")
        all_results["FLE"] = res
        afficher_resultats("FLE synthetique", res, time.time() - t0)

    # 3. WiCoPaCo grammaire
    wico_path = os.path.join(data_dir, "grammaire_wicopaco.tsv")
    if not args.skip_wicopaco and os.path.exists(wico_path):
        paires = charger_tsv(wico_path)
        if args.max_wicopaco > 0:
            par_cat: dict[str, list] = {}
            for p in paires:
                par_cat.setdefault(p[0], []).append(p)
            paires = []
            for cat, items in sorted(par_cat.items()):
                paires.extend(items[:args.max_wicopaco])
        t0 = time.time()
        res = evaluer_mot(correcteur, paires, "WiCoPaCo")
        all_results["WiCoPaCo"] = res
        afficher_resultats(f"WiCoPaCo (max {args.max_wicopaco}/cat)", res,
                           time.time() - t0)

    # 4. Negatif
    neg_path = os.path.join(data_dir, "negatif_wicopaco.tsv")
    if not args.skip_negatif and os.path.exists(neg_path):
        paires = charger_tsv(neg_path)
        if args.max_negatif > 0:
            paires = paires[:args.max_negatif]
        t0 = time.time()
        res = evaluer_negatif(correcteur, paires, "Negatif")
        all_results["Negatif"] = res
        afficher_resultats(f"Negatif ({len(paires)} phrases)", res,
                           time.time() - t0)

    # Resume global
    print(f"\n{'='*70}")
    print("RESUME")
    print(f"{'='*70}")
    header = (f"{'Corpus':20s} {'OK':>5s} {'FN':>5s} {'Wrong':>5s}"
              f" {'FP':>5s} {'Total':>5s}"
              f" | {'P':>5s} {'R':>5s} {'F1':>5s} {'F0.5':>5s}")
    print(header)
    print("-" * len(header))
    for name, par_cat in all_results.items():
        r = par_cat["TOTAL"]
        print(f"{name:20s} {r.ok:>5d} {r.fn:>5d} {r.wrong:>5d}"
              f" {r.fp:>5d} {r.total:>5d}"
              f" | {r.precision:>.3f} {r.recall:>.3f}"
              f" {r.f1:>.3f} {r.f05:>.3f}")


if __name__ == "__main__":
    main()
