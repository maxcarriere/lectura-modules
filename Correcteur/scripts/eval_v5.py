#!/usr/bin/env python3
"""Evaluation V5 — Script iteratif pour le correcteur V5 P2G.

Protocole du plan :
  - 60 phrases FLE (grammaire_fle.tsv, aleatoires)
  - 40 phrases negatives courtes (negatif_wicopaco.tsv, < 120 chars)
  - 100 phrases GEC (fr_gec_akufeldt.tsv, < 150 chars, mix categories)

Usage :
    python scripts/eval_v5.py                    # batch 200 aleatoire
    python scripts/eval_v5.py --seed 42          # reproductible
    python scripts/eval_v5.py --fle-only         # FLE uniquement
    python scripts/eval_v5.py --all-fle          # tout le corpus FLE (299)
    python scripts/eval_v5.py --details          # afficher FN/FP/WRONG
    python scripts/eval_v5.py --details --limit-details 50
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Phonemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Graphemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Tokeniseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/G2P-Pipeline/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Formules/src")

LEXIQUE_DB = os.path.join(
    _PROJECT_ROOT, "src", "lectura_correcteur", "data", "lexique_correcteur.db",
)
CORPUS_DIR = "/home/moi/Documents/work/projets/lectura/workspace/Corpus/Correcteur"


@dataclass
class Resultats:
    total: int = 0
    ok: int = 0
    fn: int = 0
    wrong: int = 0
    fp: int = 0
    details: list[dict] = field(default_factory=list)

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


def _normaliser(texte: str) -> str:
    return " ".join(texte.strip().lower().split())


def charger_tsv(path: str) -> list[tuple[str, str, str]]:
    paires = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            err = row[1].strip()
            # Strip "fix grammar: " prefix from GEC corpora
            if err.startswith("fix grammar: "):
                err = err[len("fix grammar: "):]
            paires.append((row[0].strip(), err, row[2].strip()))
    return paires


def piocher_batch(
    seed: int,
    n_fle: int = 60,
    n_neg: int = 40,
    n_gec: int = 100,
    max_len_neg: int = 120,
    max_len_gec: int = 150,
) -> list[tuple[str, str, str, str]]:
    """Pioche un batch selon le protocole du plan.

    Returns:
        List of (source, cat, erronee, attendue)
    """
    rng = random.Random(seed)
    batch: list[tuple[str, str, str, str]] = []

    # 1. FLE
    fle_path = os.path.join(CORPUS_DIR, "grammaire_fle.tsv")
    if os.path.exists(fle_path):
        fle = charger_tsv(fle_path)
        sample = rng.sample(fle, min(n_fle, len(fle)))
        for cat, err, att in sample:
            batch.append(("FLE", cat, err, att))

    # 2. Negatifs courts
    neg_path = os.path.join(CORPUS_DIR, "negatif_wicopaco.tsv")
    if os.path.exists(neg_path):
        neg_all = charger_tsv(neg_path)
        neg_courts = [(c, e, a) for c, e, a in neg_all if len(e) < max_len_neg]
        sample = rng.sample(neg_courts, min(n_neg, len(neg_courts)))
        for cat, err, att in sample:
            batch.append(("NEG", cat, err, att))

    # 3. GEC (mix categories)
    gec_path = os.path.join(CORPUS_DIR, "fr_gec_akufeldt.tsv")
    if os.path.exists(gec_path):
        gec_all = charger_tsv(gec_path)
        gec_courts = [(c, e, a) for c, e, a in gec_all if len(e) < max_len_gec]
        # Mix categories : piocher proportionnellement
        par_cat: dict[str, list] = {}
        for item in gec_courts:
            par_cat.setdefault(item[0], []).append(item)
        # Piocher de chaque categorie
        n_cats = len(par_cat)
        per_cat = max(1, n_gec // n_cats) if n_cats > 0 else 0
        gec_sample = []
        for cat_items in par_cat.values():
            rng.shuffle(cat_items)
            gec_sample.extend(cat_items[:per_cat])
        rng.shuffle(gec_sample)
        gec_sample = gec_sample[:n_gec]
        for cat, err, att in gec_sample:
            batch.append(("GEC", cat, err, att))

    return batch


def evaluer_batch(
    correcteur,
    batch: list[tuple[str, str, str, str]],
) -> dict[str, Resultats]:
    """Evalue un batch sur le correcteur V5.

    Returns:
        Dict source -> Resultats + "TOTAL"
    """
    par_source: dict[str, Resultats] = {}
    total = Resultats()

    for source, cat, erronee, attendue in batch:
        if source not in par_source:
            par_source[source] = Resultats()
        res = par_source[source]

        total.total += 1
        res.total += 1

        try:
            resultat = correcteur.corriger(erronee)
            obtenu = resultat.phrase_corrigee
        except Exception as e:
            total.wrong += 1
            res.wrong += 1
            total.details.append({
                "type": "ERROR", "source": source, "cat": cat,
                "erronee": erronee, "attendue": attendue, "obtenu": str(e),
            })
            continue

        obtenu_n = _normaliser(obtenu)
        attendu_n = _normaliser(attendue)
        erronee_n = _normaliser(erronee)

        correction_attendue = (attendu_n != erronee_n)
        correction_faite = (obtenu_n != erronee_n)

        if obtenu_n == attendu_n:
            total.ok += 1
            res.ok += 1
        elif correction_attendue and not correction_faite:
            total.fn += 1
            res.fn += 1
            detail = {
                "type": "FN", "source": source, "cat": cat,
                "erronee": erronee, "attendue": attendue, "obtenu": obtenu,
            }
            total.details.append(detail)
            res.details.append(detail)
        elif not correction_attendue and correction_faite:
            total.fp += 1
            res.fp += 1
            detail = {
                "type": "FP", "source": source, "cat": cat,
                "erronee": erronee, "attendue": attendue, "obtenu": obtenu,
            }
            total.details.append(detail)
            res.details.append(detail)
        else:
            total.wrong += 1
            res.wrong += 1
            detail = {
                "type": "WRONG", "source": source, "cat": cat,
                "erronee": erronee, "attendue": attendue, "obtenu": obtenu,
            }
            total.details.append(detail)
            res.details.append(detail)

    par_source["TOTAL"] = total
    return par_source


def evaluer_negatif(correcteur, batch):
    """Evalue les phrases negatives (FP pur)."""
    res = Resultats()
    for source, cat, erronee, attendue in batch:
        if source != "NEG":
            continue
        res.total += 1
        try:
            resultat = correcteur.corriger(erronee)
        except Exception:
            res.wrong += 1
            continue
        obtenu_n = _normaliser(resultat.phrase_corrigee)
        erronee_n = _normaliser(erronee)
        if obtenu_n != erronee_n:
            res.fp += 1
            res.details.append({
                "type": "FP", "source": source, "cat": cat,
                "erronee": erronee, "obtenu": resultat.phrase_corrigee,
            })
        else:
            res.ok += 1
    return res


def afficher(label: str, par_source: dict[str, Resultats], elapsed: float):
    print(f"\n{'='*70}")
    print(f"  {label}  ({elapsed:.1f}s)")
    print(f"{'='*70}")

    header = f"{'Source':8s} {'OK':>5s} {'FN':>5s} {'WR':>5s} {'FP':>5s} {'Tot':>5s} | {'P':>5s} {'R':>5s} {'F1':>5s} {'F0.5':>5s}"
    print(header)
    print("-" * len(header))

    for src in sorted(par_source.keys()):
        if src == "TOTAL":
            continue
        r = par_source[src]
        print(f"{src:8s} {r.ok:>5d} {r.fn:>5d} {r.wrong:>5d} {r.fp:>5d} {r.total:>5d}"
              f" | {r.precision:>.3f} {r.recall:>.3f} {r.f1:>.3f} {r.f05:>.3f}")

    if "TOTAL" in par_source:
        r = par_source["TOTAL"]
        print("-" * len(header))
        print(f"{'TOTAL':8s} {r.ok:>5d} {r.fn:>5d} {r.wrong:>5d} {r.fp:>5d} {r.total:>5d}"
              f" | {r.precision:>.3f} {r.recall:>.3f} {r.f1:>.3f} {r.f05:>.3f}")


def afficher_details(details: list[dict], limit: int = 30):
    if not details:
        return
    print(f"\n--- Details ({len(details)} erreurs, affichage {min(limit, len(details))}) ---")
    for d in details[:limit]:
        typ = d["type"]
        src = d.get("source", "?")
        cat = d.get("cat", "?")
        err = d.get("erronee", "")[:80]
        att = d.get("attendue", "")[:80]
        obt = d.get("obtenu", "")[:80]
        print(f"  [{typ}] ({src}/{cat})")
        print(f"    err: {err}")
        if att:
            print(f"    att: {att}")
        print(f"    obt: {obt}")


def afficher_par_categorie(par_source: dict[str, Resultats], batch):
    """Affiche les resultats par categorie d'erreur (FLE)."""
    par_cat: dict[str, Resultats] = {}
    # Rebuild from details
    details = par_source.get("TOTAL", Resultats()).details
    for d in details:
        cat = d.get("cat", "?")
        if cat not in par_cat:
            par_cat[cat] = Resultats()
        r = par_cat[cat]
        r.total += 1
        if d["type"] == "FN":
            r.fn += 1
        elif d["type"] == "FP":
            r.fp += 1
        elif d["type"] == "WRONG":
            r.wrong += 1

    # Count OKs from batch
    cat_totals: dict[str, int] = {}
    for source, cat, err, att in batch:
        cat_totals[cat] = cat_totals.get(cat, 0) + 1

    print("\n--- Par categorie (erreurs seulement) ---")
    for cat in sorted(par_cat.keys()):
        r = par_cat[cat]
        total_cat = cat_totals.get(cat, r.total)
        n_err = r.fn + r.wrong + r.fp
        print(f"  {cat:35s}: {n_err} erreurs / {total_cat} phrases")


def main():
    parser = argparse.ArgumentParser(description="Eval V5 iteratif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-fle", type=int, default=60)
    parser.add_argument("--n-neg", type=int, default=40)
    parser.add_argument("--n-gec", type=int, default=100)
    parser.add_argument("--all-fle", action="store_true",
                        help="Evaluer tout le corpus FLE (ignore n-fle)")
    parser.add_argument("--fle-only", action="store_true",
                        help="FLE uniquement (pas de NEG ni GEC)")
    parser.add_argument("--details", action="store_true",
                        help="Afficher les details des erreurs")
    parser.add_argument("--limit-details", type=int, default=30)
    parser.add_argument("--categories", action="store_true",
                        help="Afficher par categorie")
    parser.add_argument("--v1", action="store_true",
                        help="Utiliser V1 au lieu de V5 (pour comparer)")
    args = parser.parse_args()

    # Charger correcteur
    from lectura_correcteur._lexique_lite import LexiqueLite
    lexique = LexiqueLite(LEXIQUE_DB)

    if args.v1:
        from lectura_correcteur import Correcteur
        correcteur = Correcteur(lexique)
        label = "V1 (reference)"
    else:
        from lectura_correcteur import CorrecteurV5, CorrecteurV5Config
        cfg = CorrecteurV5Config()
        correcteur = CorrecteurV5(lexique, config=cfg)
        label = f"V5 (P2G={'ON' if correcteur.p2g_disponible else 'OFF'})"

    print(f"Correcteur: {label}")

    # Piocher le batch
    if args.all_fle:
        fle_path = os.path.join(CORPUS_DIR, "grammaire_fle.tsv")
        fle = charger_tsv(fle_path)
        batch = [("FLE", cat, err, att) for cat, err, att in fle]
        if not args.fle_only:
            neg_path = os.path.join(CORPUS_DIR, "negatif_wicopaco.tsv")
            neg_all = charger_tsv(neg_path)
            neg_courts = [(c, e, a) for c, e, a in neg_all if len(e) < 120]
            rng = random.Random(args.seed)
            neg_sample = rng.sample(neg_courts, min(100, len(neg_courts)))
            for cat, err, att in neg_sample:
                batch.append(("NEG", cat, err, att))
    elif args.fle_only:
        batch = piocher_batch(args.seed, n_fle=args.n_fle, n_neg=0, n_gec=0)
    else:
        batch = piocher_batch(args.seed, n_fle=args.n_fle, n_neg=args.n_neg, n_gec=args.n_gec)

    print(f"Batch: {len(batch)} phrases (seed={args.seed})")

    # Evaluer
    t0 = time.time()
    par_source = evaluer_batch(correcteur, batch)
    elapsed = time.time() - t0

    afficher(label, par_source, elapsed)

    # Negatifs separes
    neg_res = evaluer_negatif(correcteur, batch)
    if neg_res.total > 0:
        print(f"\nNegatifs: {neg_res.ok}/{neg_res.total} OK"
              f" (FP={neg_res.fp}, taux FP={neg_res.fp/neg_res.total*100:.1f}%)")

    if args.details:
        total = par_source.get("TOTAL", Resultats())
        afficher_details(total.details, args.limit_details)

    if args.categories:
        afficher_par_categorie(par_source, batch)


if __name__ == "__main__":
    main()
