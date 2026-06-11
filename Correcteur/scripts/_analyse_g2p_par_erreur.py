#!/usr/bin/env python3
"""Analyse detaillee du comportement G2P par type d'erreur WiCoPaCo.

Pour chaque type d'erreur (accord, conjugaison, homophone) :
  - Compare le POS/MORPHO entre phrase erronee et corrigee
  - Mesure la confiance G2P au site d'erreur
  - Sous-classifie le type de changement (POS, Number, Gender, Person, Tense...)
  - Identifie les patterns exploitables pour la correction

Usage:
    python3 _analyse_g2p_par_erreur.py              # tout
    python3 _analyse_g2p_par_erreur.py --max 0      # pas de limite
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

WICOPACO_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/grammaire_wicopaco.tsv"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019][\w]+)*", re.UNICODE)


def tokeniser(phrase: str) -> list[str]:
    return [m.group() for m in _MOT_RE.finditer(phrase)]


def tronquer_mots(phrase: str, max_mots: int = 40) -> str:
    tokens = phrase.split()
    return " ".join(tokens[:max_mots]) if len(tokens) > max_mots else phrase


def charger_wicopaco(path: str, max_n: int = 0):
    paires = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3 or row[0].startswith("#") or row[0] == "type_erreur":
                continue
            cat, err, cor = row[0].strip(), row[1].strip(), row[2].strip()
            if err and cor:
                paires.append((cat, err, cor))
                if max_n and len(paires) >= max_n:
                    break
    return paires


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--max-mots", type=int, default=40)
    args = parser.parse_args()

    print("Chargement WiCoPaCo...")
    paires = charger_wicopaco(WICOPACO_TSV, max_n=args.max)
    cat_counts = Counter(c for c, _, _ in paires)
    for c, n in cat_counts.most_common():
        print(f"  {c}: {n}")

    print("\nChargement G2P Unifie V2...")
    from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
    tagger = creer_adapter_g2p_unifie()
    if not tagger:
        print("ERREUR: G2P indisponible")
        return
    print("  G2P charge")

    nombre_map = {"s": "Sing", "p": "Plur"}
    genre_map = {"m": "Masc", "f": "Fem"}

    def tag_phrase(mots):
        try:
            tags = tagger.tag_words_rich(mots)
        except Exception:
            tags = [{}] * len(mots)
        return [{
            "pos": g.get("pos", "NOM"),
            "number": nombre_map.get(g.get("nombre", ""), "_"),
            "gender": genre_map.get(g.get("genre", ""), "_"),
            "person": g.get("personne", "_") or "_",
            "confiance": g.get("confiance_pos", 0.0),
        } for g in tags]

    # ── Structures par categorie ──
    # Pour chaque categorie, on stocke les sites d'erreur avec leur analyse
    class SiteErreur:
        __slots__ = ("cat", "mot_err", "mot_cor", "tag_err", "tag_cor",
                     "pos_diff", "number_diff", "gender_diff", "person_diff",
                     "any_morpho_diff", "any_diff")

        def __init__(self, cat, mot_err, mot_cor, te, tc):
            self.cat = cat
            self.mot_err = mot_err
            self.mot_cor = mot_cor
            self.tag_err = te
            self.tag_cor = tc
            pb = lambda p: p.split(":")[0]
            self.pos_diff = pb(te["pos"]) != pb(tc["pos"])
            self.number_diff = te["number"] != tc["number"]
            self.gender_diff = te["gender"] != tc["gender"]
            self.person_diff = te["person"] != tc["person"]
            self.any_morpho_diff = self.number_diff or self.gender_diff or self.person_diff
            self.any_diff = self.pos_diff or self.any_morpho_diff

    sites_by_cat: dict[str, list[SiteErreur]] = defaultdict(list)
    conf_identiques: list[float] = []  # confiance sur mots identiques (baseline)
    n_skip = 0
    n_traite = 0

    t0 = time.time()
    for idx, (cat, phrase_err, phrase_cor) in enumerate(paires):
        if idx % 1000 == 0 and idx > 0:
            print(f"  ... {idx}/{len(paires)}", file=sys.stderr)

        mots_err = tokeniser(tronquer_mots(phrase_err, args.max_mots))
        mots_cor = tokeniser(tronquer_mots(phrase_cor, args.max_mots))

        if len(mots_err) != len(mots_cor) or not mots_err:
            n_skip += 1
            continue

        n_traite += 1
        tags_err = tag_phrase(mots_err)
        tags_cor = tag_phrase(mots_cor)

        for i in range(min(len(mots_err), len(tags_err), len(tags_cor))):
            me, mc = mots_err[i], mots_cor[i]
            te, tc = tags_err[i], tags_cor[i]

            if me.lower() != mc.lower():
                sites_by_cat[cat].append(SiteErreur(cat, me, mc, te, tc))
            else:
                conf_identiques.append(te["confiance"])

    elapsed = time.time() - t0
    print(f"\nAnalyse terminee ({elapsed:.1f}s)")
    print(f"  Paires traitees: {n_traite}, skip: {n_skip}")

    avg_conf_base = sum(conf_identiques) / len(conf_identiques) if conf_identiques else 0
    print(f"  Confiance baseline (mots identiques): {avg_conf_base:.4f}")

    # ── Rapport par categorie ──
    for cat in ["homophone", "accord", "conjugaison"]:
        sites = sites_by_cat.get(cat, [])
        if not sites:
            continue

        print(f"\n{'='*70}")
        print(f"  {cat.upper()} — {len(sites)} sites d'erreur")
        print(f"{'='*70}")

        # 1. Taux de detection
        n_pos = sum(1 for s in sites if s.pos_diff)
        n_num = sum(1 for s in sites if s.number_diff)
        n_gen = sum(1 for s in sites if s.gender_diff)
        n_per = sum(1 for s in sites if s.person_diff)
        n_morph = sum(1 for s in sites if s.any_morpho_diff)
        n_any = sum(1 for s in sites if s.any_diff)
        n_rien = sum(1 for s in sites if not s.any_diff)

        print(f"\n  Detection (tag different err vs cor):")
        print(f"    POS different    : {n_pos:5d}/{len(sites)} ({100*n_pos/len(sites):.1f}%)")
        print(f"    Number different : {n_num:5d}/{len(sites)} ({100*n_num/len(sites):.1f}%)")
        print(f"    Gender different : {n_gen:5d}/{len(sites)} ({100*n_gen/len(sites):.1f}%)")
        print(f"    Person different : {n_per:5d}/{len(sites)} ({100*n_per/len(sites):.1f}%)")
        print(f"    Tout morpho      : {n_morph:5d}/{len(sites)} ({100*n_morph/len(sites):.1f}%)")
        print(f"    QUELQUE CHOSE    : {n_any:5d}/{len(sites)} ({100*n_any/len(sites):.1f}%)")
        print(f"    RIEN (invisible) : {n_rien:5d}/{len(sites)} ({100*n_rien/len(sites):.1f}%)")

        # 2. Confiance
        conf_err = [s.tag_err["confiance"] for s in sites]
        conf_cor = [s.tag_cor["confiance"] for s in sites]
        avg_err = sum(conf_err) / len(conf_err)
        avg_cor = sum(conf_cor) / len(conf_cor)

        # Confiance quand erreur detectee vs pas detectee
        detected = [s for s in sites if s.any_diff]
        not_detected = [s for s in sites if not s.any_diff]
        avg_det_err = sum(s.tag_err["confiance"] for s in detected) / len(detected) if detected else 0
        avg_det_cor = sum(s.tag_cor["confiance"] for s in detected) / len(detected) if detected else 0
        avg_ndet_err = sum(s.tag_err["confiance"] for s in not_detected) / len(not_detected) if not_detected else 0
        avg_ndet_cor = sum(s.tag_cor["confiance"] for s in not_detected) / len(not_detected) if not_detected else 0

        print(f"\n  Confiance G2P:")
        print(f"    Tous sites   — err: {avg_err:.4f}  cor: {avg_cor:.4f}  delta: {avg_cor-avg_err:+.4f}")
        print(f"    Detectes     — err: {avg_det_err:.4f}  cor: {avg_det_cor:.4f}  delta: {avg_det_cor-avg_det_err:+.4f}")
        if not_detected:
            print(f"    Non detectes — err: {avg_ndet_err:.4f}  cor: {avg_ndet_cor:.4f}  delta: {avg_ndet_cor-avg_ndet_err:+.4f}")
        print(f"    Baseline (mots identiques): {avg_conf_base:.4f}")

        # 3. Distribution confiance erreur
        conf_bins = {"<0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, ">0.9": 0}
        for c in conf_err:
            if c < 0.5:
                conf_bins["<0.5"] += 1
            elif c < 0.7:
                conf_bins["0.5-0.7"] += 1
            elif c < 0.9:
                conf_bins["0.7-0.9"] += 1
            else:
                conf_bins[">0.9"] += 1
        print(f"\n  Distribution confiance (phrase err, au site d'erreur):")
        for b, n in conf_bins.items():
            print(f"    {b:10s}: {n:5d} ({100*n/len(sites):.1f}%)")

        # 4. Sous-classification des changements
        if cat == "homophone":
            print(f"\n  Changements POS (homophones):")
            pb = lambda p: p.split(":")[0]
            pos_pairs = Counter()
            for s in sites:
                if s.pos_diff:
                    pos_pairs[(pb(s.tag_err["pos"]), pb(s.tag_cor["pos"]))] += 1
            for (pe, pc), n in pos_pairs.most_common(20):
                print(f"    {pe:8s} → {pc:8s} : {n}")

            # Exemples detailles
            print(f"\n  Exemples (tous les {len(sites)} sites):")
            for i, s in enumerate(sites[:60]):
                mark = ""
                if s.pos_diff:
                    mark += " *POS*"
                if s.any_morpho_diff:
                    mark += " *MOR*"
                if not s.any_diff:
                    mark = " (invisible)"
                print(f"    {s.mot_err:15s}→{s.mot_cor:15s}"
                      f" POS:{s.tag_err['pos']:8s}→{s.tag_cor['pos']:8s}"
                      f" N:{s.tag_err['number']:5s}→{s.tag_cor['number']:5s}"
                      f" G:{s.tag_err['gender']:5s}→{s.tag_cor['gender']:5s}"
                      f" P:{s.tag_err['person']:2s}→{s.tag_cor['person']:2s}"
                      f" conf:{s.tag_err['confiance']:.3f}→{s.tag_cor['confiance']:.3f}"
                      f"{mark}")

        elif cat == "accord":
            # Sous-types : nombre seul, genre seul, nombre+genre, personne
            n_num_only = sum(1 for s in sites if s.number_diff and not s.gender_diff and not s.person_diff and not s.pos_diff)
            n_gen_only = sum(1 for s in sites if s.gender_diff and not s.number_diff and not s.person_diff and not s.pos_diff)
            n_per_only = sum(1 for s in sites if s.person_diff and not s.number_diff and not s.gender_diff and not s.pos_diff)
            n_num_gen = sum(1 for s in sites if s.number_diff and s.gender_diff and not s.pos_diff)
            n_pos_only = sum(1 for s in sites if s.pos_diff and not s.any_morpho_diff)

            print(f"\n  Sous-types de changement (accords):")
            print(f"    Number seul      : {n_num_only}")
            print(f"    Gender seul      : {n_gen_only}")
            print(f"    Person seul      : {n_per_only}")
            print(f"    Number + Gender  : {n_num_gen}")
            print(f"    POS seul         : {n_pos_only}")
            print(f"    Invisible        : {n_rien}")

            # Top transitions Number
            num_trans = Counter()
            for s in sites:
                if s.number_diff:
                    num_trans[(s.tag_err["number"], s.tag_cor["number"])] += 1
            if num_trans:
                print(f"\n  Transitions Number:")
                for (ne, nc), n in num_trans.most_common():
                    print(f"    {ne:6s} → {nc:6s} : {n}")

            # Top transitions Gender
            gen_trans = Counter()
            for s in sites:
                if s.gender_diff:
                    gen_trans[(s.tag_err["gender"], s.tag_cor["gender"])] += 1
            if gen_trans:
                print(f"\n  Transitions Gender:")
                for (ge, gc), n in gen_trans.most_common():
                    print(f"    {ge:6s} → {gc:6s} : {n}")

            # Exemples invisibles (pas de diff tag)
            invisibles = [s for s in sites if not s.any_diff]
            if invisibles:
                print(f"\n  Exemples INVISIBLES (meme tag) — premiers 20:")
                for s in invisibles[:20]:
                    print(f"    {s.mot_err:15s}→{s.mot_cor:15s}"
                          f" POS:{s.tag_err['pos']:8s}"
                          f" N:{s.tag_err['number']:5s} G:{s.tag_err['gender']:5s}"
                          f" conf:{s.tag_err['confiance']:.3f}→{s.tag_cor['confiance']:.3f}")

            # Exemples detectes
            print(f"\n  Exemples DETECTES — premiers 30:")
            for s in detected[:30]:
                mark = []
                if s.pos_diff:
                    mark.append("POS")
                if s.number_diff:
                    mark.append("N")
                if s.gender_diff:
                    mark.append("G")
                if s.person_diff:
                    mark.append("P")
                print(f"    {s.mot_err:15s}→{s.mot_cor:15s}"
                      f" POS:{s.tag_err['pos']:8s}→{s.tag_cor['pos']:8s}"
                      f" N:{s.tag_err['number']:5s}→{s.tag_cor['number']:5s}"
                      f" G:{s.tag_err['gender']:5s}→{s.tag_cor['gender']:5s}"
                      f" P:{s.tag_err['person']:2s}→{s.tag_cor['person']:2s}"
                      f" [{','.join(mark)}]")

        elif cat == "conjugaison":
            # POS change : VER→NOM, VER→ADJ etc
            print(f"\n  Sous-types (conjugaison):")
            n_per_conj = sum(1 for s in sites if s.person_diff and not s.pos_diff)
            n_num_conj = sum(1 for s in sites if s.number_diff and not s.pos_diff)
            n_pos_conj = sum(1 for s in sites if s.pos_diff)
            print(f"    Person change    : {n_per_conj}")
            print(f"    Number change    : {n_num_conj}")
            print(f"    POS change       : {n_pos_conj}")
            print(f"    Invisible        : {n_rien}")

            # Exemples
            print(f"\n  Exemples — premiers 30:")
            for i, s in enumerate(sites[:30]):
                mark = []
                if s.pos_diff:
                    mark.append("POS")
                if s.number_diff:
                    mark.append("N")
                if s.gender_diff:
                    mark.append("G")
                if s.person_diff:
                    mark.append("P")
                if not mark:
                    mark.append("invisible")
                print(f"    {s.mot_err:15s}→{s.mot_cor:15s}"
                      f" POS:{s.tag_err['pos']:8s}→{s.tag_cor['pos']:8s}"
                      f" N:{s.tag_err['number']:5s}→{s.tag_cor['number']:5s}"
                      f" P:{s.tag_err['person']:2s}→{s.tag_cor['person']:2s}"
                      f" conf:{s.tag_err['confiance']:.3f}→{s.tag_cor['confiance']:.3f}"
                      f" [{','.join(mark)}]")

    # ── Resume global ──
    print(f"\n{'='*70}")
    print(f"  RESUME GLOBAL")
    print(f"{'='*70}")
    print(f"\n  {'Categorie':<15s} {'Sites':>6s} {'Detecte':>8s} {'POS':>6s} {'Num':>6s} {'Gen':>6s} {'Per':>6s} {'Invis':>6s} {'conf_e':>7s} {'conf_c':>7s} {'delta':>7s}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

    for cat in ["homophone", "accord", "conjugaison"]:
        sites = sites_by_cat.get(cat, [])
        if not sites:
            continue
        n = len(sites)
        nd = sum(1 for s in sites if s.any_diff)
        np = sum(1 for s in sites if s.pos_diff)
        nn = sum(1 for s in sites if s.number_diff)
        ng = sum(1 for s in sites if s.gender_diff)
        npe = sum(1 for s in sites if s.person_diff)
        ni = n - nd
        ce = sum(s.tag_err["confiance"] for s in sites) / n
        cc = sum(s.tag_cor["confiance"] for s in sites) / n

        print(f"  {cat:<15s} {n:6d} {100*nd/n:7.1f}% {100*np/n:5.1f}% {100*nn/n:5.1f}% {100*ng/n:5.1f}% {100*npe/n:5.1f}% {100*ni/n:5.1f}% {ce:7.4f} {cc:7.4f} {cc-ce:+7.4f}")

    print(f"\n  Baseline confiance (mots identiques): {avg_conf_base:.4f}")
    print("\nTermine.")


if __name__ == "__main__":
    main()
