#!/usr/bin/env python3
"""Analyse n-gram PM par type d'erreur WiCoPaCo.

Meme structure que _analyse_g2p_par_erreur.py, mais ajoute le score
n-gram local autour du site d'erreur. Pour chaque sous-type de
changement d'etiquette (POS, Number, Gender, Person, invisible),
on mesure si le n-gram prefere la sequence correcte ou erronee.

Usage:
    python3 _analyse_ngram_par_erreur.py
    python3 _analyse_ngram_par_erreur.py --max 2000
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


def make_pm_tag(tag: dict) -> str:
    return f"{tag['pos']}|{tag['number']}|{tag['gender']}|{tag['person']}"


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

    print("\nChargement G2P...")
    from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
    tagger = creer_adapter_g2p_unifie()

    print("Chargement n-gram PM...")
    from lectura_correcteur._pos_ngram import PosNgram
    pos_ngram_db = os.path.join(
        _PROJECT_ROOT, "src", "lectura_correcteur", "data", "pos_ngram.db",
    )
    pos_ngram = PosNgram(pos_ngram_db)

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

    def ngram_local_score(tags, idx):
        """Score n-gram local : trigrammes touchant la position idx."""
        pm_seq = [make_pm_tag(t) for t in tags]
        BOS = "<BOS>"
        padded = [BOS, BOS] + pm_seq
        # Position dans padded = idx + 2
        # Trigrammes touchant idx : positions idx+2, idx+3, idx+4 dans padded
        total = 0.0
        for k in range(max(2, idx), min(len(padded), idx + 5)):
            total += pos_ngram.logp_pm_trigram(padded[k-2], padded[k-1], padded[k])
        return total

    # ── Structures ──
    class SiteErreur:
        __slots__ = ("cat", "mot_err", "mot_cor", "tag_err", "tag_cor",
                     "pos_diff", "number_diff", "gender_diff", "person_diff",
                     "any_morpho_diff", "any_diff",
                     "ng_local_err", "ng_local_cor", "ng_delta")

        def __init__(self, cat, mot_err, mot_cor, te, tc,
                     ng_err, ng_cor):
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
            self.ng_local_err = ng_err
            self.ng_local_cor = ng_cor
            self.ng_delta = ng_cor - ng_err

    sites_by_cat: dict[str, list[SiteErreur]] = defaultdict(list)
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
            if mots_err[i].lower() != mots_cor[i].lower():
                ng_err = ngram_local_score(tags_err, i)
                ng_cor = ngram_local_score(tags_cor, i)
                sites_by_cat[cat].append(SiteErreur(
                    cat, mots_err[i], mots_cor[i],
                    tags_err[i], tags_cor[i],
                    ng_err, ng_cor,
                ))

    elapsed = time.time() - t0
    print(f"\nAnalyse terminee ({elapsed:.1f}s), skip={n_skip}, traite={n_traite}")

    # ── Helper pour stats n-gram sur un sous-ensemble de sites ──
    def ng_stats(sites_list):
        if not sites_list:
            return None
        n = len(sites_list)
        deltas = [s.ng_delta for s in sites_list]
        avg = sum(deltas) / n
        prefer_cor = sum(1 for d in deltas if d > 0.01)
        prefer_err = sum(1 for d in deltas if d < -0.01)
        equal = n - prefer_cor - prefer_err

        conf_err = [s.tag_err["confiance"] for s in sites_list]
        conf_cor = [s.tag_cor["confiance"] for s in sites_list]
        avg_ce = sum(conf_err) / n
        avg_cc = sum(conf_cor) / n

        return {
            "n": n,
            "ng_avg": avg,
            "ng_cor": prefer_cor, "ng_err": prefer_err, "ng_eq": equal,
            "conf_err": avg_ce, "conf_cor": avg_cc,
        }

    def print_ng_row(label, stats, total_n=None):
        if not stats:
            return
        n = stats["n"]
        pct = f"({100*n/total_n:.0f}%)" if total_n else ""
        print(f"    {label:25s} {n:5d} {pct:6s}"
              f"  cor:{stats['ng_cor']:4d}({100*stats['ng_cor']/n:5.1f}%)"
              f"  err:{stats['ng_err']:4d}({100*stats['ng_err']/n:5.1f}%)"
              f"  eq:{stats['ng_eq']:4d}({100*stats['ng_eq']/n:5.1f}%)"
              f"  delta:{stats['ng_avg']:+.2f}"
              f"  conf:{stats['conf_err']:.3f}→{stats['conf_cor']:.3f}"
              f"({stats['conf_cor']-stats['conf_err']:+.3f})")

    # ── Rapport par categorie ──
    for cat in ["homophone", "accord", "conjugaison"]:
        sites = sites_by_cat.get(cat, [])
        if not sites:
            continue
        n = len(sites)

        print(f"\n{'='*70}")
        print(f"  {cat.upper()} — {n} sites d'erreur")
        print(f"{'='*70}")

        # Global
        print(f"\n  N-gram local par sous-type de changement d'etiquette:")
        print(f"  {'':25s} {'N':>5s} {'':6s}  {'prefere cor':>15s}  {'prefere err':>15s}  {'egal':>15s}  {'delta':>7s}  {'confiance G2P':>20s}")

        all_stats = ng_stats(sites)
        print_ng_row("TOUS", all_stats, n)

        # Par type de changement
        pos_sites = [s for s in sites if s.pos_diff]
        num_sites = [s for s in sites if s.number_diff]
        gen_sites = [s for s in sites if s.gender_diff]
        per_sites = [s for s in sites if s.person_diff]
        morpho_sites = [s for s in sites if s.any_morpho_diff]
        detected_sites = [s for s in sites if s.any_diff]
        invis_sites = [s for s in sites if not s.any_diff]

        print()
        print_ng_row("POS different", ng_stats(pos_sites), n)
        print_ng_row("Number different", ng_stats(num_sites), n)
        print_ng_row("Gender different", ng_stats(gen_sites), n)
        print_ng_row("Person different", ng_stats(per_sites), n)
        print_ng_row("Morpho (any)", ng_stats(morpho_sites), n)
        print()
        print_ng_row("DETECTE (any diff)", ng_stats(detected_sites), n)
        print_ng_row("INVISIBLE (same tag)", ng_stats(invis_sites), n)

        # Sous-types mutuellement exclusifs
        pos_only = [s for s in sites if s.pos_diff and not s.any_morpho_diff]
        num_only = [s for s in sites if s.number_diff and not s.gender_diff and not s.person_diff and not s.pos_diff]
        gen_only = [s for s in sites if s.gender_diff and not s.number_diff and not s.person_diff and not s.pos_diff]
        per_only = [s for s in sites if s.person_diff and not s.number_diff and not s.gender_diff and not s.pos_diff]
        num_gen = [s for s in sites if s.number_diff and s.gender_diff and not s.pos_diff]
        pos_morpho = [s for s in sites if s.pos_diff and s.any_morpho_diff]

        print(f"\n  Sous-types exclusifs:")
        print_ng_row("POS seul", ng_stats(pos_only), n)
        print_ng_row("Number seul", ng_stats(num_only), n)
        print_ng_row("Gender seul", ng_stats(gen_only), n)
        print_ng_row("Person seul", ng_stats(per_only), n)
        print_ng_row("Number + Gender", ng_stats(num_gen), n)
        print_ng_row("POS + Morpho", ng_stats(pos_morpho), n)

        # Pour homophones : detail des transitions POS avec n-gram
        if cat == "homophone":
            print(f"\n  Transitions POS (homophones) avec n-gram:")
            pb = lambda p: p.split(":")[0]
            by_trans = defaultdict(list)
            for s in sites:
                if s.pos_diff:
                    key = f"{pb(s.tag_err['pos'])}→{pb(s.tag_cor['pos'])}"
                    by_trans[key].append(s)
            for key in sorted(by_trans, key=lambda k: -len(by_trans[k])):
                sl = by_trans[key]
                st = ng_stats(sl)
                print(f"      {key:15s} n={st['n']:3d}"
                      f"  ng_cor:{st['ng_cor']:2d}({100*st['ng_cor']/st['n']:4.0f}%)"
                      f"  ng_err:{st['ng_err']:2d}({100*st['ng_err']/st['n']:4.0f}%)"
                      f"  delta:{st['ng_avg']:+.1f}")

        # Exemples avec n-gram
        print(f"\n  Exemples (premiers 40):")
        for i, s in enumerate(sites[:40]):
            pm_e = make_pm_tag(s.tag_err)
            pm_c = make_pm_tag(s.tag_cor)
            marks = []
            if s.pos_diff:
                marks.append("POS")
            if s.number_diff:
                marks.append("N")
            if s.gender_diff:
                marks.append("G")
            if s.person_diff:
                marks.append("P")
            if not marks:
                marks.append("invis")
            ng_mark = "+" if s.ng_delta > 0.01 else ("-" if s.ng_delta < -0.01 else "=")
            print(f"    {s.mot_err:15s}→{s.mot_cor:15s}"
                  f" {pm_e:>22s}→{pm_c:<22s}"
                  f" ng:{s.ng_delta:+6.1f}{ng_mark}"
                  f" conf:{s.tag_err['confiance']:.3f}→{s.tag_cor['confiance']:.3f}"
                  f" [{','.join(marks)}]")

    # ── Resume global ──
    print(f"\n{'='*70}")
    print(f"  RESUME")
    print(f"{'='*70}")
    print(f"\n  {'Categorie':<13s} {'sous-type':<18s} {'N':>5s}"
          f" {'ng>cor':>6s} {'ng>err':>6s} {'ng=':>5s}"
          f" {'ng_d':>6s} {'conf_d':>7s}")
    print(f"  {'-'*13} {'-'*18} {'-'*5} {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*7}")

    for cat in ["homophone", "accord", "conjugaison"]:
        sites = sites_by_cat.get(cat, [])
        if not sites:
            continue

        subtypes = [
            ("TOUS", sites),
            ("POS diff", [s for s in sites if s.pos_diff]),
            ("Number diff", [s for s in sites if s.number_diff]),
            ("Gender diff", [s for s in sites if s.gender_diff]),
            ("Person diff", [s for s in sites if s.person_diff]),
            ("Detecte", [s for s in sites if s.any_diff]),
            ("Invisible", [s for s in sites if not s.any_diff]),
        ]

        for label, sl in subtypes:
            if not sl:
                continue
            st = ng_stats(sl)
            print(f"  {cat:<13s} {label:<18s} {st['n']:5d}"
                  f" {100*st['ng_cor']/st['n']:5.1f}%"
                  f" {100*st['ng_err']/st['n']:5.1f}%"
                  f" {100*st['ng_eq']/st['n']:4.1f}%"
                  f" {st['ng_avg']:+6.2f}"
                  f" {st['conf_cor']-st['conf_err']:+7.4f}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
