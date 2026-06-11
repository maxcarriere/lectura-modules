#!/usr/bin/env python3
"""Compare le score n-gram PM entre phrase erronee et corrigee (WiCoPaCo).

Pour chaque paire :
  1. Passe le G2P sur les deux phrases → obtient les tags POS+MORPHO
  2. Construit les sequences PM (POS|Number|Gender|Person)
  3. Score chaque sequence avec le n-gram PM
  4. Compare les scores

Cela permet de voir si le n-gram "prefere" la phrase corrigee.
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
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

WICOPACO_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/grammaire_wicopaco.tsv"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019][\w]+)*", re.UNICODE)


def tokeniser(phrase: str) -> list[str]:
    return [m.group() for m in _MOT_RE.finditer(phrase)]


def tronquer_mots(phrase: str, max_mots: int = 40) -> str:
    tokens = phrase.split()
    return " ".join(tokens[:max_mots]) if len(tokens) > max_mots else phrase


def charger_wicopaco(path, max_n=0):
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
    """Construit un tag PM : POS|Number|Gender|Person."""
    pos = tag["pos"]
    num = tag["number"] if tag["number"] != "_" else "_"
    gen = tag["gender"] if tag["gender"] != "_" else "_"
    per = tag["person"] if tag["person"] != "_" else "_"
    return f"{pos}|{num}|{gen}|{per}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=2000)
    parser.add_argument("--max-mots", type=int, default=40)
    args = parser.parse_args()

    paires = charger_wicopaco(WICOPACO_TSV, max_n=args.max)
    cat_counts = Counter(c for c, _, _ in paires)
    print(f"{len(paires)} paires")
    for c, n in cat_counts.most_common():
        print(f"  {c}: {n}")

    # Charger G2P
    print("\nChargement G2P...")
    from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
    tagger = creer_adapter_g2p_unifie()

    # Charger n-gram PM
    print("Chargement n-gram PM...")
    from lectura_correcteur._pos_ngram import PosNgram
    pos_ngram_db = os.path.join(
        _PROJECT_ROOT, "src", "lectura_correcteur", "data", "pos_ngram.db",
    )
    pos_ngram = PosNgram(pos_ngram_db)

    nombre_map = {"s": "Sing", "p": "Plur"}
    genre_map = {"m": "Masc", "f": "Fem"}

    def g2p_tags(mots):
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

    def score_pm_sequence(tags: list[dict]) -> float:
        """Score n-gram PM d'une sequence de tags."""
        pm_seq = [make_pm_tag(t) for t in tags]
        # Utiliser les trigrammes avec BOS padding
        BOS = "<BOS>"
        padded = [BOS, BOS] + pm_seq
        total_logp = 0.0
        for i in range(2, len(padded)):
            w1, w2, w3 = padded[i-2], padded[i-1], padded[i]
            logp = pos_ngram.logp_pm_trigram(w1, w2, w3)
            total_logp += logp
        return total_logp

    def score_g2p_sequence(tags: list[dict]) -> float:
        """Score G2P = somme log-confiance."""
        import math
        total = 0.0
        for t in tags:
            c = max(t["confiance"], 1e-6)
            total += math.log(c)
        return total

    # ── Analyse ──
    print("\nAnalyse en cours...")

    results_by_cat = defaultdict(lambda: {
        "total": 0,
        "ngram_prefer_cor": 0,
        "ngram_prefer_err": 0,
        "ngram_equal": 0,
        "g2p_prefer_cor": 0,
        "g2p_prefer_err": 0,
        "g2p_equal": 0,
        "ngram_deltas": [],
        "g2p_deltas": [],
        # Au site d'erreur seulement
        "site_ngram_deltas": [],
        "site_g2p_deltas": [],
    })

    exemples = []
    n_skip = 0

    t0 = time.time()
    for idx, (cat, phrase_err, phrase_cor) in enumerate(paires):
        if idx % 500 == 0 and idx > 0:
            print(f"  ... {idx}/{len(paires)}", file=sys.stderr)

        mots_err = tokeniser(tronquer_mots(phrase_err, args.max_mots))
        mots_cor = tokeniser(tronquer_mots(phrase_cor, args.max_mots))

        if len(mots_err) != len(mots_cor) or not mots_err:
            n_skip += 1
            continue

        tags_err = g2p_tags(mots_err)
        tags_cor = g2p_tags(mots_cor)

        # Scores sequences completes
        ngram_err = score_pm_sequence(tags_err)
        ngram_cor = score_pm_sequence(tags_cor)
        g2p_err = score_g2p_sequence(tags_err)
        g2p_cor = score_g2p_sequence(tags_cor)

        r = results_by_cat[cat]
        r["total"] += 1

        delta_ng = ngram_cor - ngram_err
        delta_g2p = g2p_cor - g2p_err
        r["ngram_deltas"].append(delta_ng)
        r["g2p_deltas"].append(delta_g2p)

        if delta_ng > 0.01:
            r["ngram_prefer_cor"] += 1
        elif delta_ng < -0.01:
            r["ngram_prefer_err"] += 1
        else:
            r["ngram_equal"] += 1

        if delta_g2p > 0.01:
            r["g2p_prefer_cor"] += 1
        elif delta_g2p < -0.01:
            r["g2p_prefer_err"] += 1
        else:
            r["g2p_equal"] += 1

        # Score local autour du site d'erreur (trigramme local)
        for i in range(len(mots_err)):
            if mots_err[i].lower() != mots_cor[i].lower():
                # Trigramme local centré sur le site d'erreur
                pm_err = [make_pm_tag(tags_err[j]) for j in range(len(tags_err))]
                pm_cor = [make_pm_tag(tags_cor[j]) for j in range(len(tags_cor))]
                BOS = "<BOS>"

                padded_err = [BOS, BOS] + pm_err
                padded_cor = [BOS, BOS] + pm_cor

                # Score local : trigrammes touchant le site i (positions i, i+1, i+2 dans padded)
                local_err = 0.0
                local_cor = 0.0
                for k in range(max(2, i), min(len(padded_err), i + 5)):
                    local_err += pos_ngram.logp_pm_trigram(padded_err[k-2], padded_err[k-1], padded_err[k])
                    local_cor += pos_ngram.logp_pm_trigram(padded_cor[k-2], padded_cor[k-1], padded_cor[k])

                r["site_ngram_deltas"].append(local_cor - local_err)
                r["site_g2p_deltas"].append(tags_cor[i]["confiance"] - tags_err[i]["confiance"])

        if len(exemples) < 60:
            # Trouver le site d'erreur
            for i in range(len(mots_err)):
                if mots_err[i].lower() != mots_cor[i].lower():
                    exemples.append((
                        cat, mots_err[i], mots_cor[i],
                        make_pm_tag(tags_err[i]), make_pm_tag(tags_cor[i]),
                        tags_err[i]["confiance"], tags_cor[i]["confiance"],
                        ngram_err, ngram_cor, delta_ng, delta_g2p,
                    ))
                    break

    elapsed = time.time() - t0
    print(f"\nTermine ({elapsed:.1f}s), skip={n_skip}")

    # ── Rapport ──
    for cat in ["homophone", "accord", "conjugaison"]:
        r = results_by_cat.get(cat)
        if not r or not r["total"]:
            continue

        n = r["total"]
        print(f"\n{'='*70}")
        print(f"  {cat.upper()} — {n} paires")
        print(f"{'='*70}")

        print(f"\n  Score SEQUENCE COMPLETE (phrase entiere):")
        print(f"  {'':35s} {'N-gram PM':>12s} {'G2P (sum log)':>14s}")
        print(f"  {'Prefere phrase corrigee':35s} {r['ngram_prefer_cor']:5d} ({100*r['ngram_prefer_cor']/n:.1f}%) "
              f"{r['g2p_prefer_cor']:5d} ({100*r['g2p_prefer_cor']/n:.1f}%)")
        print(f"  {'Prefere phrase erronee':35s} {r['ngram_prefer_err']:5d} ({100*r['ngram_prefer_err']/n:.1f}%) "
              f"{r['g2p_prefer_err']:5d} ({100*r['g2p_prefer_err']/n:.1f}%)")
        print(f"  {'Egal (delta < 0.01)':35s} {r['ngram_equal']:5d} ({100*r['ngram_equal']/n:.1f}%) "
              f"{r['g2p_equal']:5d} ({100*r['g2p_equal']/n:.1f}%)")

        avg_ng = sum(r["ngram_deltas"]) / n
        avg_g2p = sum(r["g2p_deltas"]) / n
        print(f"\n  Delta moyen (cor - err):")
        print(f"    N-gram PM  : {avg_ng:+.4f} logp")
        print(f"    G2P confiance: {avg_g2p:+.4f} log")

        if r["site_ngram_deltas"]:
            ns = len(r["site_ngram_deltas"])
            avg_site_ng = sum(r["site_ngram_deltas"]) / ns
            avg_site_g2p = sum(r["site_g2p_deltas"]) / ns
            print(f"\n  Score LOCAL (trigrammes autour du site d'erreur):")
            print(f"    N-gram PM delta moyen  : {avg_site_ng:+.4f} logp ({ns} sites)")
            print(f"    G2P confiance delta    : {avg_site_g2p:+.4f} ({ns} sites)")

            # Distribution du delta n-gram local
            pos_ng = sum(1 for d in r["site_ngram_deltas"] if d > 0.01)
            neg_ng = sum(1 for d in r["site_ngram_deltas"] if d < -0.01)
            eq_ng = ns - pos_ng - neg_ng
            print(f"    N-gram local prefere cor : {pos_ng}/{ns} ({100*pos_ng/ns:.1f}%)")
            print(f"    N-gram local prefere err : {neg_ng}/{ns} ({100*neg_ng/ns:.1f}%)")
            print(f"    N-gram local egal        : {eq_ng}/{ns} ({100*eq_ng/ns:.1f}%)")

    # Exemples
    print(f"\n{'='*70}")
    print(f"  EXEMPLES (premiers 40)")
    print(f"{'='*70}")
    for i, (cat, me, mc, pm_e, pm_c, ce, cc, ng_e, ng_c, d_ng, d_g2p) in enumerate(exemples[:40]):
        pm_mark = "*" if pm_e != pm_c else " "
        ng_mark = "+" if d_ng > 0.01 else ("-" if d_ng < -0.01 else "=")
        print(f"  [{i+1:3d}] {cat:12s} {me:15s}→{mc:15s}"
              f" PM:{pm_e:>20s}→{pm_c:<20s} {pm_mark}"
              f" ng:{d_ng:+.2f}{ng_mark} conf:{ce:.3f}→{cc:.3f}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
