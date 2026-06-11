#!/usr/bin/env python3
"""Benchmark G2P sur phrases erronees WiCoPaCo.

Compare l'etiquetage G2P entre la phrase erronee et la phrase corrigee.
Objectif : comprendre si le G2P detecte la faute (POS/Morpho differents)
ou s'il l'ignore (meme etiquetage).

Pour chaque paire (erronee, corrigee) :
  - tokenise les deux versions
  - passe le G2P sur les deux
  - compare les tags aux positions qui different (le site de l'erreur)
  - reporte les metriques

Usage :
    python3 _bench_g2p_wicopaco.py                   # 2000 premieres paires
    python3 _bench_g2p_wicopaco.py --max 500          # 500 paires
    python3 _bench_g2p_wicopaco.py --categories homophone
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


def charger_wicopaco(path: str, max_n: int = 0, categories: set | None = None):
    """Charge les paires (cat, erronee, corrigee) depuis le TSV."""
    paires = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3 or row[0].startswith("#") or row[0] == "type_erreur":
                continue
            cat, err, cor = row[0].strip(), row[1].strip(), row[2].strip()
            if not err or not cor:
                continue
            if categories and cat not in categories:
                continue
            paires.append((cat, err, cor))
            if max_n and len(paires) >= max_n:
                break
    return paires


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark G2P sur WiCoPaCo")
    parser.add_argument("--max", type=int, default=2000,
                        help="Nombre max de paires (0=toutes)")
    parser.add_argument("--categories", type=str, default="",
                        help="Filtrer par categories (virgule-separe)")
    parser.add_argument("--max-mots", type=int, default=40,
                        help="Troncature phrase (mots)")
    args = parser.parse_args()

    cats = set(args.categories.split(",")) if args.categories else None

    print("Chargement WiCoPaCo...")
    paires = charger_wicopaco(WICOPACO_TSV, max_n=args.max, categories=cats)
    print(f"  {len(paires)} paires chargees")

    # Stats categories
    cat_counts = Counter(cat for cat, _, _ in paires)
    for cat, n in cat_counts.most_common():
        print(f"    {cat}: {n}")

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
        result = []
        for g in tags:
            result.append({
                "pos": g.get("pos", "NOM"),
                "number": nombre_map.get(g.get("nombre", ""), "_"),
                "gender": genre_map.get(g.get("genre", ""), "_"),
                "person": g.get("personne", "_") or "_",
                "confiance": g.get("confiance_pos", 0.0),
            })
        return result

    # ── Analyse ──────────────────────────────────────────────────────────

    # Compteurs globaux
    n_paires_traitees = 0
    n_paires_skip_len = 0  # longueur differente
    n_sites_erreur = 0     # positions ou err != cor
    n_sites_pos_diff = 0   # le G2P donne un POS different err vs cor
    n_sites_morpho_diff = 0  # morpho different

    # Par categorie
    by_cat = defaultdict(lambda: {
        "total": 0,
        "sites_erreur": 0,
        "pos_diff": 0,
        "morpho_diff": 0,
    })

    # Confiance au site d'erreur vs ailleurs
    conf_erreur = []   # confiance G2P au site d'erreur (phrase erronee)
    conf_correct = []  # confiance G2P au meme site (phrase corrigee)
    conf_ailleurs = [] # confiance G2P sur mots identiques

    # Types de changements POS detectes par le G2P
    pos_changes_at_error = Counter()  # (pos_err, pos_cor) au site d'erreur

    # Exemples detailles
    exemples = []  # (cat, mot_err, mot_cor, tag_err, tag_cor)

    t0 = time.time()

    for idx, (cat, phrase_err, phrase_cor) in enumerate(paires):
        if idx % 500 == 0 and idx > 0:
            print(f"  ... {idx}/{len(paires)}", file=sys.stderr)

        phrase_err = tronquer_mots(phrase_err, args.max_mots)
        phrase_cor = tronquer_mots(phrase_cor, args.max_mots)

        mots_err = tokeniser(phrase_err)
        mots_cor = tokeniser(phrase_cor)

        if len(mots_err) != len(mots_cor) or not mots_err:
            n_paires_skip_len += 1
            continue

        n_paires_traitees += 1
        tags_err = tag_phrase(mots_err)
        tags_cor = tag_phrase(mots_cor)

        for i, (me, mc) in enumerate(zip(mots_err, mots_cor)):
            if i >= len(tags_err) or i >= len(tags_cor):
                break

            te = tags_err[i]
            tc = tags_cor[i]

            if me.lower() != mc.lower():
                # Site d'erreur
                n_sites_erreur += 1
                by_cat[cat]["sites_erreur"] += 1

                conf_erreur.append(te["confiance"])
                conf_correct.append(tc["confiance"])

                pos_diff = te["pos"] != tc["pos"]
                morpho_diff = (
                    te["number"] != tc["number"]
                    or te["gender"] != tc["gender"]
                    or te["person"] != tc["person"]
                )

                if pos_diff:
                    n_sites_pos_diff += 1
                    by_cat[cat]["pos_diff"] += 1
                    pos_changes_at_error[(te["pos"], tc["pos"])] += 1

                if pos_diff or morpho_diff:
                    n_sites_morpho_diff += 1
                    by_cat[cat]["morpho_diff"] += 1

                if len(exemples) < 80:
                    exemples.append((cat, me, mc, te, tc))
            else:
                # Mot identique
                conf_ailleurs.append(te["confiance"])

        by_cat[cat]["total"] += 1

    elapsed = time.time() - t0

    # ── Rapport ──────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  BENCHMARK G2P sur WiCoPaCo ({elapsed:.1f}s)")
    print(f"{'='*70}")

    print(f"\n  Paires traitees   : {n_paires_traitees}")
    print(f"  Paires skippees   : {n_paires_skip_len} (longueur differente)")
    print(f"  Sites d'erreur    : {n_sites_erreur}")

    if n_sites_erreur:
        print(f"\n  --- Detection au site d'erreur ---")
        print(f"  POS different     : {n_sites_pos_diff}/{n_sites_erreur} "
              f"({100*n_sites_pos_diff/n_sites_erreur:.1f}%)")
        print(f"  POS ou Morpho diff: {n_sites_morpho_diff}/{n_sites_erreur} "
              f"({100*n_sites_morpho_diff/n_sites_erreur:.1f}%)")

    if conf_erreur:
        avg_err = sum(conf_erreur) / len(conf_erreur)
        avg_cor = sum(conf_correct) / len(conf_correct)
        avg_ail = sum(conf_ailleurs) / len(conf_ailleurs) if conf_ailleurs else 0
        print(f"\n  --- Confiance G2P ---")
        print(f"  Au site d'erreur (phrase err)  : {avg_err:.4f}")
        print(f"  Au site d'erreur (phrase cor)  : {avg_cor:.4f}")
        print(f"  Sur mots identiques            : {avg_ail:.4f}")
        print(f"  Delta (cor - err)              : {avg_cor - avg_err:+.4f}")

    if by_cat:
        print(f"\n  --- Par categorie ---")
        print(f"  {'Categorie':<20s} {'Paires':>7s} {'Sites':>7s} {'POS diff':>10s} {'Morpho diff':>12s}")
        print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*10} {'-'*12}")
        for cat in sorted(by_cat.keys()):
            d = by_cat[cat]
            pos_pct = f"{100*d['pos_diff']/d['sites_erreur']:.1f}%" if d["sites_erreur"] else "-"
            mor_pct = f"{100*d['morpho_diff']/d['sites_erreur']:.1f}%" if d["sites_erreur"] else "-"
            print(f"  {cat:<20s} {d['total']:7d} {d['sites_erreur']:7d} "
                  f"{pos_pct:>10s} {mor_pct:>12s}")

    if pos_changes_at_error:
        print(f"\n  --- Top 20 changements POS au site d'erreur ---")
        print(f"  (POS phrase_err → POS phrase_cor)")
        for (pe, pc), n in pos_changes_at_error.most_common(20):
            print(f"    {pe:12s} → {pc:12s} : {n}")

    if exemples:
        print(f"\n  --- Exemples detailles (premiers {min(40, len(exemples))}) ---")
        for i, (cat, me, mc, te, tc) in enumerate(exemples[:40]):
            pos_mark = " *POS*" if te["pos"] != tc["pos"] else ""
            morpho_mark = " *MOR*" if (te["number"] != tc["number"]
                                       or te["gender"] != tc["gender"]
                                       or te["person"] != tc["person"]) else ""
            print(f"  [{i+1:3d}] {cat:12s} {me:15s} → {mc:15s} | "
                  f"POS: {te['pos']:8s}→{tc['pos']:8s} "
                  f"N:{te['number']:5s}→{tc['number']:5s} "
                  f"G:{te['gender']:5s}→{tc['gender']:5s} "
                  f"conf: {te['confiance']:.3f}→{tc['confiance']:.3f}"
                  f"{pos_mark}{morpho_mark}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
