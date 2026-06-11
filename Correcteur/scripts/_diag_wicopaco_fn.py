#!/usr/bin/env python3
"""Diagnostic des faux negatifs WiCoPaCo — analyse exhaustive.

Pour chaque FN, affiche les infos de diagnostic du correcteur V6 :
- div_ortho, div_pos, p2g_ortho, p2g_confiance
- g2p_pos, p2g_pos
- regle appliquee (si autre regle a pris le mot)
- contexte (mot precedent, mot suivant)

Usage:
    python scripts/_diag_wicopaco_fn.py
    python scripts/_diag_wicopaco_fn.py --max 500
    python scripts/_diag_wicopaco_fn.py --categories conjugaison
    python scripts/_diag_wicopaco_fn.py --categories accord --top 20
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WORKSPACE = ROOT.parent.parent
CORPUS = WORKSPACE / "Corpus" / "Correcteur" / "grammaire_wicopaco.tsv"


def charger_corpus(max_n: int = 0, categories: set[str] | None = None):
    """Charge le corpus WiCoPaCo grammaire."""
    cases = []
    with open(CORPUS, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            if len(parts[1]) > 500:
                continue
            t = parts[0]
            if categories and t not in categories:
                continue
            cases.append({
                "type": t,
                "erronee": parts[1],
                "corrigee": parts[2],
            })
    if max_n and len(cases) > max_n:
        import random
        rng = random.Random(42)
        cases = rng.sample(cases, max_n)
    return cases


def trouver_edits(src_mots, tgt_mots):
    """Retourne les edits (position_src, mot_src, mot_tgt)."""
    sm = SequenceMatcher(None, src_mots, tgt_mots)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                edits.append((i1 + k, src_mots[i1 + k], tgt_mots[j1 + k]))
        elif tag == "delete":
            for k in range(i2 - i1):
                edits.append((i1 + k, src_mots[i1 + k], ""))
        elif tag == "insert":
            for k in range(j2 - j1):
                edits.append((-1, "", tgt_mots[j1 + k]))
    return edits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0,
                        help="Max phrases (0=toutes)")
    parser.add_argument("--categories", type=str, default="",
                        help="Filtrer (comma-sep: accord,conjugaison,homophone)")
    parser.add_argument("--top", type=int, default=30,
                        help="Nombre de patterns a afficher par categorie")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    cats = set(args.categories.split(",")) if args.categories else None
    corpus = charger_corpus(args.max, cats)
    print(f"Corpus charge: {len(corpus)} phrases")

    # Distribution par type
    type_counts = Counter(c["type"] for c in corpus)
    for t, n in sorted(type_counts.items()):
        print(f"  {t}: {n}")

    # Charger le correcteur
    from lectura_correcteur._lexique_lite import LexiqueLite
    from lectura_correcteur.correcteur_v6 import CorrecteurV6, MotV6, PUNCT_RE

    lex = LexiqueLite(str(ROOT / "src" / "lectura_correcteur" / "data" / "lexique_correcteur.db"))
    correcteur = CorrecteurV6(lex)

    # Resultats
    fn_details: list[dict] = []
    tp_count = Counter()  # type -> count
    fn_count = Counter()
    total_count = Counter()

    for cas in corpus:
        t = cas["type"]
        erronee = cas["erronee"]
        corrigee_gold = cas["corrigee"]

        # Corriger
        result = correcteur.corriger(erronee)
        sortie = result.phrase_corrigee

        # Edits gold
        src_mots = erronee.lower().split()
        tgt_mots = corrigee_gold.lower().split()
        out_mots = sortie.lower().split()

        gold_edits = trouver_edits(src_mots, tgt_mots)
        sys_edits_map = {}
        for pos, s, o in trouver_edits(src_mots, out_mots):
            if pos >= 0:
                sys_edits_map[pos] = o

        # Pour chaque edit gold, verifier si le systeme l'a reproduit
        for pos, src_w, tgt_w in gold_edits:
            total_count[t] += 1
            if pos >= 0 and sys_edits_map.get(pos) == tgt_w:
                tp_count[t] += 1
                continue

            fn_count[t] += 1

            # Diagnostic : acceder aux mots V6 internes
            # On refait l'analyse pour obtenir les MotV6
            tokens = correcteur._tokenize(erronee)
            if not tokens:
                fn_details.append({
                    "type": t, "src": src_w, "tgt": tgt_w,
                    "phrase": erronee[:120],
                    "diag": "no_tokens",
                })
                continue

            tokens, _ = correcteur._pretraitement_fusion_elision(tokens)
            is_punct = [bool(PUNCT_RE.match(tk)) for tk in tokens]
            word_tokens = [tk for tk, p in zip(tokens, is_punct) if not p]
            word_indices = [i for i, p in enumerate(is_punct) if not p]
            if not word_tokens:
                continue

            formes = list(word_tokens)
            correcteur._v6_etape1_ortho(formes, followed_by_apo=set())
            mots_v6 = correcteur._v6_etape2_analyse(formes, word_tokens)

            # Trouver le mot correspondant
            # pos est l'index dans src_mots (split simple)
            # On cherche le mot dans word_tokens (tokenisation correcteur)
            mv_match = None
            mv_idx = -1
            src_w_lower = src_w.lower()
            # Strategie : chercher par forme
            for j, mv in enumerate(mots_v6):
                if mv.forme.lower() == src_w_lower:
                    mv_match = mv
                    mv_idx = j
                    break
            # Fallback : chercher la correction
            if mv_match is None:
                for j, mv in enumerate(mots_v6):
                    if mv.correction.lower() == src_w_lower:
                        mv_match = mv
                        mv_idx = j
                        break

            # Contexte
            prev_forme = mots_v6[mv_idx - 1].forme if mv_match and mv_idx > 0 else ""
            next_forme = mots_v6[mv_idx + 1].forme if mv_match and mv_idx < len(mots_v6) - 1 else ""

            detail = {
                "type": t,
                "src": src_w,
                "tgt": tgt_w,
                "obtenu": sys_edits_map.get(pos, src_w) if pos >= 0 else "?",
                "phrase": erronee[:150],
                "prev": prev_forme,
                "next": next_forme,
            }

            if mv_match:
                detail.update({
                    "div_ortho": mv_match.div_ortho,
                    "div_pos": mv_match.div_pos,
                    "p2g_ortho": mv_match.p2g_ortho,
                    "p2g_conf": round(mv_match.p2g_confiance, 3),
                    "g2p_pos": mv_match.g2p_pos,
                    "p2g_pos": mv_match.p2g_pos,
                    "regle": mv_match.regle,
                    "correction": mv_match.correction,
                    "forme": mv_match.forme,
                })
            else:
                detail["diag"] = "mot_non_trouve"

            fn_details.append(detail)

    # ====================================================================
    # Rapport
    # ====================================================================
    print("\n" + "=" * 80)
    print("  RECALL PAR TYPE")
    print("=" * 80)
    for t in sorted(total_count):
        tp = tp_count[t]
        tot = total_count[t]
        fn = fn_count[t]
        pct = 100 * tp / tot if tot else 0
        print(f"  {t:15s}  {tp:4d}/{tot:4d}  = {pct:5.1f}%   (FN={fn})")

    # Analyse des FN par categorie
    for t in sorted(total_count):
        fns = [d for d in fn_details if d["type"] == t]
        if not fns:
            continue

        print(f"\n{'=' * 80}")
        print(f"  FN — {t.upper()} ({len(fns)} cas)")
        print(f"{'=' * 80}")

        # Sous-categories de diagnostic
        div_ortho_true = [d for d in fns if d.get("div_ortho")]
        div_ortho_false = [d for d in fns if not d.get("div_ortho") and "div_ortho" in d]
        has_regle = [d for d in fns if d.get("regle")]
        mot_non_trouve = [d for d in fns if d.get("diag") == "mot_non_trouve"]

        print(f"\n  Signal P2G:")
        print(f"    div_ortho=True  : {len(div_ortho_true):4d}  ({100*len(div_ortho_true)/len(fns):.0f}%)")
        print(f"    div_ortho=False : {len(div_ortho_false):4d}  ({100*len(div_ortho_false)/len(fns):.0f}%)")
        print(f"    mot non trouve  : {len(mot_non_trouve):4d}")
        print(f"    autre regle     : {len(has_regle):4d}")

        # P2G propose la bonne correction ?
        p2g_correct = [d for d in div_ortho_true
                       if d.get("p2g_ortho", "").lower() == d["tgt"].lower()]
        p2g_autre = [d for d in div_ortho_true
                     if d.get("p2g_ortho", "").lower() != d["tgt"].lower()
                     and d.get("p2g_ortho")]
        print(f"\n  Parmi div_ortho=True ({len(div_ortho_true)}):")
        print(f"    P2G propose la bonne forme : {len(p2g_correct):4d}")
        print(f"    P2G propose autre chose     : {len(p2g_autre):4d}")
        print(f"    P2G vide                    : {len(div_ortho_true) - len(p2g_correct) - len(p2g_autre):4d}")

        # Distribution de confiance P2G pour div_ortho=True
        confs = [d["p2g_conf"] for d in div_ortho_true if "p2g_conf" in d]
        if confs:
            confs.sort()
            print(f"\n  Confiance P2G (div_ortho=True):")
            print(f"    min={confs[0]:.3f}  median={confs[len(confs)//2]:.3f}  max={confs[-1]:.3f}")
            # Buckets
            for lo, hi in [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
                n = sum(1 for c in confs if lo <= c < hi)
                print(f"    [{lo:.1f}, {hi:.1f}): {n:4d}")

        # Distribution des POS G2P
        pos_counter = Counter()
        for d in div_ortho_true:
            pos = d.get("g2p_pos", "?")
            pos_counter[pos[:3] if pos else "?"] += 1
        print(f"\n  POS G2P (div_ortho=True):")
        for pos, n in pos_counter.most_common(args.top):
            print(f"    {pos:10s}: {n:4d}")

        # Regles deja appliquees (blocage)
        if has_regle:
            regle_counter = Counter(d.get("regle", "?") for d in has_regle)
            print(f"\n  Regles deja appliquees (FN bloques par autre regle):")
            for r, n in regle_counter.most_common(10):
                print(f"    {r:35s}: {n:4d}")

        # Edits src->tgt les plus frequents
        edit_counter = Counter()
        for d in fns:
            edit_counter[(d["src"], d["tgt"])] += 1
        print(f"\n  Edits FN les plus frequents:")
        for (s, tg), n in edit_counter.most_common(args.top):
            print(f"    {s:15s} -> {tg:15s} : {n:3d}")

        # Contexte precedent le plus frequent (pour div_ortho=True)
        if div_ortho_true:
            prev_counter = Counter(d.get("prev", "").lower() for d in div_ortho_true if d.get("prev"))
            print(f"\n  Mot precedent (div_ortho=True, top {min(15, len(prev_counter))}):")
            for w, n in prev_counter.most_common(15):
                print(f"    {w:15s}: {n:4d}")

        # Exemples detailles
        if args.verbose:
            print(f"\n  Exemples detailles (top 20):")
            for d in fns[:20]:
                print(f"    [{d['src']}]->[{d['tgt']}]  "
                      f"div={d.get('div_ortho','?')}  "
                      f"p2g={d.get('p2g_ortho','?')}  "
                      f"conf={d.get('p2g_conf','?')}  "
                      f"pos={d.get('g2p_pos','?')}/{d.get('p2g_pos','?')}  "
                      f"regle={d.get('regle','')}")
                print(f"      prev={d.get('prev','')}  next={d.get('next','')}")
                print(f"      {d['phrase'][:120]}")


if __name__ == "__main__":
    main()
