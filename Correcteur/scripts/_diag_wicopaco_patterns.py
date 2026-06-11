#!/usr/bin/env python3
"""Diagnostic approfondi des FN WiCoPaCo — patterns exploitables.

Categorise les FN ou div_ortho=True ET P2G propose la bonne forme
par pattern structurel pour identifier les regles a implementer.

Usage:
    python scripts/_diag_wicopaco_patterns.py
    python scripts/_diag_wicopaco_patterns.py --verbose
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WORKSPACE = ROOT.parent.parent
CORPUS = WORKSPACE / "Corpus" / "Correcteur" / "grammaire_wicopaco.tsv"


def charger_corpus():
    cases = []
    with open(CORPUS, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and len(parts[1]) <= 500:
                cases.append((parts[0], parts[1], parts[2]))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--examples", "-e", type=int, default=5,
                        help="Nb exemples par pattern")
    args = parser.parse_args()

    corpus = charger_corpus()
    print(f"Corpus: {len(corpus)} phrases\n")

    from lectura_correcteur._lexique_lite import LexiqueLite
    from lectura_correcteur.correcteur_v6 import (
        CorrecteurV6, PUNCT_RE,
        _MODAUX, _MODAUX_ELARGI, _AVOIR_CONJUGUE, _AUXILIAIRES_ETRE,
        _AUXILIAIRES, _CLITIQUES_OBJETS, _MOTS_PROTEGES,
    )

    lex = LexiqueLite(str(ROOT / "src" / "lectura_correcteur" / "data" / "lexique_correcteur.db"))
    correcteur = CorrecteurV6(lex)

    # Preparer les prepositions
    PREPS = frozenset({
        "de", "d'", "d\u2019", "\u00e0", "a", "pour", "sans",
        "par", "en", "dans", "sur", "avec", "entre", "vers",
        "sous", "avant", "apr\u00e8s", "apres", "afin",
        "contre", "depuis", "pendant", "durant",
    })
    CONJS = frozenset({"et", "ou", "mais", "ni", "car", "donc", "or", "que", "qu'", "qu\u2019"})
    DET_PL = frozenset({"les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "leurs",
                        "quelques", "plusieurs", "certains", "certaines"})
    DET_SG = frozenset({"le", "la", "l'", "l\u2019", "un", "une", "du", "au",
                        "ce", "cet", "cette", "mon", "ma", "ton", "ta", "son", "sa",
                        "notre", "votre", "leur", "chaque", "quel", "quelle"})

    # Collecter les FN exploitables (div_ortho=True, P2G = bonne forme)
    # On separe par type d'erreur
    fn_exploitables = {"accord": [], "conjugaison": [], "homophone": []}

    for typ, erronee, corrigee_gold in corpus:
        result = correcteur.corriger(erronee)
        sortie = result.phrase_corrigee

        # Trouver l'edit gold
        src = erronee.lower().split()
        tgt = corrigee_gold.lower().split()
        out = sortie.lower().split()

        sm = SequenceMatcher(None, src, tgt)
        gold_edits = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    gold_edits.append((i1 + k, src[i1 + k], tgt[j1 + k]))

        sm2 = SequenceMatcher(None, src, out)
        sys_map = {}
        for tag, i1, i2, j1, j2 in sm2.get_opcodes():
            if tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    sys_map[i1 + k] = out[j1 + k]

        for pos, src_w, tgt_w in gold_edits:
            if pos >= 0 and sys_map.get(pos) == tgt_w:
                continue  # TP, pas FN

            # C'est un FN — analyser le MotV6
            tokens = correcteur._tokenize(erronee)
            if not tokens:
                continue
            tokens, _ = correcteur._pretraitement_fusion_elision(tokens)
            is_punct = [bool(PUNCT_RE.match(tk)) for tk in tokens]
            word_tokens = [tk for tk, p in zip(tokens, is_punct) if not p]
            if not word_tokens:
                continue
            formes = list(word_tokens)
            correcteur._v6_etape1_ortho(formes, followed_by_apo=set())
            mots_v6 = correcteur._v6_etape2_analyse(formes, word_tokens)

            # Trouver le MotV6
            mv = None
            mv_idx = -1
            for j, m in enumerate(mots_v6):
                if m.forme.lower() == src_w:
                    mv = m
                    mv_idx = j
                    break
            if mv is None:
                for j, m in enumerate(mots_v6):
                    if m.correction.lower() == src_w:
                        mv = m
                        mv_idx = j
                        break
            if mv is None or not mv.div_ortho:
                continue
            if mv.p2g_ortho.lower() != tgt_w:
                continue

            # C'est un FN exploitable!
            prev1 = mots_v6[mv_idx - 1].forme.lower() if mv_idx > 0 else ""
            prev2 = mots_v6[mv_idx - 2].forme.lower() if mv_idx > 1 else ""
            prev3 = mots_v6[mv_idx - 3].forme.lower() if mv_idx > 2 else ""
            next1 = mots_v6[mv_idx + 1].forme.lower() if mv_idx < len(mots_v6) - 1 else ""
            prev1_pos = mots_v6[mv_idx - 1].g2p_pos if mv_idx > 0 else ""
            next1_pos = mots_v6[mv_idx + 1].g2p_pos if mv_idx < len(mots_v6) - 1 else ""

            # Info lexique sur le mot
            g2p_pos = mv.g2p_pos or ""
            p2g_pos = mv.p2g_pos or ""
            pos_short = g2p_pos[:3] if g2p_pos else (p2g_pos[:3] if p2g_pos else "?")

            # Determiner le pattern structurel
            is_nom = pos_short == "NOM"
            is_ver = pos_short == "VER"
            is_adj = pos_short == "ADJ"

            # Infos morpho
            g2p_nombre = mv.g2p_nombre
            p2g_nombre = mv.p2g_nombre
            g2p_genre = mv.g2p_genre
            p2g_genre = mv.p2g_genre

            # Classifier le sous-type
            subtype = "autre"

            if typ == "accord":
                # Nombre ou genre?
                src_ends_s = src_w.endswith("s") or src_w.endswith("x")
                tgt_ends_s = tgt_w.endswith("s") or tgt_w.endswith("x")
                nombre_change = src_ends_s != tgt_ends_s

                # Suffixes genre
                genre_suffixes = [
                    ("e", ""), ("", "e"),
                    ("\u00e9e", "\u00e9"), ("\u00e9", "\u00e9e"),
                    ("\u00e9es", "\u00e9s"), ("\u00e9s", "\u00e9es"),
                    ("le", "l"), ("l", "le"),
                    ("ne", "n"), ("n", "ne"),
                    ("se", "s"), ("s", "se"),
                    ("ve", "f"), ("f", "ve"),
                    ("ive", "if"), ("if", "ive"),
                    ("ives", "ifs"), ("ifs", "ives"),
                ]
                genre_change = False
                for suf_a, suf_b in genre_suffixes:
                    if src_w.endswith(suf_a) and tgt_w.endswith(suf_b):
                        if len(src_w) - len(suf_a) == len(tgt_w) - len(suf_b):
                            genre_change = True
                            break
                    if src_w.endswith(suf_b) and tgt_w.endswith(suf_a):
                        if len(src_w) - len(suf_b) == len(tgt_w) - len(suf_a):
                            genre_change = True
                            break

                if is_nom:
                    if nombre_change and prev1 in DET_PL | DET_SG:
                        subtype = "NOM_apres_DET_nombre"
                    elif nombre_change and prev1 in PREPS:
                        subtype = "NOM_apres_PREP_nombre"
                    elif nombre_change and (prev1_pos or "").startswith("ADJ"):
                        subtype = "NOM_apres_ADJ_nombre"
                    elif nombre_change:
                        subtype = "NOM_autre_nombre"
                    elif genre_change:
                        subtype = "NOM_genre"
                    else:
                        subtype = "NOM_autre"
                elif is_ver:
                    if prev1 in _AVOIR_CONJUGUE or prev2 in _AVOIR_CONJUGUE or prev3 in _AVOIR_CONJUGUE:
                        if genre_change and not nombre_change:
                            subtype = "VER_PP_avoir_genre"
                        elif nombre_change:
                            subtype = "VER_PP_avoir_nombre"
                        else:
                            subtype = "VER_PP_avoir_autre"
                    elif prev1 in _AUXILIAIRES_ETRE or prev2 in _AUXILIAIRES_ETRE:
                        subtype = "VER_PP_etre"
                    elif "inf" in g2p_pos.lower() or "inf" in p2g_pos.lower():
                        subtype = "VER_INF_PP"
                    else:
                        subtype = "VER_accord_autre"
                elif is_adj:
                    if nombre_change and (prev1_pos or "").startswith("NOM"):
                        subtype = "ADJ_apres_NOM_nombre"
                    elif nombre_change and prev1 in DET_PL | DET_SG:
                        subtype = "ADJ_apres_DET_nombre"
                    elif genre_change and (prev1_pos or "").startswith("NOM"):
                        subtype = "ADJ_apres_NOM_genre"
                    elif nombre_change:
                        subtype = "ADJ_autre_nombre"
                    elif genre_change:
                        subtype = "ADJ_genre_autre"
                    else:
                        subtype = "ADJ_autre"
                else:
                    subtype = f"POS_{pos_short}"

            elif typ == "conjugaison":
                # INF <-> PP ?
                inf_suffixes = ("er", "ir", "re", "oir")
                pp_suffixes = ("\u00e9", "\u00e9e", "\u00e9s", "\u00e9es", "i", "is", "it", "u", "us", "ut")
                src_is_inf = src_w.endswith(inf_suffixes)
                tgt_is_inf = tgt_w.endswith(inf_suffixes)
                src_is_pp = src_w.endswith(pp_suffixes)
                tgt_is_pp = tgt_w.endswith(pp_suffixes)

                if src_is_inf and tgt_is_pp:
                    if prev1 in _AVOIR_CONJUGUE or prev2 in _AVOIR_CONJUGUE or prev3 in _AVOIR_CONJUGUE:
                        subtype = "INF_to_PP_avoir"
                    elif prev1 in _AUXILIAIRES_ETRE or prev2 in _AUXILIAIRES_ETRE:
                        subtype = "INF_to_PP_etre"
                    else:
                        subtype = "INF_to_PP_autre"
                elif src_is_pp and tgt_is_inf:
                    if prev1 in _MODAUX_ELARGI or prev2 in _MODAUX_ELARGI:
                        subtype = "PP_to_INF_modal"
                    elif prev1 in PREPS or prev2 in PREPS:
                        subtype = "PP_to_INF_prep"
                    else:
                        subtype = "PP_to_INF_autre"
                else:
                    # Temps verbal different
                    if src_w.endswith("ait") and not tgt_w.endswith("ait"):
                        subtype = "TEMPS_imparfait_to_autre"
                    elif tgt_w.endswith("ait") and not src_w.endswith("ait"):
                        subtype = "TEMPS_autre_to_imparfait"
                    elif src_w.endswith("ent") != tgt_w.endswith("ent"):
                        subtype = "CONJ_personne_nombre"
                    else:
                        subtype = "CONJ_autre"

            elif typ == "homophone":
                # Classifier par paire
                subtype = f"{src_w}/{tgt_w}"

            fn_exploitables[typ].append({
                "subtype": subtype,
                "src": src_w, "tgt": tgt_w,
                "conf": mv.p2g_confiance,
                "g2p_pos": g2p_pos, "p2g_pos": p2g_pos,
                "prev1": prev1, "prev2": prev2, "prev3": prev3,
                "next1": next1,
                "prev1_pos": prev1_pos, "next1_pos": next1_pos,
                "regle": mv.regle,
                "phrase": erronee[:140],
            })

    # ====================================================================
    # RAPPORT
    # ====================================================================
    for typ in ("accord", "conjugaison", "homophone"):
        fns = fn_exploitables[typ]
        if not fns:
            continue

        print(f"\n{'=' * 80}")
        print(f"  FN EXPLOITABLES — {typ.upper()} ({len(fns)} cas)")
        print(f"  (div_ortho=True ET P2G propose la bonne forme)")
        print(f"{'=' * 80}")

        # Par subtype
        sub_counter = Counter(d["subtype"] for d in fns)
        print(f"\n  {'Pattern':<35s} {'N':>5s} {'%':>6s}  {'Conf moy':>8s}")
        print(f"  {'-'*35} {'-'*5} {'-'*6}  {'-'*8}")
        for sub, n in sub_counter.most_common(40):
            sub_fns = [d for d in fns if d["subtype"] == sub]
            avg_conf = sum(d["conf"] for d in sub_fns) / len(sub_fns)
            pct = 100 * n / len(fns)
            print(f"  {sub:<35s} {n:5d} {pct:5.1f}%  {avg_conf:8.3f}")

        # Exemples par subtype (les plus gros)
        for sub, n in sub_counter.most_common(15):
            sub_fns = [d for d in fns if d["subtype"] == sub]
            print(f"\n  --- {sub} ({n} cas) ---")

            # Contexte: mot precedent
            prev_ctr = Counter(d["prev1"] for d in sub_fns if d["prev1"])
            if prev_ctr:
                top_prev = prev_ctr.most_common(8)
                print(f"  Mot precedent: {', '.join(f'{w}({c})' for w, c in top_prev)}")

            # Contexte: prev1_pos
            prev_pos_ctr = Counter(d["prev1_pos"][:3] for d in sub_fns if d["prev1_pos"])
            if prev_pos_ctr:
                top_pp = prev_pos_ctr.most_common(5)
                print(f"  POS precedent: {', '.join(f'{p}({c})' for p, c in top_pp)}")

            # Regles deja appliquees
            regles = Counter(d["regle"] for d in sub_fns if d["regle"])
            if regles:
                print(f"  Bloques par regle: {regles.most_common(5)}")

            # Confiance
            confs = sorted(d["conf"] for d in sub_fns)
            hi_conf = sum(1 for c in confs if c >= 0.90)
            print(f"  Confiance: min={confs[0]:.3f} med={confs[len(confs)//2]:.3f} "
                  f"max={confs[-1]:.3f}  >=0.90: {hi_conf}/{n}")

            # Exemples
            n_ex = min(args.examples, n)
            if args.verbose or n <= 8:
                n_ex = min(8, n)
            for d in sub_fns[:n_ex]:
                print(f"    {d['src']:15s} -> {d['tgt']:15s}  "
                      f"prev=[{d['prev2']} {d['prev1']}]  "
                      f"next=[{d['next1']}]  "
                      f"conf={d['conf']:.3f}  "
                      f"pos={d['g2p_pos'][:10]}")
                if args.verbose:
                    print(f"      {d['phrase']}")

    # ====================================================================
    # SYNTHESE : priorites d'implementation
    # ====================================================================
    print(f"\n{'=' * 80}")
    print(f"  SYNTHESE — PRIORITES D'IMPLEMENTATION")
    print(f"{'=' * 80}")

    all_fns = []
    for typ, fns in fn_exploitables.items():
        for d in fns:
            d["typ"] = typ
            all_fns.append(d)

    sub_counter = Counter(d["subtype"] for d in all_fns)
    print(f"\n  {'Pattern':<35s} {'N':>5s} {'Type':<12s} {'Conf>=0.9':>9s} {'Faisabilite':>11s}")
    print(f"  {'-'*35} {'-'*5} {'-'*12} {'-'*9} {'-'*11}")
    for sub, n in sub_counter.most_common(25):
        sub_fns = [d for d in all_fns if d["subtype"] == sub]
        typ = sub_fns[0]["typ"]
        hi = sum(1 for d in sub_fns if d["conf"] >= 0.90)
        # Heuristique faisabilite
        if hi >= n * 0.7 and n >= 10:
            fais = "*** FORT"
        elif hi >= n * 0.5 and n >= 5:
            fais = "** MOYEN"
        else:
            fais = "* FAIBLE"
        print(f"  {sub:<35s} {n:5d} {typ:<12s} {hi:4d}/{n:<4d}  {fais}")


if __name__ == "__main__":
    main()
