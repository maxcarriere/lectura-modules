#!/usr/bin/env python3
"""Benchmark — Precision POS+MORPHO du module d'analyse grammaticale.

Methode :
- Sur les phrases WiCoPaCo (phrase corrigee = gold standard)
  1. Analyser la phrase corrigee → gold POS+MORPHO (lexique = 1 seule hypothese)
  2. Analyser la phrase erronee → predicted POS+MORPHO
  3. Comparer position par position

Metriques :
- POS accuracy : % de mots avec bon POS
- MORPHO accuracy : % de mots avec bon nombre+genre
- Anchor coverage : % de mots fixes comme ancres
- Conflict detection : % de conflits reels detectes
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

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
WICOPACO_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/grammaire_wicopaco.tsv"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019][\w]+)*", re.UNICODE)


def tokeniser(phrase: str) -> list[str]:
    """Tokenise en mots (pas de ponctuation)."""
    return [m.group() for m in _MOT_RE.finditer(phrase)]


def charger_wicopaco(path: str, max_n: int = 0) -> list[tuple[str, str, str]]:
    """Charge (type_erreur, phrase_erronee, phrase_corrigee) depuis le TSV."""
    paires = []
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


def aligner_mots(mots_err: list[str], mots_cor: list[str]) -> list[tuple[str, str]] | None:
    """Aligne les mots si meme nombre de tokens. Retourne None sinon."""
    if len(mots_err) != len(mots_cor):
        return None
    return list(zip(mots_err, mots_cor))


def tronquer(phrase: str, max_mots: int = 30) -> str:
    """Tronque les phrases longues pour analyse plus rapide."""
    tokens = phrase.split()
    if len(tokens) > max_mots:
        return " ".join(tokens[:max_mots])
    return phrase


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark analyse grammaticale")
    parser.add_argument("--max-phrases", type=int, default=2000,
                        help="Max phrases a analyser (0=toutes)")
    parser.add_argument("--max-mots", type=int, default=30,
                        help="Max mots par phrase (tronquer les longues)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Afficher les erreurs individuelles")
    parser.add_argument("--cat", type=str, default="",
                        help="Filtrer par categorie (accord, conjugaison, homophone)")
    parser.add_argument("--g2p", action="store_true",
                        help="Utiliser G2P Unifie V2 comme prior d'emission")
    parser.add_argument("--w-g2p", type=float, default=1.0,
                        help="Poids du prior G2P (defaut: 1.0)")
    parser.add_argument("--expand", action="store_true",
                        help="Activer les candidats elargis (morpho + homophones)")
    parser.add_argument("--no-morpho", action="store_true",
                        help="Desactiver les variantes morphologiques (avec --expand)")
    parser.add_argument("--no-homophones", action="store_true",
                        help="Desactiver les variantes homophones (avec --expand)")
    parser.add_argument("--penalite-morpho", type=float, default=-3.0,
                        help="Penalite emission variantes morpho (defaut: -3.0)")
    parser.add_argument("--penalite-homophone", type=float, default=-5.0,
                        help="Penalite emission homophones (defaut: -5.0)")
    args = parser.parse_args()

    from lectura_lexique import Lexique
    from lectura_correcteur._utils import LexiqueNormalise
    from lectura_correcteur._pos_ngram import PosNgram
    from lectura_correcteur._analyse_grammaticale import (
        analyser_phrase,
        formater_analyse,
        score_pm_sequence,
        AnalyseMot,
    )

    # Optionnel : LM homophones
    lm_homophones = None
    try:
        from lectura_correcteur._lm_homophones import LMHomophones
        lm_db = os.path.join(
            _PROJECT_ROOT, "src", "lectura_correcteur", "data", "homophones_trigrams.db",
        )
        if os.path.exists(lm_db):
            lm_homophones = LMHomophones(lm_db)
    except Exception:
        pass

    print("Chargement lexique...")
    lexique = Lexique(LEXIQUE_DB)
    lex = LexiqueNormalise(lexique)

    # Optionnel : G2P Unifie V2
    g2p_tagger = None
    if args.g2p:
        try:
            from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
            g2p_tagger = creer_adapter_g2p_unifie()
            if g2p_tagger:
                print(f"G2P Unifie V2 charge (w_g2p={args.w_g2p})")
            else:
                print("ATTENTION: G2P Unifie V2 indisponible (--g2p ignore)")
        except Exception as e:
            print(f"ATTENTION: G2P Unifie V2 erreur: {e}")

    print("Chargement POS n-gram...")
    pos_ngram_db = os.path.join(
        _PROJECT_ROOT, "src", "lectura_correcteur", "data", "pos_ngram.db",
    )
    if not os.path.exists(pos_ngram_db):
        print(f"ERREUR: pos_ngram.db introuvable: {pos_ngram_db}")
        sys.exit(1)
    pos_ngram = PosNgram(pos_ngram_db)

    print("Chargement corpus WiCoPaCo...")
    paires = charger_wicopaco(WICOPACO_TSV, max_n=args.max_phrases)
    if args.cat:
        paires = [(c, e, co) for c, e, co in paires if args.cat in c]
    print(f"  {len(paires)} paires chargees")

    # Flags expand
    do_expand_morpho = args.expand and not args.no_morpho
    do_expand_homophones = args.expand and not args.no_homophones

    # --- Metriques ---
    total_mots = 0
    pos_correct = 0
    nombre_correct = 0
    genre_correct = 0
    morpho_correct = 0  # nombre ET genre
    ancres_total = 0
    ancres_count = 0

    # Conflits
    vrais_conflits = 0  # conflit detecte sur un mot reellement errone
    faux_conflits = 0   # conflit detecte sur un mot correct
    conflits_manques = 0  # mot errone sans conflit detecte

    # Correction (forme_corrigee)
    corr_vp = 0   # forme corrigee et correction = gold
    corr_fp = 0   # forme corrigee mais correction fausse
    corr_fn = 0   # mot errone, non corrige
    corr_vn = 0   # mot correct, non corrige (bon)
    corr_fp_correct = 0  # mot correct mais modifie (fausse correction)

    # Par categorie
    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0, "pos_ok": 0, "nombre_ok": 0, "genre_ok": 0,
        "conflits_vp": 0, "conflits_fp": 0, "conflits_fn": 0,
        "corr_vp": 0, "corr_fp": 0, "corr_fn": 0, "corr_fp_correct": 0,
    })

    # Confusion POS
    pos_confusion: Counter = Counter()

    phrases_analysees = 0
    phrases_non_alignees = 0

    t0 = time.time()

    for idx, (cat, phrase_err, phrase_cor) in enumerate(paires):
        if idx % 500 == 0 and idx > 0:
            elapsed = time.time() - t0
            print(f"  ... {idx}/{len(paires)} ({elapsed:.1f}s)")

        # Tronquer
        phrase_err = tronquer(phrase_err, args.max_mots)
        phrase_cor = tronquer(phrase_cor, args.max_mots)

        mots_err = tokeniser(phrase_err)
        mots_cor = tokeniser(phrase_cor)

        aligned = aligner_mots(mots_err, mots_cor)
        if aligned is None:
            phrases_non_alignees += 1
            continue

        if not mots_err:
            continue

        # Analyse de la phrase erronee
        analyse_err = analyser_phrase(
            mots_err, lex, pos_ngram,
            lm_homophones=lm_homophones,
            tagger=g2p_tagger,
            w_g2p=args.w_g2p,
            expand_morpho=do_expand_morpho,
            expand_homophones=do_expand_homophones,
            penalite_morpho=args.penalite_morpho,
            penalite_homophone=args.penalite_homophone,
        )

        # Gold : analyse de la phrase corrigee (pas de expand pour le gold)
        analyse_gold = analyser_phrase(
            mots_cor, lex, pos_ngram,
            lm_homophones=lm_homophones,
            tagger=g2p_tagger,
            w_g2p=args.w_g2p,
        )

        phrases_analysees += 1

        for i, (mot_e, mot_c) in enumerate(aligned):
            if i >= len(analyse_err) or i >= len(analyse_gold):
                break

            a_err = analyse_err[i]
            a_gold = analyse_gold[i]
            is_error_word = (mot_e.lower() != mot_c.lower())

            total_mots += 1

            # POS
            pos_e = a_err.pos.split(":")[0]
            pos_g = a_gold.pos.split(":")[0]
            if pos_e == pos_g:
                pos_correct += 1
            else:
                pos_confusion[(pos_g, pos_e)] += 1

            # Nombre
            if a_err.nombre == a_gold.nombre or a_gold.nombre == "_":
                nombre_correct += 1

            # Genre
            if a_err.genre == a_gold.genre or a_gold.genre == "_":
                genre_correct += 1

            # Morpho (nombre ET genre)
            if (
                (a_err.nombre == a_gold.nombre or a_gold.nombre == "_")
                and (a_err.genre == a_gold.genre or a_gold.genre == "_")
            ):
                morpho_correct += 1

            # Ancres
            ancres_total += 1
            if a_err.ancre:
                ancres_count += 1

            # Conflits
            has_conflict = bool(a_err.conflits)
            if is_error_word:
                if has_conflict:
                    vrais_conflits += 1
                    cat_stats[cat]["conflits_vp"] += 1
                else:
                    conflits_manques += 1
                    cat_stats[cat]["conflits_fn"] += 1
            else:
                if has_conflict:
                    faux_conflits += 1
                    cat_stats[cat]["conflits_fp"] += 1

            # Corrections (forme_corrigee)
            has_correction = bool(a_err.forme_corrigee)
            if is_error_word:
                if has_correction:
                    # Correction proposee : est-ce la bonne forme ?
                    if a_err.forme_corrigee.lower() == mot_c.lower():
                        corr_vp += 1
                        cat_stats[cat]["corr_vp"] += 1
                    else:
                        corr_fp += 1
                        cat_stats[cat]["corr_fp"] += 1
                else:
                    corr_fn += 1
                    cat_stats[cat]["corr_fn"] += 1
            else:
                if has_correction:
                    corr_fp_correct += 1
                    cat_stats[cat]["corr_fp_correct"] += 1
                else:
                    corr_vn += 1

            # Stats par cat
            cat_stats[cat]["total"] += 1
            if pos_e == pos_g:
                cat_stats[cat]["pos_ok"] += 1
            if a_err.nombre == a_gold.nombre or a_gold.nombre == "_":
                cat_stats[cat]["nombre_ok"] += 1
            if a_err.genre == a_gold.genre or a_gold.genre == "_":
                cat_stats[cat]["genre_ok"] += 1

            # Verbose : afficher les erreurs
            if args.verbose and is_error_word:
                if has_correction:
                    correct_corr = a_err.forme_corrigee.lower() == mot_c.lower()
                    label = "VP" if correct_corr else "FP_corr"
                    print(f"  {label}: [{cat}] '{mot_e}' -> corr='{a_err.forme_corrigee}' gold='{mot_c}'")
                elif not has_conflict:
                    print(f"  FN: [{cat}] '{mot_e}' -> '{mot_c}'")
                    print(f"      analyse: {a_err.pos} {a_err.nombre} {a_err.genre} conf={a_err.confiance:.2f}")
                    print(f"      gold:    {a_gold.pos} {a_gold.nombre} {a_gold.genre}")
                    if a_err.candidats_pm:
                        print(f"      cands:   {a_err.candidats_pm}")

    elapsed = time.time() - t0

    # --- Rapport ---
    mode_label = "baseline"
    if args.expand:
        parts = []
        if do_expand_morpho:
            parts.append(f"morpho(p={args.penalite_morpho})")
        if do_expand_homophones:
            parts.append(f"homo(p={args.penalite_homophone})")
        mode_label = " + ".join(parts) if parts else "expand (aucun)"
    if args.g2p:
        mode_label += " + G2P"

    print(f"\n{'='*80}")
    print(f"  BENCHMARK ANALYSE GRAMMATICALE — Pass 2")
    print(f"  Mode : {mode_label}")
    print(f"{'='*80}")
    print(f"  Phrases analysees     : {phrases_analysees}")
    print(f"  Phrases non alignees  : {phrases_non_alignees} (ignorees)")
    print(f"  Mots compares         : {total_mots}")
    print(f"  Temps                 : {elapsed:.2f}s ({total_mots/max(elapsed,0.01):.0f} mots/s)")

    print(f"\n  --- Accuracy ---")
    pct = lambda n, d: 100 * n / max(d, 1)
    print(f"  POS accuracy          : {pos_correct}/{total_mots} ({pct(pos_correct, total_mots):.1f}%)")
    print(f"  Nombre accuracy       : {nombre_correct}/{total_mots} ({pct(nombre_correct, total_mots):.1f}%)")
    print(f"  Genre accuracy        : {genre_correct}/{total_mots} ({pct(genre_correct, total_mots):.1f}%)")
    print(f"  Morpho accuracy (N+G) : {morpho_correct}/{total_mots} ({pct(morpho_correct, total_mots):.1f}%)")

    print(f"\n  --- Ancres ---")
    print(f"  Anchor coverage       : {ancres_count}/{ancres_total} ({pct(ancres_count, ancres_total):.1f}%)")

    print(f"\n  --- Detection de conflits ---")
    total_err_mots = vrais_conflits + conflits_manques
    print(f"  Vrais positifs (VP)   : {vrais_conflits}/{total_err_mots} ({pct(vrais_conflits, total_err_mots):.1f}%)")
    print(f"  Faux positifs (FP)    : {faux_conflits}")
    print(f"  Faux negatifs (FN)    : {conflits_manques}")
    precision_c = vrais_conflits / max(vrais_conflits + faux_conflits, 1)
    recall_c = vrais_conflits / max(vrais_conflits + conflits_manques, 1)
    f1_c = 2 * precision_c * recall_c / max(precision_c + recall_c, 1e-9)
    print(f"  Precision             : {precision_c:.3f}")
    print(f"  Recall                : {recall_c:.3f}")
    print(f"  F1                    : {f1_c:.3f}")

    # Correction metrics
    print(f"\n  --- Correction (forme_corrigee) ---")
    total_modifies = corr_vp + corr_fp + corr_fp_correct
    total_err_mots_corr = corr_vp + corr_fp + corr_fn
    print(f"  Corrections correctes (VP)    : {corr_vp}")
    print(f"  Corrections fausses (FP err)  : {corr_fp} (mot errone, mauvaise correction)")
    print(f"  Fausses corrections (FP ok)   : {corr_fp_correct} (mot correct, modifie a tort)")
    print(f"  Erreurs non corrigees (FN)    : {corr_fn}")
    print(f"  Mots corrects non modifies    : {corr_vn}")
    precision_corr = corr_vp / max(total_modifies, 1)
    recall_corr = corr_vp / max(total_err_mots_corr, 1)
    f1_corr = 2 * precision_corr * recall_corr / max(precision_corr + recall_corr, 1e-9)
    print(f"  Precision correction  : {precision_corr:.3f} ({corr_vp}/{total_modifies})")
    print(f"  Recall correction     : {recall_corr:.3f} ({corr_vp}/{total_err_mots_corr})")
    print(f"  F1 correction         : {f1_corr:.3f}")

    # Par categorie
    print(f"\n  --- Par categorie ---")
    header = (
        f"  {'Cat':<20s} {'Mots':>6s} {'POS%':>6s} {'Nbr%':>6s} {'Gen%':>6s}"
        f" {'VP':>5s} {'FP':>5s} {'FN':>5s}"
        f" {'cVP':>5s} {'cFP':>5s} {'cFN':>5s} {'cFPok':>5s}"
    )
    print(header)
    sep = f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}"
    print(sep)
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        n = s["total"]
        print(
            f"  {cat:<20s} {n:>6d}"
            f" {pct(s['pos_ok'], n):>5.1f}%"
            f" {pct(s['nombre_ok'], n):>5.1f}%"
            f" {pct(s['genre_ok'], n):>5.1f}%"
            f" {s['conflits_vp']:>5d}"
            f" {s['conflits_fp']:>5d}"
            f" {s['conflits_fn']:>5d}"
            f" {s['corr_vp']:>5d}"
            f" {s['corr_fp']:>5d}"
            f" {s['corr_fn']:>5d}"
            f" {s['corr_fp_correct']:>5d}"
        )

    # Top confusions POS
    print(f"\n  --- Top 20 confusions POS (gold -> predict) ---")
    for (gold, pred), cnt in pos_confusion.most_common(20):
        print(f"    {gold:>10s} -> {pred:<10s} : {cnt}")

    print()
    pos_ngram.close()


if __name__ == "__main__":
    main()
