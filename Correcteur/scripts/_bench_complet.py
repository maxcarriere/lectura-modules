#!/usr/bin/env python3
"""Benchmark COMPLET du correcteur.

Evalue sur TOUTES les donnees disponibles (pas d'echantillonnage) et produit
une analyse detaillee des erreurs pour identifier les axes d'amelioration.

Configs testees: Baseline (LexiqueTagger) + Hybride (TaggerHybride)
"""

import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
WICOPACO_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/grammaire_wicopaco.tsv"
NEGATIF_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/negatif_wicopaco.tsv"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


def normaliser(texte):
    return " ".join(texte.strip().lower().split())


def extraire_mots(texte):
    return [m.group().lower() for m in _MOT_RE.finditer(texte)]


def tronquer_contexte(erronee, attendue, fenetre=12):
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
    return (" ".join(tokens_err[start:end]), " ".join(tokens_att[start:end]))


def charger_tsv(path):
    categories = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            cat = row[0].strip()
            categories.setdefault(cat, []).append((row[1].strip(), row[2].strip()))
    return categories


def charger_negatif(path):
    phrases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                phrases.append(parts[-1] if len(parts) > 1 else parts[0])
    return phrases


def evaluer_wicopaco_mot(correcteur, paires, max_n=999999):
    """Evaluation mode mot. Retourne details par paire mot_err→mot_att."""
    ok = ko = fn = fp = 0
    errors = []
    details = []  # (status, mot_err, mot_att, contexte_court)

    for erronee, attendue in paires[:max_n]:
        tronque = tronquer_contexte(erronee, attendue)
        if tronque is None:
            ko += 1
            details.append(("SKIP", "", "", ""))
            continue
        err_ctx, att_ctx = tronque

        mots_err = extraire_mots(err_ctx)
        mots_att = extraire_mots(att_ctx)
        if len(mots_err) != len(mots_att):
            ko += 1
            details.append(("SKIP", "", "", ""))
            continue
        idx = None
        mot_err = mot_att = ""
        for i, (a, b) in enumerate(zip(mots_err, mots_att)):
            if a != b:
                idx = i
                mot_err, mot_att = a, b
                break
        if idx is None:
            ko += 1
            details.append(("SKIP", "", "", ""))
            continue

        res = correcteur.corriger(err_ctx)
        mots_obtenu = extraire_mots(res.phrase_corrigee)

        mot_trouve = None
        for offset in (0, -1, 1, -2, 2, -3, 3):
            j = idx + offset
            if 0 <= j < len(mots_obtenu):
                if mots_obtenu[j] in (mot_att, mot_err):
                    mot_trouve = mots_obtenu[j]
                    break
        if mot_trouve is None:
            for m in mots_obtenu:
                if m in (mot_att, mot_err):
                    mot_trouve = m
                    break
        if mot_trouve is None:
            for m in mots_obtenu:
                for cand in (mot_att, mot_err):
                    if cand in m and len(cand) >= 2:
                        mot_trouve = cand
                        break
                if mot_trouve is not None:
                    break

        correction_attendue = (mot_err != mot_att)
        if mot_trouve == mot_att:
            ok += 1
            details.append(("OK", mot_err, mot_att, ""))
        elif mot_trouve == mot_err and correction_attendue:
            fn += 1
            details.append(("FN", mot_err, mot_att, err_ctx[:100]))
            errors.append(("FN", f"{mot_err}\u2192{mot_att}", err_ctx[:80]))
        elif not correction_attendue:
            fp += 1
            details.append(("FP", mot_err, mot_att, err_ctx[:100]))
            errors.append(("FP", f"{mot_err}", err_ctx[:80]))
        else:
            ko += 1
            details.append(("WRONG", mot_err, mot_att, err_ctx[:100]))
            errors.append(("WRONG", f"{mot_err}\u2192{mot_att} (got {mot_trouve})", err_ctx[:80]))

    total = ok + ko + fn + fp
    return ok, total, fn, fp, ko, errors, details


def evaluer_negatif(correcteur, phrases, max_n=999999):
    """Mesure FP sur phrases correctes. Retourne aussi les types de FP."""
    ok = fp = 0
    errors = []
    fp_types = []  # (mot_original, mot_modifie)

    for phrase in phrases[:max_n]:
        res = correcteur.corriger(phrase)
        if normaliser(res.phrase_corrigee) == normaliser(phrase):
            ok += 1
        else:
            fp += 1
            errors.append(("FP", phrase[:80], res.phrase_corrigee[:80]))
            # Trouver les mots modifies
            m_orig = extraire_mots(phrase)
            m_corr = extraire_mots(res.phrase_corrigee)
            for i in range(min(len(m_orig), len(m_corr))):
                if m_orig[i] != m_corr[i]:
                    fp_types.append((m_orig[i], m_corr[i]))

    total = ok + fp
    return ok, total, fp, errors, fp_types


def analyser_erreurs_par_paire(details, cat_name):
    """Analyse les erreurs par paire mot_err→mot_att."""
    paire_stats = defaultdict(lambda: {"ok": 0, "fn": 0, "wrong": 0, "total": 0})
    for status, mot_err, mot_att, ctx in details:
        if status == "SKIP":
            continue
        key = f"{mot_err}\u2192{mot_att}"
        paire_stats[key]["total"] += 1
        if status == "OK":
            paire_stats[key]["ok"] += 1
        elif status == "FN":
            paire_stats[key]["fn"] += 1
        elif status == "WRONG":
            paire_stats[key]["wrong"] += 1

    return paire_stats


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig

    print("=" * 80)
    print("  BENCHMARK COMPLET DU CORRECTEUR")
    print("=" * 80)
    print()

    print("Chargement du lexique...")
    lexique = Lexique(LEXIQUE_DB)

    cats_wicopaco = charger_tsv(WICOPACO_TSV) if os.path.exists(WICOPACO_TSV) else {}
    phrases_neg = charger_negatif(NEGATIF_TSV) if os.path.exists(NEGATIF_TSV) else []

    print(f"Donnees chargees:")
    for cat, pairs in sorted(cats_wicopaco.items()):
        print(f"  WiCoPaCo [{cat}]: {len(pairs)} paires")
    print(f"  Negatif: {len(phrases_neg)} phrases")
    print()

    # On teste: Baseline (LexiqueTagger) et Hybride (TaggerHybride)
    configs = {
        "Baseline": CorrecteurConfig(),
        "Hybride": CorrecteurConfig(activer_tagger_hybride=True),
    }

    all_results = {}
    all_details = {}
    all_fp_types = {}

    for config_name, cfg in configs.items():
        print(f"\n{'#' * 70}")
        print(f"# {config_name}")
        print(f"{'#' * 70}")

        corr = Correcteur(lexique, config=cfg)
        t0 = time.time()

        results = {}
        cat_details = {}
        n_eval = 0

        # WiCoPaCo par categorie - TOUTES les donnees
        ok_all = fn_all = fp_all = ko_all = total_all = 0
        for cat in sorted(cats_wicopaco.keys()):
            pairs = cats_wicopaco[cat]
            t_cat = time.time()
            ok_c, total_c, fn_c, fp_c, ko_c, errors_c, details_c = evaluer_wicopaco_mot(
                corr, pairs, max_n=999999,
            )
            elapsed_cat = time.time() - t_cat
            pct = 100 * ok_c / total_c if total_c > 0 else 0
            print(f"\n  [{cat}] {ok_c}/{total_c} ({pct:.1f}%)  FN={fn_c}  WRONG={ko_c}  [{elapsed_cat:.0f}s]")

            results[f"wico_{cat}"] = (ok_c, total_c, fn_c, fp_c, ko_c)
            cat_details[cat] = details_c
            n_eval += total_c
            ok_all += ok_c; fn_all += fn_c; fp_all += fp_c
            ko_all += ko_c; total_all += total_c

        pct_all = 100 * ok_all / total_all if total_all > 0 else 0
        print(f"\n  --- WiCoPaCo TOTAL: {ok_all}/{total_all} ({pct_all:.1f}%)  FN={fn_all}  WRONG={ko_all} ---")
        results["wico_TOTAL"] = (ok_all, total_all, fn_all, fp_all, ko_all)

        # Negatif - TOUTES les donnees
        t_neg = time.time()
        ok_n, total_n, fp_n, errors_n, fp_types = evaluer_negatif(corr, phrases_neg, max_n=999999)
        elapsed_neg = time.time() - t_neg
        pct_neg = 100 * ok_n / total_n if total_n > 0 else 0
        print(f"\n  [Negatif] {ok_n}/{total_n} ({pct_neg:.1f}%)  FP={fp_n}  [{elapsed_neg:.0f}s]")
        results["negatif"] = (ok_n, total_n, 0, fp_n, 0)

        elapsed = time.time() - t0
        n_eval += total_n
        print(f"\n  Total: {n_eval} evaluations en {elapsed:.0f}s ({elapsed/n_eval:.2f}s/eval)")

        all_results[config_name] = results
        all_details[config_name] = cat_details
        all_fp_types[config_name] = fp_types

    # ================================================================
    # TABLEAU COMPARATIF
    # ================================================================
    print(f"\n\n{'=' * 80}")
    print("  TABLEAU COMPARATIF")
    print(f"{'=' * 80}\n")

    ordered_keys = sorted(set(k for r in all_results.values() for k in r.keys()))
    labels = list(all_results.keys())

    hdr = f"{'Metrique':25s}"
    for lbl in labels:
        hdr += f" | {lbl:>30s}"
    print(hdr)
    print("-" * len(hdr))

    for key in ordered_keys:
        row = f"{key:25s}"
        for lbl in labels:
            if key in all_results[lbl]:
                ok, total, fn, fp, ko = all_results[lbl][key]
                pct = 100 * ok / total if total > 0 else 0
                cell = f"{ok}/{total} ({pct:.1f}%) FN={fn} W={ko}"
            else:
                cell = "---"
            row += f" | {cell:>30s}"
        print(row)

    # ================================================================
    # ANALYSE DETAILLEE DES ERREURS (sur Hybride)
    # ================================================================
    best_config = "Hybride"
    print(f"\n\n{'=' * 80}")
    print(f"  ANALYSE DETAILLEE DES ERREURS ({best_config})")
    print(f"{'=' * 80}")

    for cat in sorted(all_details.get(best_config, {}).keys()):
        details = all_details[best_config][cat]
        paire_stats = analyser_erreurs_par_paire(details, cat)

        # Trier par nombre de FN (plus gros gisements d'amelioration)
        sorted_paires = sorted(paire_stats.items(), key=lambda x: -x[1]["fn"])

        total_fn = sum(v["fn"] for v in paire_stats.values())
        total_wrong = sum(v["wrong"] for v in paire_stats.values())
        total_ok = sum(v["ok"] for v in paire_stats.values())
        total_all = sum(v["total"] for v in paire_stats.values())

        print(f"\n{'─' * 70}")
        print(f"  {cat.upper()} : {total_ok}/{total_all} OK,  {total_fn} FN,  {total_wrong} WRONG")
        print(f"  {len(paire_stats)} paires distinctes")
        print(f"{'─' * 70}")

        # Top 20 FN (gisements)
        fn_paires = [(k, v) for k, v in sorted_paires if v["fn"] > 0]
        if fn_paires:
            print(f"\n  Top {min(30, len(fn_paires))} paires les plus manquees (FN) :")
            for i, (paire, stats) in enumerate(fn_paires[:30]):
                pct = 100 * stats["ok"] / stats["total"] if stats["total"] > 0 else 0
                print(f"    {i+1:3d}. {paire:30s}  FN={stats['fn']:3d}  OK={stats['ok']:3d}/{stats['total']}  ({pct:.0f}%)")

        # Top WRONG
        wrong_paires = [(k, v) for k, v in sorted_paires if v["wrong"] > 0]
        if wrong_paires:
            print(f"\n  Paires avec WRONG ({len(wrong_paires)} paires) :")
            for i, (paire, stats) in enumerate(wrong_paires[:20]):
                print(f"    {i+1:3d}. {paire:30s}  WRONG={stats['wrong']:3d}  OK={stats['ok']}/{stats['total']}")

    # ================================================================
    # ANALYSE DES FAUX POSITIFS (sur Hybride)
    # ================================================================
    print(f"\n\n{'=' * 80}")
    print(f"  ANALYSE DES FAUX POSITIFS ({best_config})")
    print(f"{'=' * 80}")

    fp_types = all_fp_types.get(best_config, [])
    if fp_types:
        # Compter par mot original → mot corrige
        fp_counter = Counter()
        for orig, corr in fp_types:
            fp_counter[f"{orig}\u2192{corr}"] += 1

        print(f"\n  {len(fp_types)} modifications fausses sur {all_results[best_config].get('negatif', (0,0))[1]} phrases")
        print(f"  Top 30 faux positifs les plus frequents :\n")
        for i, (paire, count) in enumerate(fp_counter.most_common(30)):
            print(f"    {i+1:3d}. {paire:35s}  x{count}")

        # Regrouper par type de modification
        print(f"\n  Par pattern de modification :")
        pattern_counter = Counter()
        for orig, corr in fp_types:
            if orig.lower() == "a" and corr.lower() == "\u00e0":
                pattern_counter["a\u2192\u00e0 (accent)"] += 1
            elif orig.lower() == "\u00e0" and corr.lower() == "a":
                pattern_counter["\u00e0\u2192a (desaccentuation)"] += 1
            elif len(orig) > 0 and len(corr) > 0 and orig[0].isupper() != corr[0].isupper():
                pattern_counter["changement de casse"] += 1
            elif orig.endswith("s") and not corr.endswith("s"):
                pattern_counter["suppression pluriel"] += 1
            elif not orig.endswith("s") and corr.endswith("s"):
                pattern_counter["ajout pluriel"] += 1
            elif orig.endswith("e") and not corr.endswith("e"):
                pattern_counter["suppression -e"] += 1
            elif not orig.endswith("e") and corr.endswith("e"):
                pattern_counter["ajout -e"] += 1
            elif orig.endswith("\u00e9") and corr.endswith("er") or orig.endswith("er") and corr.endswith("\u00e9"):
                pattern_counter["\u00e9/er confusion"] += 1
            else:
                pattern_counter["autre"] += 1

        for pattern, count in pattern_counter.most_common():
            print(f"    {pattern:35s}  x{count}")

    # ================================================================
    # RESUME ET RECOMMANDATIONS
    # ================================================================
    print(f"\n\n{'=' * 80}")
    print(f"  RESUME")
    print(f"{'=' * 80}")

    for config_name in labels:
        r = all_results[config_name]
        wico = r.get("wico_TOTAL", (0, 0, 0, 0, 0))
        neg = r.get("negatif", (0, 0, 0, 0, 0))
        print(f"\n  {config_name}:")
        print(f"    WiCoPaCo total : {wico[0]}/{wico[1]} ({100*wico[0]/wico[1]:.1f}%)")
        for key in sorted(r.keys()):
            if key.startswith("wico_") and key != "wico_TOTAL":
                ok, total, fn, fp, ko = r[key]
                cat_short = key.replace("wico_", "")
                print(f"      {cat_short:15s}: {ok}/{total} ({100*ok/total:.1f}%)  FN={fn}  WRONG={ko}")
        print(f"    Negatif (FP)   : {neg[3]} FP sur {neg[1]} phrases ({100*neg[3]/neg[1]:.1f}% FP rate)")

    print(f"\n{'=' * 80}")
    print("  BENCHMARK TERMINE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
