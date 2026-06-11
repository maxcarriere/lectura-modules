#!/usr/bin/env python3
"""Investigation des faux positifs (FP) : mots corrects modifies a tort.

Dump les 1270 FP sur mots corrects pour comprendre les patterns dominants.
"""
from __future__ import annotations

import csv
import os
import re
import sys
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
    return [m.group() for m in _MOT_RE.finditer(phrase)]


def charger_wicopaco(path, max_n=2000):
    paires = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3 or row[0].startswith("#") or row[0] == "type_erreur":
                continue
            cat, err, cor = row[0].strip(), row[1].strip(), row[2].strip()
            if err and cor:
                paires.append((cat, err, cor))
                if max_n and len(paires) >= max_n:
                    break
    return paires


def tronquer(phrase, max_mots=30):
    tokens = phrase.split()
    return " ".join(tokens[:max_mots]) if len(tokens) > max_mots else phrase


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur._utils import LexiqueNormalise
    from lectura_correcteur._pos_ngram import PosNgram
    from lectura_correcteur._analyse_grammaticale import analyser_phrase

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

    print("Chargement...", file=sys.stderr)
    lexique = Lexique(LEXIQUE_DB)
    lex = LexiqueNormalise(lexique)
    pos_ngram_db = os.path.join(
        _PROJECT_ROOT, "src", "lectura_correcteur", "data", "pos_ngram.db",
    )
    pos_ngram = PosNgram(pos_ngram_db)

    paires = charger_wicopaco(WICOPACO_TSV, max_n=2000)
    print(f"{len(paires)} paires chargees", file=sys.stderr)

    # Compteurs FP
    fp_list = []  # (mot_original, correction, pos_analyse, pm_tag, confiance, contexte, cat)
    fp_by_pos_change = Counter()       # "NOM->ADJ" etc
    fp_by_nombre_change = Counter()    # "Sing->Plur" etc
    fp_by_genre_change = Counter()     # "Masc->Fem" etc
    fp_by_correction_type = Counter()  # "morpho" vs "homophone"
    fp_by_cat = Counter()              # categorie WiCoPaCo
    fp_by_forme = Counter()            # forme originale
    fp_by_confiance = Counter()        # tranche de confiance
    fp_ancre_count = 0
    fp_total = 0

    for idx, (cat, phrase_err, phrase_cor) in enumerate(paires):
        if idx % 500 == 0 and idx > 0:
            print(f"  ... {idx}/{len(paires)}", file=sys.stderr)

        phrase_err = tronquer(phrase_err, 30)
        phrase_cor = tronquer(phrase_cor, 30)
        mots_err = tokeniser(phrase_err)
        mots_cor = tokeniser(phrase_cor)

        if len(mots_err) != len(mots_cor) or not mots_err:
            continue

        analyse_err = analyser_phrase(
            mots_err, lex, pos_ngram,
            lm_homophones=lm_homophones,
            expand_morpho=True,
            expand_homophones=True,
            penalite_morpho=-3.0,
            penalite_homophone=-5.0,
        )

        for i, (mot_e, mot_c) in enumerate(zip(mots_err, mots_cor)):
            if i >= len(analyse_err):
                break
            a = analyse_err[i]

            is_correct_word = mot_e.lower() == mot_c.lower()
            has_correction = bool(a.forme_corrigee)

            if is_correct_word and has_correction:
                fp_total += 1

                # Determiner le type de changement
                corr_lower = a.forme_corrigee.lower()
                orig_lower = mot_e.lower()

                # Contexte : 2 mots avant + 2 mots apres
                ctx_before = " ".join(mots_err[max(0, i-2):i])
                ctx_after = " ".join(mots_err[i+1:min(len(mots_err), i+3)])
                contexte = f"...{ctx_before} [{orig_lower}->{corr_lower}] {ctx_after}..."

                # POS de la correction vs POS original
                # Chercher le POS de la forme corrigee dans les candidats
                corr_pm = a.pm_tag
                orig_pos = ""
                corr_pos = a.pos
                # Chercher le PM tag original (sans correction)
                for cand_pm in a.candidats_pm:
                    parts = cand_pm.split("|")
                    # Le candidat dont la forme est l'originale
                    # On ne peut pas le savoir directement, mais on peut regarder le pm_tag final

                fp_by_cat[cat] += 1
                fp_by_forme[orig_lower] += 1

                # Tranche confiance
                conf_tranche = f"{int(a.confiance * 10) / 10:.1f}"
                fp_by_confiance[conf_tranche] += 1

                if a.ancre:
                    fp_ancre_count += 1

                fp_list.append((
                    orig_lower, corr_lower, a.pos, a.pm_tag,
                    a.confiance, contexte, cat,
                    len(a.candidats_pm),
                ))

    # --- Rapport ---
    print(f"\n{'='*80}")
    print(f"  INVESTIGATION FP : {fp_total} mots corrects modifies a tort")
    print(f"{'='*80}")

    print(f"\n  FP sur ancres : {fp_ancre_count}")

    print(f"\n--- FP par categorie WiCoPaCo ---")
    for cat, n in fp_by_cat.most_common():
        print(f"  {cat:30s} : {n:5d}")

    print(f"\n--- FP par forme originale (top 40) ---")
    for forme, n in fp_by_forme.most_common(40):
        print(f"  {forme:20s} : {n:5d}")

    print(f"\n--- FP par tranche de confiance ---")
    for tranche, n in sorted(fp_by_confiance.items()):
        print(f"  conf={tranche} : {n:5d}")

    # Analyser les types de changements
    morpho_changes = Counter()  # "Sing->Plur", "Masc->Fem", etc
    pos_changes = Counter()
    same_lemme = 0
    diff_lemme = 0

    for orig, corr, pos, pm_tag, conf, ctx, cat, n_cands in fp_list:
        # Essayer de determiner la nature du changement
        # Si meme longueur +/- 1-2 chars et meme debut → probablement morpho
        if len(orig) > 2 and len(corr) > 2 and orig[:3] == corr[:3]:
            same_lemme += 1
        else:
            diff_lemme += 1

    print(f"\n--- Nature des corrections FP ---")
    print(f"  Meme lemme (morpho) : {same_lemme}")
    print(f"  Lemme different     : {diff_lemme}")

    # Top 50 corrections FP (forme → correction)
    fp_transitions = Counter()
    for orig, corr, pos, pm_tag, conf, ctx, cat, n_cands in fp_list:
        fp_transitions[(orig, corr)] += 1

    print(f"\n--- Top 50 transitions FP (forme -> correction) ---")
    for (orig, corr), n in fp_transitions.most_common(50):
        print(f"  {orig:20s} -> {corr:20s} : {n:4d}")

    # Exemples detailles (premiers 30)
    print(f"\n--- 30 premiers exemples FP ---")
    for i, (orig, corr, pos, pm_tag, conf, ctx, cat, n_cands) in enumerate(fp_list[:30]):
        print(f"  [{i+1:3d}] {orig} -> {corr}  (pos={pos}, pm={pm_tag}, conf={conf:.2f}, cands={n_cands})")
        print(f"        cat={cat}  {ctx}")


if __name__ == "__main__":
    main()
