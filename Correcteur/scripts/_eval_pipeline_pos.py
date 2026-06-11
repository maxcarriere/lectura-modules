#!/usr/bin/env python3
"""Evaluation du pipeline POS/MORPHO COMPLET sur phrases erronees.

Passe par Correcteur.corriger() pour mesurer le POS final que les
regles de grammaire voient reellement, avec toutes les couches :
- Tagger initial (LexiqueTagger ou Hybride)
- Fallback POS lexique
- Viterbi POS n-gram (si active)
- Re-tagging LexiqueTagger avant grammaire

On compare le pipeline baseline vs hybride sur 1000 paires WiCoPaCo
accord en mesurant :
1. POS du mot cible errone (le tagger comprend-il la categorie malgre la faute ?)
2. NOMBRE du mot cible (le pipeline detecte-t-il le bon nombre ?)
3. POS des voisins (stabilite)
4. Qualite NOMBRE sur les DET/PRO (ancres pour les regles d'accord)
"""

import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
WICOPACO_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/grammaire_wicopaco.tsv"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)

# Ancres POS incontestables
_ANCRES_POS = {
    "le": "ART", "la": "ART", "les": "ART", "un": "ART", "une": "ART",
    "des": "ART", "du": "ART", "au": "PRE", "aux": "ART",
    "ce": "PRO", "cette": "ADJ", "ces": "ADJ",
    "mon": "ADJ", "ma": "ADJ", "ton": "ADJ", "ta": "ADJ",
    "son": "ADJ", "sa": "ADJ", "mes": "ADJ", "tes": "ADJ",
    "ses": "ADJ", "nos": "ADJ", "vos": "ADJ", "leurs": "ADJ",
    "de": "PRE", "dans": "PRE", "par": "PRE", "pour": "PRE",
    "avec": "PRE", "sur": "PRE", "sous": "PRE", "entre": "PRE",
    "et": "CON", "ou": "CON", "mais": "CON",
    "je": "PRO", "tu": "PRO", "il": "PRO", "elle": "PRO",
    "nous": "PRO", "vous": "PRO", "ils": "PRO", "elles": "PRO",
    "est": "AUX", "sont": "AUX", "a": "AUX", "ont": "AUX",
}

_ANCRES_NOMBRE = {
    "le": "s", "la": "s", "un": "s", "une": "s", "ce": "s", "cette": "s",
    "les": "p", "des": "p", "ces": "p", "mes": "p", "ses": "p",
    "nos": "p", "vos": "p", "leurs": "p",
    "il": "s", "elle": "s", "ils": "p", "elles": "p",
}


def tronquer(phrase, max_mots=30):
    tokens = phrase.split()
    return " ".join(tokens[:max_mots]) if len(tokens) > max_mots else phrase


def evaluer_pipeline(correcteur, paires, lex_norm, label):
    """Evalue le pipeline complet via correcteur.corriger()."""

    # Metriques POS
    ancre_pos_ok = 0
    ancre_pos_total = 0
    ancre_pos_errors = Counter()

    # Metriques NOMBRE
    ancre_nombre_ok = 0
    ancre_nombre_total = 0
    ancre_nombre_errors = Counter()

    # POS du mot cible (errone)
    target_pos_ok = 0
    target_pos_total = 0
    target_pos_errors = Counter()

    # NOMBRE du mot cible
    target_nombre_ok_err = 0   # nombre correct malgre la faute
    target_nombre_ok_cor = 0   # nombre correct sur la phrase corrigee
    target_nombre_total = 0

    # POS correct mots de contenu (NOM, VER, ADJ)
    contenu_pos_ok = 0
    contenu_pos_total = 0
    contenu_pos_errors = Counter()

    # Stabilite POS errone vs corrige
    stable_pos = 0
    unstable_pos = 0

    skipped = 0

    for phrase_err, phrase_cor in paires:
        phrase_err = tronquer(phrase_err)
        phrase_cor = tronquer(phrase_cor)

        # Corriger les deux versions pour obtenir les MotAnalyse
        # On desactive grammaire/ortho pour ne mesurer que le tagging
        res_err = correcteur.corriger(phrase_err)
        res_cor = correcteur.corriger(phrase_cor)

        mots_err = res_err.mots or []
        mots_cor = res_cor.mots or []

        if len(mots_err) != len(mots_cor):
            skipped += 1
            continue
        if not mots_err:
            skipped += 1
            continue

        n = len(mots_err)

        # Trouver le mot cible
        target_idx = None
        for j in range(n):
            if mots_err[j].original.lower() != mots_cor[j].original.lower():
                target_idx = j
                break

        if target_idx is None:
            skipped += 1
            continue

        # --- Metriques sur tous les mots ---
        for j in range(n):
            mot_low = mots_err[j].original.lower()
            pos_pipeline = mots_err[j].pos

            # Ancres POS
            if mot_low in _ANCRES_POS:
                ancre_pos_total += 1
                expected_base = _ANCRES_POS[mot_low]
                actual_base = pos_pipeline.split(":")[0] if pos_pipeline else ""
                if actual_base == expected_base:
                    ancre_pos_ok += 1
                else:
                    ancre_pos_errors[(mot_low, expected_base, actual_base)] += 1

            # Ancres NOMBRE
            if mot_low in _ANCRES_NOMBRE:
                ancre_nombre_total += 1
                expected_n = _ANCRES_NOMBRE[mot_low]
                actual_n = mots_err[j].morpho.get("nombre", "") if mots_err[j].morpho else ""
                if actual_n == expected_n:
                    ancre_nombre_ok += 1
                else:
                    ancre_nombre_errors[(mot_low, expected_n, actual_n)] += 1

            # POS mots de contenu (non-ancres)
            if mot_low not in _ANCRES_POS:
                infos = lex_norm.info(mot_low)
                if infos:
                    cgrams = {e.get("cgram") for e in infos if e.get("cgram")}
                    if len(cgrams) == 1:
                        expected_pos = next(iter(cgrams)).split(":")[0]
                        actual_pos = pos_pipeline.split(":")[0] if pos_pipeline else ""
                        contenu_pos_total += 1
                        if actual_pos == expected_pos:
                            contenu_pos_ok += 1
                        else:
                            contenu_pos_errors[(mot_low, expected_pos, actual_pos)] += 1

        # --- Metriques sur le mot cible ---
        pos_err = mots_err[target_idx].pos
        pos_cor = mots_cor[target_idx].pos

        # Stabilite
        base_err = pos_err.split(":")[0] if pos_err else ""
        base_cor = pos_cor.split(":")[0] if pos_cor else ""
        if base_err == base_cor:
            stable_pos += 1
        else:
            unstable_pos += 1

        # POS correct vs lexique du mot corrige
        mot_cor_low = mots_cor[target_idx].original.lower()
        infos_cor = lex_norm.info(mot_cor_low)
        if infos_cor:
            best = max(infos_cor, key=lambda e: float(e.get("freq") or 0))
            lex_pos = best.get("cgram", "").split(":")[0]
            if lex_pos:
                target_pos_total += 1
                if base_err == lex_pos:
                    target_pos_ok += 1
                else:
                    target_pos_errors[(mots_err[target_idx].original.lower(), lex_pos, base_err)] += 1

        # NOMBRE du mot cible
        if infos_cor:
            best_n = max(infos_cor, key=lambda e: float(e.get("freq") or 0))
            nombre_lex = best_n.get("nombre", "")
            if nombre_lex:
                if nombre_lex in ("Sing", "s"):
                    nombre_lex_n = "s"
                elif nombre_lex in ("Plur", "p"):
                    nombre_lex_n = "p"
                else:
                    nombre_lex_n = nombre_lex

                target_nombre_total += 1
                morpho_err = mots_err[target_idx].morpho or {}
                morpho_cor = mots_cor[target_idx].morpho or {}
                if morpho_err.get("nombre") == nombre_lex_n:
                    target_nombre_ok_err += 1
                if morpho_cor.get("nombre") == nombre_lex_n:
                    target_nombre_ok_cor += 1

    # Affichage
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  (skipped: {skipped})")

    def pct(ok, total):
        return f"{100*ok/total:.1f}%" if total > 0 else "N/A"

    print(f"\n  --- POS ancres (mots-outils) ---")
    print(f"  Correct: {ancre_pos_ok}/{ancre_pos_total} ({pct(ancre_pos_ok, ancre_pos_total)})")
    if ancre_pos_errors:
        print(f"  Erreurs :")
        for (mot, att, pred), cnt in ancre_pos_errors.most_common(10):
            print(f"    {mot:12s} attendu={att:6s} obtenu={pred:6s} : {cnt}")

    print(f"\n  --- NOMBRE ancres (DET/PRO) ---")
    print(f"  Correct: {ancre_nombre_ok}/{ancre_nombre_total} ({pct(ancre_nombre_ok, ancre_nombre_total)})")
    if ancre_nombre_errors:
        print(f"  Erreurs :")
        for (mot, att, pred), cnt in ancre_nombre_errors.most_common(10):
            print(f"    {mot:12s} attendu={att} obtenu={pred:5s} : {cnt}")

    print(f"\n  --- POS mots de contenu non-ambigus ---")
    print(f"  Correct: {contenu_pos_ok}/{contenu_pos_total} ({pct(contenu_pos_ok, contenu_pos_total)})")
    if contenu_pos_errors:
        print(f"  Top erreurs :")
        for (mot, att, pred), cnt in contenu_pos_errors.most_common(15):
            print(f"    {mot:15s} attendu={att:6s} obtenu={pred:6s} : {cnt}")

    total_t = stable_pos + unstable_pos
    print(f"\n  --- Stabilite POS mot cible (errone vs corrige) ---")
    print(f"  Stable: {stable_pos}/{total_t} ({pct(stable_pos, total_t)})")

    print(f"\n  --- POS correct mot cible errone (vs lexique corrige) ---")
    print(f"  Correct: {target_pos_ok}/{target_pos_total} ({pct(target_pos_ok, target_pos_total)})")
    if target_pos_errors:
        print(f"  Top erreurs :")
        for (mot, att, pred), cnt in target_pos_errors.most_common(10):
            print(f"    {mot:15s} attendu={att:6s} obtenu={pred:6s} : {cnt}")

    print(f"\n  --- NOMBRE mot cible ---")
    print(f"  Correct sur phrase erronee:  {target_nombre_ok_err}/{target_nombre_total} ({pct(target_nombre_ok_err, target_nombre_total)})")
    print(f"  Correct sur phrase corrigee: {target_nombre_ok_cor}/{target_nombre_total} ({pct(target_nombre_ok_cor, target_nombre_total)})")

    return {
        "ancre_pos": (ancre_pos_ok, ancre_pos_total),
        "ancre_nombre": (ancre_nombre_ok, ancre_nombre_total),
        "contenu_pos": (contenu_pos_ok, contenu_pos_total),
        "target_pos": (target_pos_ok, target_pos_total),
        "target_nombre_err": (target_nombre_ok_err, target_nombre_total),
        "target_nombre_cor": (target_nombre_ok_cor, target_nombre_total),
        "stabilite": (stable_pos, total_t),
    }


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig
    from lectura_correcteur._utils import LexiqueNormalise

    print("Chargement du lexique...")
    lexique = Lexique(LEXIQUE_DB)
    lex_norm = LexiqueNormalise(lexique)

    # Charger paires accord
    paires = []
    with open(WICOPACO_TSV, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3 or row[0].startswith("#") or row[0] == "type_erreur":
                continue
            if row[0].strip() != "accord":
                continue
            paires.append((row[1].strip(), row[2].strip()))
            if len(paires) >= 1000:
                break
    print(f"Paires : {len(paires)}")

    # Pipeline 1 : Baseline (LexiqueTagger, pas de Viterbi)
    print("\n--- Pipeline BASELINE ---")
    t0 = time.time()
    config_base = CorrecteurConfig()
    c_base = Correcteur(lexique, config=config_base)
    r_base = evaluer_pipeline(c_base, paires, lex_norm, "PIPELINE BASELINE (LexiqueTagger)")
    print(f"  Temps: {time.time()-t0:.1f}s")

    # Pipeline 2 : Hybride (G2P + overrides + boost)
    print("\n--- Pipeline HYBRIDE ---")
    t0 = time.time()
    config_hyb = CorrecteurConfig(activer_tagger_hybride=True)
    c_hyb = Correcteur(lexique, config=config_hyb)
    r_hyb = evaluer_pipeline(c_hyb, paires, lex_norm, "PIPELINE HYBRIDE (G2P + overrides + boost)")
    print(f"  Temps: {time.time()-t0:.1f}s")

    # Pipeline 3 : Hybride + Viterbi POS
    print("\n--- Pipeline HYBRIDE + VITERBI ---")
    t0 = time.time()
    config_vit = CorrecteurConfig(activer_tagger_hybride=True, activer_viterbi=True)
    c_vit = Correcteur(lexique, config=config_vit)
    r_vit = evaluer_pipeline(c_vit, paires, lex_norm, "PIPELINE HYBRIDE + VITERBI POS")
    print(f"  Temps: {time.time()-t0:.1f}s")

    # Tableau comparatif
    print(f"\n{'='*70}")
    print(f"  COMPARAISON PIPELINES")
    print(f"{'='*70}")

    def fmt(r, key):
        ok, total = r[key]
        return f"{100*ok/total:.1f}%" if total > 0 else "N/A"

    metrics = [
        ("POS ancres (mots-outils)", "ancre_pos"),
        ("NOMBRE ancres (DET/PRO)", "ancre_nombre"),
        ("POS contenu non-ambigus", "contenu_pos"),
        ("POS mot cible errone", "target_pos"),
        ("Stabilite POS cible", "stabilite"),
        ("NOMBRE cible (phrase err)", "target_nombre_err"),
        ("NOMBRE cible (phrase cor)", "target_nombre_cor"),
    ]

    print(f"\n  {'Metrique':<30s} {'Baseline':>10s} {'Hybride':>10s} {'Hyb+Viterbi':>12s}")
    print(f"  {'-'*62}")
    for label, key in metrics:
        print(f"  {label:<30s} {fmt(r_base, key):>10s} {fmt(r_hyb, key):>10s} {fmt(r_vit, key):>12s}")


if __name__ == "__main__":
    main()
