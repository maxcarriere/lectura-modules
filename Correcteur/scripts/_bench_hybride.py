#!/usr/bin/env python3
"""Benchmark comparatif : Baseline (LexiqueTagger) vs Hybride (G2P+overrides).

Lance les deux configurations sur :
1. Cas built-in (272 paires)
2. WiCoPaCo grammaire (echantillon 500 paires, mode mot)
3. Negatif WiCoPaCo (1000 phrases correctes → mesure FP)
"""

import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
WICOPACO_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/grammaire_wicopaco.tsv"
NEGATIF_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/negatif_wicopaco.tsv"

# --- Cas built-in (copie depuis evaluer_corpus_grammaire.py) ---
_CAS_CONJUGAISON = [
    ("je revoir ce film", "je revois ce film"),
    ("tu manger bien", "tu manges bien"),
    ("il partir demain", "il part demain"),
    ("nous partir demain", "nous partons demain"),
    ("vous finir le travail", "vous finissez le travail"),
    ("ils prendre le train", "ils prennent le train"),
    ("je veux revoir ce film", "je veux revoir ce film"),
    ("il va manger", "il va manger"),
    ("pour revoir ce film", "pour revoir ce film"),
    ("il peut partir", "il peut partir"),
    ("elle doit finir", "elle doit finir"),
    ("je mange bien", "je mange bien"),
    ("tu manges bien", "tu manges bien"),
    ("il mange bien", "il mange bien"),
    ("le chat mange", "le chat mange"),
    ("les chats mangent", "les chats mangent"),
]

_CAS_HOMOPHONES = [
    ("il a chaque fois", "il a chaque fois"),
    ("a chaque fois", "à chaque fois"),
    ("il et venu", "il est venu"),
    ("ils on dit", "ils ont dit"),
    ("grâce a la vie", "grâce à la vie"),
    ("il a été tres content", "il a été très content"),
    ("un peut plus tard", "un peu plus tard"),
    ("de son coté", "de son côté"),
    ("face a la mer", "face à la mer"),
    ("trois sans mètres", "trois cent mètres"),
    ("elle a vu sa fille", "elle a vu sa fille"),
    ("on a vu sont film", "on a vu son film"),
    ("il est aller la bas", "il est allé là-bas"),
    ("il a raison", "il a raison"),
    ("de son côté", "de son côté"),
    ("un peu plus", "un peu plus"),
    ("sans doute", "sans doute"),
    ("elle est là", "elle est là"),
    ("le téléphone a sonné", "le téléphone a sonné"),
    ("ils sont partis avec son vélo", "ils sont partis avec son vélo"),
    ("ils on prepare le repas", "ils ont préparé le repas"),
    ("il a prépare le repas", "il a préparé le repas"),
    ("il a sonne a la porte", "il a sonné à la porte"),
]

_CAS_ACCORDS = [
    ("les chat", "les chats"),
    ("le chats", "le chat"),
    ("une bon idee", "une bonne idee"),
]


def normaliser(texte):
    return " ".join(texte.strip().lower().split())


_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


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


def evaluer_builtin(correcteur, paires, label=""):
    ok = ko = fn = fp = 0
    errors = []
    for cat, erronee, attendue in paires:
        res = correcteur.corriger(erronee)
        obtenu = res.phrase_corrigee
        o_n, a_n, e_n = normaliser(obtenu), normaliser(attendue), normaliser(erronee)
        correction_attendue = (a_n != e_n)
        correction_faite = (o_n != e_n)
        if o_n == a_n:
            ok += 1
        elif correction_attendue and not correction_faite:
            fn += 1
            errors.append(("FN", cat, erronee, attendue, obtenu))
        elif not correction_attendue and correction_faite:
            fp += 1
            errors.append(("FP", cat, erronee, attendue, obtenu))
        else:
            ko += 1
            errors.append(("WRONG", cat, erronee, attendue, obtenu))
    total = ok + ko + fn + fp
    return ok, total, fn, fp, ko, errors


def evaluer_wicopaco_mot(correcteur, paires, max_n=500):
    """Evaluation mode mot sur WiCoPaCo (phrases longues)."""
    ok = ko = fn = fp = 0
    errors = []
    for erronee, attendue in paires[:max_n]:
        tronque = tronquer_contexte(erronee, attendue)
        if tronque is None:
            ko += 1
            continue
        err_ctx, att_ctx = tronque

        # Trouver mot cible
        mots_err = extraire_mots(err_ctx)
        mots_att = extraire_mots(att_ctx)
        if len(mots_err) != len(mots_att):
            ko += 1
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
            continue

        res = correcteur.corriger(err_ctx)
        mots_obtenu = extraire_mots(res.phrase_corrigee)

        mot_trouve = None
        # 1. Recherche positionnelle (±3 tokens)
        for offset in (0, -1, 1, -2, 2, -3, 3):
            j = idx + offset
            if 0 <= j < len(mots_obtenu):
                if mots_obtenu[j] in (mot_att, mot_err):
                    mot_trouve = mots_obtenu[j]
                    break
        # 2. Fallback: recherche dans tous les tokens (fusion apostrophes)
        if mot_trouve is None:
            for m in mots_obtenu:
                if m in (mot_att, mot_err):
                    mot_trouve = m
                    break
        # 3. Fallback: sous-token (ex: "l'est" contient "est")
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
        elif mot_trouve == mot_err and correction_attendue:
            fn += 1
            errors.append(("FN", f"{mot_err}→{mot_att}", err_ctx[:80]))
        elif not correction_attendue:
            fp += 1
            errors.append(("FP", f"{mot_err}", err_ctx[:80]))
        else:
            ko += 1
            errors.append(("WRONG", f"{mot_err}→{mot_att} (got {mot_trouve})", err_ctx[:80]))

    total = ok + ko + fn + fp
    return ok, total, fn, fp, ko, errors


def evaluer_negatif(correcteur, phrases, max_n=1000):
    """Mesure le taux de faux positifs sur des phrases correctes."""
    ok = fp = 0
    errors = []
    for phrase in phrases[:max_n]:
        res = correcteur.corriger(phrase)
        if normaliser(res.phrase_corrigee) == normaliser(phrase):
            ok += 1
        else:
            fp += 1
            errors.append(("FP", phrase[:80], res.phrase_corrigee[:80]))
    total = ok + fp
    return ok, total, fp, errors


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
                # Format : phrase seule ou tsv
                parts = line.split("\t")
                phrases.append(parts[-1] if len(parts) > 1 else parts[0])
    return phrases


def print_section(title, ok, total, fn=0, fp=0, ko=0, errors=None, show_errors=True, max_errors=20):
    pct = 100 * ok / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Score: {ok}/{total} ({pct:.1f}%)")
    if fn or fp or ko:
        print(f"  FN={fn}  FP={fp}  WRONG={ko}")
    if errors and show_errors:
        print(f"\n  --- Erreurs (top {min(len(errors), max_errors)}) ---")
        for err in errors[:max_errors]:
            if len(err) == 5:
                typ, cat, e, a, o = err
                print(f"    [{typ}][{cat}] \"{e}\"")
                print(f"        attendu: \"{a}\"")
                print(f"        obtenu:  \"{o}\"")
            elif len(err) == 3:
                typ, detail, ctx = err
                print(f"    [{typ}] {detail} | {ctx}")
            elif len(err) == 4:
                typ, detail, ctx, _ = err
                print(f"    [{typ}] {detail} | {ctx}")


def run_config(label, correcteur, cats_wicopaco=None, phrases_neg=None,
               max_per_cat=500, max_neg=500):
    """Evalue une config sur built-in + WiCoPaCo par categorie + negatif.

    Retourne un dict {cat: (ok, total, fn, fp, ko)} pour le tableau comparatif.
    """
    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"# Tagger: {type(correcteur._tagger).__name__}")
    print(f"{'#'*60}")

    t0 = time.time()
    results = {}

    # 1. Built-in
    paires_bi = []
    paires_bi.extend([("conjugaison", e, a) for e, a in _CAS_CONJUGAISON])
    paires_bi.extend([("homophones", e, a) for e, a in _CAS_HOMOPHONES])
    paires_bi.extend([("accords", e, a) for e, a in _CAS_ACCORDS])
    ok, total, fn, fp, ko, errors = evaluer_builtin(correcteur, paires_bi)
    print_section(f"Built-in ({total} cas)", ok, total, fn, fp, ko, errors)
    results["built-in"] = (ok, total, fn, fp, ko)

    # 2. WiCoPaCo par categorie
    if cats_wicopaco:
        ok_all = fn_all = fp_all = ko_all = total_all = 0
        for cat in sorted(cats_wicopaco.keys()):
            pairs = cats_wicopaco[cat]
            n_eval = min(len(pairs), max_per_cat)
            ok_c, total_c, fn_c, fp_c, ko_c, errors_c = evaluer_wicopaco_mot(
                correcteur, pairs, max_n=n_eval,
            )
            pct = 100 * ok_c / total_c if total_c > 0 else 0
            print_section(
                f"WiCoPaCo [{cat}] ({n_eval}/{len(pairs)})",
                ok_c, total_c, fn_c, fp_c, ko_c, errors_c,
                show_errors=True, max_errors=10,
            )
            results[f"wico_{cat}"] = (ok_c, total_c, fn_c, fp_c, ko_c)
            ok_all += ok_c; fn_all += fn_c; fp_all += fp_c
            ko_all += ko_c; total_all += total_c

        pct_all = 100 * ok_all / total_all if total_all > 0 else 0
        print(f"\n  --- WiCoPaCo TOTAL: {ok_all}/{total_all} ({pct_all:.1f}%)  FN={fn_all}  FP={fp_all}  WRONG={ko_all} ---")
        results["wico_TOTAL"] = (ok_all, total_all, fn_all, fp_all, ko_all)

    # 3. Negatif
    if phrases_neg:
        ok_n, total_n, fp_n, errors_n = evaluer_negatif(correcteur, phrases_neg, max_n=max_neg)
        print_section(f"Negatif (FP sur {total_n} phrases correctes)", ok_n, total_n, fp=fp_n, errors=errors_n, max_errors=10)
        results["negatif"] = (ok_n, total_n, 0, fp_n, 0)

    elapsed = time.time() - t0
    print(f"\n  Temps total: {elapsed:.1f}s")
    return results, elapsed


def print_comparative_table(all_results):
    """Affiche un tableau comparatif final entre toutes les configs."""
    print(f"\n{'#'*80}")
    print("# TABLEAU COMPARATIF")
    print(f"{'#'*80}")

    # Collect all metric keys
    all_keys = set()
    for results in all_results.values():
        all_keys.update(results.keys())
    ordered_keys = sorted(all_keys)

    labels = list(all_results.keys())
    # Shortened labels for table
    short = {l: l[:25] for l in labels}

    # Header
    hdr = f"{'Metrique':30s}"
    for lbl in labels:
        hdr += f" | {short[lbl]:>25s}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for key in ordered_keys:
        row = f"{key:30s}"
        for lbl in labels:
            results = all_results[lbl]
            if key in results:
                ok, total, fn, fp, ko = results[key]
                pct = 100 * ok / total if total > 0 else 0
                cell = f"{ok}/{total} ({pct:.1f}%) FN={fn}"
            else:
                cell = "---"
            row += f" | {cell:>25s}"
        print(row)


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig

    print("Chargement du lexique...")
    lexique = Lexique(LEXIQUE_DB)

    # Pre-charger les donnees WiCoPaCo et Negatif
    cats_wicopaco = None
    phrases_neg = None
    if os.path.exists(WICOPACO_TSV):
        cats_wicopaco = charger_tsv(WICOPACO_TSV)
        for cat, pairs in cats_wicopaco.items():
            print(f"  WiCoPaCo [{cat}]: {len(pairs)} paires")
    if os.path.exists(NEGATIF_TSV):
        phrases_neg = charger_negatif(NEGATIF_TSV)
        print(f"  Negatif: {len(phrases_neg)} phrases")

    # Max par categorie : 500 accord, toutes conjugaison/homophone, 500 negatif
    MAX_PER_CAT = 500
    MAX_NEG = 500

    all_results = {}

    # --- Baseline ---
    config_base = CorrecteurConfig()
    c_base = Correcteur(lexique, config=config_base)
    r, _ = run_config("Baseline", c_base,
                      cats_wicopaco, phrases_neg, MAX_PER_CAT, MAX_NEG)
    all_results["Baseline"] = r

    # --- Baseline + Accord PM ---
    config_pm = CorrecteurConfig(activer_accord_pm=True)
    c_pm = Correcteur(lexique, config=config_pm)
    r, _ = run_config("Baseline+PM", c_pm,
                      cats_wicopaco, phrases_neg, MAX_PER_CAT, MAX_NEG)
    all_results["Baseline+PM"] = r

    # --- Hybride ---
    config_hyb = CorrecteurConfig(activer_tagger_hybride=True)
    c_hyb = Correcteur(lexique, config=config_hyb)
    r, _ = run_config("Hybride", c_hyb,
                      cats_wicopaco, phrases_neg, MAX_PER_CAT, MAX_NEG)
    all_results["Hybride"] = r

    # --- Hybride + Accord PM ---
    config_hyb_pm = CorrecteurConfig(
        activer_tagger_hybride=True, activer_accord_pm=True,
    )
    c_hyb_pm = Correcteur(lexique, config=config_hyb_pm)
    r, _ = run_config("Hybride+PM", c_hyb_pm,
                      cats_wicopaco, phrases_neg, MAX_PER_CAT, MAX_NEG)
    all_results["Hybride+PM"] = r

    # Tableau comparatif
    print_comparative_table(all_results)

    print(f"\n{'#'*60}")
    print("# BENCHMARK TERMINE")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
