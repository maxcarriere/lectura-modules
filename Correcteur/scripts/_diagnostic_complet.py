#!/usr/bin/env python3
"""Diagnostic complet FP + FN sur tous les corpus disponibles.

Analyse les patterns d'erreur pour identifier les axes d'amélioration.
- FP : catégorise TOUTES les corrections parasites (négatif + WiCoPaCo)
- FN : catégorise TOUS les cas manqués (WiCoPaCo complet 6166 paires)
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
NEGATIF_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/negatif_wicopaco.tsv"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


def extraire_mots(texte):
    return [m.group().lower() for m in _MOT_RE.finditer(texte)]


def normaliser(texte):
    return " ".join(texte.strip().lower().split())


def diff_mots(phrase_a, phrase_b):
    """Retourne les différences mot à mot entre deux phrases.

    Returns list of (index, mot_a, mot_b) pour chaque position différente.
    """
    mots_a = extraire_mots(phrase_a)
    mots_b = extraire_mots(phrase_b)
    diffs = []
    for i in range(min(len(mots_a), len(mots_b))):
        if mots_a[i] != mots_b[i]:
            diffs.append((i, mots_a[i], mots_b[i]))
    # Mots en plus/moins
    if len(mots_a) < len(mots_b):
        for i in range(len(mots_a), len(mots_b)):
            diffs.append((i, "<ABSENT>", mots_b[i]))
    elif len(mots_b) < len(mots_a):
        for i in range(len(mots_b), len(mots_a)):
            diffs.append((i, mots_a[i], "<SUPPRIME>"))
    return diffs


def classifier_changement(mot_orig, mot_corr, lexique):
    """Classifie un changement mot_orig → mot_corr."""
    lo, lc = mot_orig.lower(), mot_corr.lower()

    # A → À (début de phrase)
    if lo == "a" and lc == "à":
        return "A→À"
    if lo == "à" and lc == "a":
        return "À→A"

    # Homophone grammatical connu
    _HOMO_GRAM = {
        frozenset({"et", "est"}), frozenset({"son", "sont"}),
        frozenset({"a", "à"}), frozenset({"ou", "où"}),
        frozenset({"on", "ont"}), frozenset({"ce", "se"}),
        frozenset({"la", "là"}), frozenset({"leur", "leurs"}),
        frozenset({"ça", "sa"}), frozenset({"ces", "ses"}),
        frozenset({"peu", "peut", "peux"}), frozenset({"ma", "m'a"}),
        frozenset({"dans", "d'en"}), frozenset({"sans", "s'en"}),
        frozenset({"mais", "mes"}), frozenset({"quand", "quant"}),
    }
    for groupe in _HOMO_GRAM:
        if lo in groupe and lc in groupe:
            return f"HOMO_GRAM:{lo}→{lc}"

    # Accent seul (même base sans accents)
    import unicodedata
    def strip_accents(s):
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    if strip_accents(lo) == strip_accents(lc):
        return f"ACCENT:{lo}→{lc}"

    # Pluriel/singulier
    if lo + "s" == lc or lc + "s" == lo:
        return f"NOMBRE:{lo}→{lc}"
    if lo + "x" == lc or lc + "x" == lo:
        return f"NOMBRE:{lo}→{lc}"

    # Conjugaison (même radical)
    if len(lo) > 3 and len(lc) > 3 and lo[:3] == lc[:3]:
        # Vérifier si les deux sont des VER dans le lexique
        infos_o = lexique.info(lo)
        infos_c = lexique.info(lc)
        ver_o = any(e.get("cgram", "").startswith("VER") or e.get("cgram") == "AUX" for e in infos_o)
        ver_c = any(e.get("cgram", "").startswith("VER") or e.get("cgram") == "AUX" for e in infos_c)
        if ver_o and ver_c:
            return f"CONJUGAISON:{lo}→{lc}"

    # Homophone lexical (même prononciation)
    phone_o = lexique.phone_de(lo) if hasattr(lexique, "phone_de") else None
    phone_c = lexique.phone_de(lc) if hasattr(lexique, "phone_de") else None
    if phone_o and phone_c and phone_o == phone_c:
        return f"HOMO_LEX:{lo}→{lc}"

    # OOV → correction ortho
    existe_o = lexique.existe(lo)
    existe_c = lexique.existe(lc)
    if not existe_o and existe_c:
        return f"ORTHO_OOV:{lo}→{lc}"
    if existe_o and existe_c:
        return f"SUBST_LEX:{lo}→{lc}"
    if existe_o and not existe_c:
        return f"CASSE_OOV:{lo}→{lc}"

    return f"AUTRE:{lo}→{lc}"


def analyser_fp_negatif(correcteur, lexique, max_n=0):
    """Analyse tous les FP sur le corpus négatif."""
    phrases = []
    with open(NEGATIF_TSV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                phrases.append(parts[-1] if len(parts) > 1 else parts[0])

    if max_n > 0:
        phrases = phrases[:max_n]

    fp_categories = Counter()
    fp_details = defaultdict(list)
    total = len(phrases)
    ok = 0

    for phrase in phrases:
        res = correcteur.corriger(phrase)
        if normaliser(res.phrase_corrigee) == normaliser(phrase):
            ok += 1
            continue

        diffs = diff_mots(phrase, res.phrase_corrigee)
        for idx, mot_orig, mot_corr in diffs:
            cat = classifier_changement(mot_orig, mot_corr, lexique)
            cat_base = cat.split(":")[0]
            fp_categories[cat_base] += 1
            if len(fp_details[cat]) < 5:  # max 5 exemples par sous-cat
                fp_details[cat].append(phrase[:100])

    return total, ok, fp_categories, fp_details


def analyser_fn_wicopaco(correcteur, lexique, max_n=0):
    """Analyse tous les FN sur WiCoPaCo (corpus complet)."""
    cats_tsv = {}
    with open(WICOPACO_TSV, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3 or row[0].startswith("#") or row[0] == "type_erreur":
                continue
            cat = row[0].strip()
            cats_tsv.setdefault(cat, []).append((row[1].strip(), row[2].strip()))

    all_paires = []
    for cat, pairs in cats_tsv.items():
        for e, a in pairs:
            all_paires.append((cat, e, a))

    if max_n > 0:
        all_paires = all_paires[:max_n]

    fn_categories = Counter()
    fn_details = defaultdict(list)
    wrong_categories = Counter()
    total = len(all_paires)
    ok = 0
    fn = 0
    wrong = 0
    fp = 0

    # Compteurs par catégorie WiCoPaCo
    stats_par_cat = defaultdict(lambda: {"total": 0, "ok": 0, "fn": 0, "wrong": 0})

    for wico_cat, erronee, attendue in all_paires:
        # Trouver le mot cible
        mots_err = extraire_mots(erronee)
        mots_att = extraire_mots(attendue)

        stats_par_cat[wico_cat]["total"] += 1

        # Tronquer pour phrases longues
        tokens_err = erronee.split()
        tokens_att = attendue.split()
        if len(tokens_err) != len(tokens_att):
            wrong += 1
            stats_par_cat[wico_cat]["wrong"] += 1
            continue

        idx_diff = None
        for i, (a, b) in enumerate(zip(tokens_err, tokens_att)):
            if a != b:
                idx_diff = i
                break
        if idx_diff is None:
            ok += 1
            stats_par_cat[wico_cat]["ok"] += 1
            continue

        # Tronquer contexte
        fenetre = 12
        start = max(0, idx_diff - fenetre)
        end = min(len(tokens_err), idx_diff + fenetre + 1)
        err_ctx = " ".join(tokens_err[start:end])
        att_ctx = " ".join(tokens_att[start:end])

        # Trouver mot cible dans le contexte tronqué
        mots_err_ctx = extraire_mots(err_ctx)
        mots_att_ctx = extraire_mots(att_ctx)
        if len(mots_err_ctx) != len(mots_att_ctx):
            wrong += 1
            stats_par_cat[wico_cat]["wrong"] += 1
            continue

        mot_err = mot_att = ""
        idx_local = None
        for i, (a, b) in enumerate(zip(mots_err_ctx, mots_att_ctx)):
            if a != b:
                idx_local = i
                mot_err, mot_att = a, b
                break
        if idx_local is None:
            ok += 1
            stats_par_cat[wico_cat]["ok"] += 1
            continue

        res = correcteur.corriger(err_ctx)
        mots_obtenu = extraire_mots(res.phrase_corrigee)

        mot_trouve = None
        for offset in (0, -1, 1, -2, 2, -3, 3):
            j = idx_local + offset
            if 0 <= j < len(mots_obtenu):
                if mots_obtenu[j] in (mot_att, mot_err):
                    mot_trouve = mots_obtenu[j]
                    break

        if mot_trouve == mot_att:
            ok += 1
            stats_par_cat[wico_cat]["ok"] += 1
        elif mot_trouve == mot_err:
            fn += 1
            stats_par_cat[wico_cat]["fn"] += 1
            pair_key = f"{mot_err}→{mot_att}"
            fn_categories[pair_key] += 1
            if len(fn_details[pair_key]) < 3:
                fn_details[pair_key].append(err_ctx[:100])
        else:
            wrong += 1
            stats_par_cat[wico_cat]["wrong"] += 1
            pair_key = f"{mot_err}→{mot_att}(got:{mot_trouve})"
            wrong_categories[pair_key] += 1

    return total, ok, fn, wrong, fn_categories, fn_details, wrong_categories, stats_par_cat


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig
    from lectura_correcteur._utils import LexiqueNormalise

    print("Chargement du lexique...")
    lexique = Lexique(LEXIQUE_DB)
    lex_norm = LexiqueNormalise(lexique)

    # Utiliser le baseline pour le diagnostic (les FP/FN existent dans les deux)
    config = CorrecteurConfig()
    correcteur = Correcteur(lexique, config=config)
    print(f"Tagger: {type(correcteur._tagger).__name__}")

    # ================================================================
    # PARTIE 1 : ANALYSE FP SUR CORPUS NEGATIF (1000 phrases correctes)
    # ================================================================
    print(f"\n{'='*70}")
    print("  PARTIE 1 : ANALYSE DES FAUX POSITIFS (corpus négatif)")
    print(f"{'='*70}")

    t0 = time.time()
    total_n, ok_n, fp_cats, fp_details = analyser_fp_negatif(correcteur, lex_norm)
    fp_count = total_n - ok_n
    print(f"\n  Phrases correctes: {ok_n}/{total_n} ({100*ok_n/total_n:.1f}%)")
    print(f"  Faux positifs: {fp_count} ({100*fp_count/total_n:.1f}%)")

    print(f"\n  --- Répartition des FP par catégorie ---")
    for cat, count in fp_cats.most_common(20):
        print(f"    {cat:20s} : {count:4d}")

    print(f"\n  --- Top 40 FP détaillés (avec exemples) ---")
    # Trier par fréquence
    sorted_details = sorted(fp_details.items(), key=lambda x: -len(x[1]))
    # Regrouper par catégorie et compter
    detail_counts = Counter()
    for key, examples in fp_details.items():
        # Compter occurrences réelles (pas limité à 5)
        pass

    for key, examples in sorted_details[:40]:
        cat_base = key.split(":")[0]
        detail = key.split(":", 1)[1] if ":" in key else key
        print(f"    {key}")
        for ex in examples[:2]:
            print(f"      ex: {ex}")

    elapsed_fp = time.time() - t0
    print(f"\n  Temps FP: {elapsed_fp:.1f}s")

    # ================================================================
    # PARTIE 2 : ANALYSE FN SUR WICOPACO COMPLET (6166 paires)
    # ================================================================
    print(f"\n{'='*70}")
    print("  PARTIE 2 : ANALYSE DES FAUX NEGATIFS (WiCoPaCo complet)")
    print(f"{'='*70}")

    t0 = time.time()
    total_w, ok_w, fn_w, wrong_w, fn_cats, fn_details, wrong_cats, stats_cat = \
        analyser_fn_wicopaco(correcteur, lex_norm)

    print(f"\n  Total: {total_w}")
    print(f"  Corrects: {ok_w} ({100*ok_w/total_w:.1f}%)")
    print(f"  FN (non corrigés): {fn_w} ({100*fn_w/total_w:.1f}%)")
    print(f"  WRONG (mauvaise correction): {wrong_w} ({100*wrong_w/total_w:.1f}%)")

    print(f"\n  --- Score par catégorie WiCoPaCo ---")
    for cat in sorted(stats_cat.keys()):
        s = stats_cat[cat]
        t = s["total"]
        o = s["ok"]
        pct = 100 * o / t if t > 0 else 0
        print(f"    {cat:25s} : {o:4d}/{t:4d} ({pct:5.1f}%)  FN={s['fn']}  WRONG={s['wrong']}")

    print(f"\n  --- Top 50 paires FN les plus fréquentes ---")
    for pair, count in fn_cats.most_common(50):
        examples = fn_details.get(pair, [])
        print(f"    {pair:30s} : {count:3d}")
        for ex in examples[:1]:
            print(f"      ex: {ex}")

    print(f"\n  --- Top 20 paires WRONG les plus fréquentes ---")
    for pair, count in wrong_cats.most_common(20):
        print(f"    {pair:45s} : {count:3d}")

    # Regrouper FN par type de confusion
    print(f"\n  --- FN regroupés par type ---")
    fn_types = Counter()
    for pair, count in fn_cats.items():
        parts = pair.split("→")
        if len(parts) == 2:
            mo, ma = parts
            cat = classifier_changement(mo, ma, lex_norm)
            cat_base = cat.split(":")[0]
            fn_types[cat_base] += count
    for typ, count in fn_types.most_common():
        print(f"    {typ:20s} : {count:4d}")

    elapsed_fn = time.time() - t0
    print(f"\n  Temps FN: {elapsed_fn:.1f}s")

    print(f"\n{'='*70}")
    print("  DIAGNOSTIC TERMINE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
