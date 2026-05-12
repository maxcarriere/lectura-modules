#!/usr/bin/env python3
"""Analyse detaillee des FN sur le corpus WiCoPaCo grammaire.

Dump les patterns les plus frequents par categorie pour prioriser
les ameliorations du correcteur.

Usage :
    python scripts/analyser_fn_grammaire.py --corpus data/grammaire_wicopaco.tsv
    python scripts/analyser_fn_grammaire.py --corpus data/grammaire_wicopaco.tsv --max 200
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"

_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


def _extraire_mots(texte: str) -> list[str]:
    return [m.group().lower() for m in _MOT_RE.finditer(texte)]


def _trouver_mot_cible(erronee: str, attendue: str) -> tuple[str, str, int] | None:
    mots_err = _extraire_mots(erronee)
    mots_att = _extraire_mots(attendue)
    if len(mots_err) != len(mots_att):
        return None
    for i, (a, b) in enumerate(zip(mots_err, mots_att)):
        if a != b:
            return (a, b, i)
    return None


def _tronquer_contexte(erronee: str, attendue: str, fenetre: int = 12):
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
    return (
        " ".join(tokens_err[start:end]),
        " ".join(tokens_att[start:end]),
    )


def _chercher_mot(mots_obtenu, idx, mot_att, mot_err):
    for offset in (0, -1, 1, -2, 2, -3, 3):
        j = idx + offset
        if 0 <= j < len(mots_obtenu):
            if mots_obtenu[j] == mot_att or mots_obtenu[j] == mot_err:
                return mots_obtenu[j]
    return None


def _classifier_pattern(mot_err: str, mot_att: str) -> str:
    """Classifie le type de changement entre mot_err et mot_att."""
    e, a = mot_err.lower(), mot_att.lower()

    # Accent seul
    import unicodedata
    def sans_accents(s):
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    if sans_accents(e) == sans_accents(a):
        return f"accent:{e}→{a}"

    # er/é confusion
    if (e.endswith("er") and a.endswith("é")) or (e.endswith("é") and a.endswith("er")):
        if e[:-2] == a[:-2] or e[:-1] == a[:-2] or e[:-2] == a[:-1]:
            return "er_e"

    # Suffixe nombre (-s, -x, -ent/-e)
    if a == e + "s" or e == a + "s":
        return "nombre_s"
    if a == e + "x" or e == a + "x":
        return "nombre_x"
    if a == e + "nt" or e == a + "nt":
        return "nombre_nt"
    if a == e + "ent" or e == a + "ent":
        return "nombre_ent"

    # Genre (-e/-euse/-trice etc.)
    if a == e + "e" or e == a + "e":
        return "genre_e"
    if (e.endswith("eux") and a.endswith("euse")) or (e.endswith("euse") and a.endswith("eux")):
        return "genre_eux_euse"

    # Homophones connus
    homophones = {
        frozenset(("a", "à")), frozenset(("ou", "où")),
        frozenset(("et", "est")), frozenset(("son", "sont")),
        frozenset(("on", "ont")), frozenset(("ce", "se")),
        frozenset(("ces", "ses")), frozenset(("la", "là")),
        frozenset(("leur", "leurs")), frozenset(("ça", "sa")),
        frozenset(("ma", "m'a")), frozenset(("ta", "t'a")),
        frozenset(("peu", "peut")), frozenset(("près", "prêt")),
        frozenset(("quand", "quant")), frozenset(("dans", "d'en")),
    }
    for pair in homophones:
        if e in pair and a in pair:
            return f"homophone:{sorted(pair)[0]}/{sorted(pair)[1]}"

    # Conjugaison (meme racine, terminaison differente)
    prefix_commun = 0
    for c1, c2 in zip(e, a):
        if c1 == c2:
            prefix_commun += 1
        else:
            break
    if prefix_commun >= 3 and prefix_commun >= len(e) - 3:
        return f"conjugaison:{e[-3:]}→{a[-3:]}"

    return f"autre:{e}→{a}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    # Charger corpus
    paires = []
    with open(args.corpus, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            paires.append((row[0].strip(), row[1].strip(), row[2].strip()))

    if args.max > 0:
        par_cat: dict[str, list] = {}
        for p in paires:
            par_cat.setdefault(p[0], []).append(p)
        paires = []
        for cat, items in sorted(par_cat.items()):
            paires.extend(items[:args.max])

    print(f"Corpus: {len(paires)} paires")

    # Charger correcteur
    sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    lexique = Lexique(LEXIQUE_DB)
    correcteur = Correcteur(lexique)

    # Evaluer et collecter les FN
    fn_par_cat: dict[str, list[dict]] = {}
    wrong_par_cat: dict[str, list[dict]] = {}
    stats = {"total": 0, "ok": 0, "fn": 0, "wrong": 0, "skip": 0}

    for cat, erronee, attendue in paires:
        stats["total"] += 1
        tronque = _tronquer_contexte(erronee, attendue)
        if tronque is None:
            stats["skip"] += 1
            continue
        err_ctx, att_ctx = tronque
        cible = _trouver_mot_cible(err_ctx, att_ctx)
        if cible is None:
            stats["skip"] += 1
            continue

        mot_err, mot_att, idx = cible
        try:
            obtenu = correcteur.corriger(err_ctx).phrase_corrigee
        except Exception:
            stats["skip"] += 1
            continue

        mots_obtenu = _extraire_mots(obtenu)
        mot_trouve = _chercher_mot(mots_obtenu, idx, mot_att, mot_err)

        if mot_trouve == mot_att:
            stats["ok"] += 1
        elif mot_trouve == mot_err:
            stats["fn"] += 1
            pattern = _classifier_pattern(mot_err, mot_att)
            fn_par_cat.setdefault(cat, []).append({
                "mot_err": mot_err, "mot_att": mot_att,
                "pattern": pattern,
                "contexte": err_ctx[:100],
            })
        else:
            stats["wrong"] += 1
            wrong_par_cat.setdefault(cat, []).append({
                "mot_err": mot_err, "mot_att": mot_att,
                "mot_obtenu": mot_trouve,
                "contexte": err_ctx[:100],
            })

    print(f"\n{'='*70}")
    print(f"STATS: OK={stats['ok']} FN={stats['fn']} Wrong={stats['wrong']} "
          f"Skip={stats['skip']} Total={stats['total']}")
    print(f"{'='*70}\n")

    # Analyse des FN par pattern
    for cat in sorted(fn_par_cat.keys()):
        items = fn_par_cat[cat]
        print(f"\n--- FN {cat} ({len(items)} total) ---")
        pattern_counter = Counter(d["pattern"] for d in items)
        for pattern, count in pattern_counter.most_common(20):
            print(f"  {count:>4d}  {pattern}")
            # Montrer 3 exemples
            exemples = [d for d in items if d["pattern"] == pattern][:3]
            for ex in exemples:
                print(f"        {ex['mot_err']} → {ex['mot_att']}"
                      f"  | {ex['contexte'][:60]}")

    # Analyse des Wrong par pattern
    for cat in sorted(wrong_par_cat.keys()):
        items = wrong_par_cat[cat]
        print(f"\n--- WRONG {cat} ({len(items)} total) ---")
        pattern_counter = Counter(
            _classifier_pattern(d["mot_err"], d["mot_att"]) for d in items
        )
        for pattern, count in pattern_counter.most_common(10):
            print(f"  {count:>4d}  {pattern}")
            exemples = [d for d in items
                        if _classifier_pattern(d["mot_err"], d["mot_att"]) == pattern][:3]
            for ex in exemples:
                print(f"        {ex['mot_err']} → {ex['mot_att']} "
                      f"(obtenu: {ex['mot_obtenu']})")

    # Resume global des patterns FN
    print(f"\n{'='*70}")
    print("RESUME GLOBAL FN par pattern (toutes categories)")
    print(f"{'='*70}")
    all_fn = []
    for items in fn_par_cat.values():
        all_fn.extend(items)
    global_counter = Counter(d["pattern"] for d in all_fn)
    for pattern, count in global_counter.most_common(30):
        print(f"  {count:>4d}  {pattern}")


if __name__ == "__main__":
    main()
