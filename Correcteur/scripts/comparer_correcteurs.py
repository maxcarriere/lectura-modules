#!/usr/bin/env python3
"""Comparatif Lectura vs Grammalecte vs LanguageTool sur tous les corpus.

Evalue chaque correcteur sur 3 corpus :
    1. FLE synthetique (mode phrase)
    2. WiCoPaCo grammaire (mode mot, contexte ±12)
    3. Negatif (phrases correctes, mesure FP)

Usage :
    python scripts/comparer_correcteurs.py
    python scripts/comparer_correcteurs.py --max-wicopaco 200 --max-negatif 200
    python scripts/comparer_correcteurs.py --skip-langtool
    python scripts/comparer_correcteurs.py --only-fle
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field

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


def _normaliser(texte: str) -> str:
    return " ".join(texte.strip().lower().split())


@dataclass
class Resultats:
    total: int = 0
    corrects: int = 0
    fn: int = 0
    wrong: int = 0
    fp: int = 0
    skip: int = 0

    @property
    def precision(self) -> float:
        tp = self.corrects
        fp_tot = self.wrong + self.fp
        return tp / (tp + fp_tot) if (tp + fp_tot) > 0 else 1.0

    @property
    def recall(self) -> float:
        tp = self.corrects
        return tp / (tp + self.fn) if (tp + self.fn) > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Adaptateurs
# ---------------------------------------------------------------------------

class AdaptateurLectura:
    def __init__(self):
        sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
        from lectura_lexique import Lexique
        from lectura_correcteur import Correcteur
        self.correcteur = Correcteur(Lexique(LEXIQUE_DB))

    def corriger(self, texte: str) -> str:
        return self.correcteur.corriger(texte).phrase_corrigee


class AdaptateurGrammalecte:
    def __init__(self):
        import grammalecte
        self.gc = grammalecte.GrammarChecker("fr")

    def corriger(self, texte: str) -> str:
        aGrammErrs, aSpellErrs = self.gc.getParagraphErrors(texte)
        corrections = []
        for err in aGrammErrs:
            if err.get("aSuggestions"):
                corrections.append((
                    err["nStart"], err["nEnd"], err["aSuggestions"][0],
                ))
        for err in aSpellErrs:
            if err.get("aSuggestions"):
                corrections.append((
                    err["nStart"], err["nEnd"], err["aSuggestions"][0],
                ))
        corrections.sort(key=lambda x: x[0], reverse=True)
        result = texte
        for start, end, suggestion in corrections:
            result = result[:start] + suggestion + result[end:]
        return result


class AdaptateurLanguageTool:
    def __init__(self):
        import language_tool_python
        self._ltp = language_tool_python
        self.lt = language_tool_python.LanguageTool("fr-FR")

    def corriger(self, texte: str) -> str:
        matches = self.lt.check(texte)
        return self._ltp.utils.correct(texte, matches)

    def close(self):
        self.lt.close()


# ---------------------------------------------------------------------------
# Evaluateurs
# ---------------------------------------------------------------------------

def evaluer_phrase(paires, adaptateur, label):
    """Mode phrase : comparaison globale (pour FLE, phrases courtes)."""
    par_cat: dict[str, Resultats] = {}
    total = Resultats()
    t0 = time.time()

    for i, (cat, erronee, attendue) in enumerate(paires):
        if cat not in par_cat:
            par_cat[cat] = Resultats()
        res = par_cat[cat]
        total.total += 1
        res.total += 1

        try:
            obtenu = adaptateur.corriger(erronee)
        except Exception:
            res.skip += 1
            total.skip += 1
            continue

        obtenu_n = _normaliser(obtenu)
        attendu_n = _normaliser(attendue)
        erronee_n = _normaliser(erronee)

        correction_attendue = (attendu_n != erronee_n)
        correction_faite = (obtenu_n != erronee_n)

        if obtenu_n == attendu_n:
            res.corrects += 1
            total.corrects += 1
        elif correction_attendue and not correction_faite:
            res.fn += 1
            total.fn += 1
        elif not correction_attendue and correction_faite:
            res.fp += 1
            total.fp += 1
        else:
            res.wrong += 1
            total.wrong += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(paires)} "
                  f"({elapsed:.0f}s, ~{elapsed/(i+1)*1000:.0f}ms/paire)")

    elapsed = time.time() - t0
    par_cat["TOTAL"] = total
    print(f"  [{label}] {len(paires)} paires en {elapsed:.1f}s "
          f"({elapsed/max(len(paires),1)*1000:.0f}ms/paire)")
    return par_cat


def evaluer_mot(paires, adaptateur, label):
    """Mode mot : comparaison du mot cible (pour WiCoPaCo, phrases longues)."""
    par_cat: dict[str, Resultats] = {}
    total = Resultats()
    t0 = time.time()

    for i, (cat, erronee, attendue) in enumerate(paires):
        if cat not in par_cat:
            par_cat[cat] = Resultats()
        res = par_cat[cat]
        total.total += 1
        res.total += 1

        tronque = _tronquer_contexte(erronee, attendue)
        if tronque is None:
            res.skip += 1
            total.skip += 1
            continue
        err_ctx, att_ctx = tronque
        cible = _trouver_mot_cible(err_ctx, att_ctx)
        if cible is None:
            res.skip += 1
            total.skip += 1
            continue

        mot_err, mot_att, idx = cible

        try:
            obtenu = adaptateur.corriger(err_ctx)
        except Exception:
            res.skip += 1
            total.skip += 1
            continue

        mots_obtenu = _extraire_mots(obtenu)
        mot_trouve = _chercher_mot(mots_obtenu, idx, mot_att, mot_err)

        if mot_trouve == mot_att:
            res.corrects += 1
            total.corrects += 1
        elif mot_trouve == mot_err and (mot_err != mot_att):
            res.fn += 1
            total.fn += 1
        elif mot_err == mot_att and mot_trouve != mot_att:
            res.fp += 1
            total.fp += 1
        else:
            res.wrong += 1
            total.wrong += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(paires)} "
                  f"({elapsed:.0f}s, ~{elapsed/(i+1)*1000:.0f}ms/paire)")

    elapsed = time.time() - t0
    par_cat["TOTAL"] = total
    print(f"  [{label}] {len(paires)} paires en {elapsed:.1f}s "
          f"({elapsed/max(len(paires),1)*1000:.0f}ms/paire)")
    return par_cat


def evaluer_negatif(paires, adaptateur, label):
    """Mode negatif : phrases correctes, mesure FP."""
    total = Resultats()
    t0 = time.time()

    for i, (_, phrase, _) in enumerate(paires):
        total.total += 1
        try:
            obtenu = adaptateur.corriger(phrase)
        except Exception:
            total.skip += 1
            continue

        obtenu_n = _normaliser(obtenu)
        phrase_n = _normaliser(phrase)

        if obtenu_n != phrase_n:
            total.fp += 1
        else:
            total.corrects += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(paires)} "
                  f"({elapsed:.0f}s, ~{elapsed/(i+1)*1000:.0f}ms/paire)")

    elapsed = time.time() - t0
    print(f"  [{label}] {len(paires)} paires en {elapsed:.1f}s "
          f"({elapsed/max(len(paires),1)*1000:.0f}ms/paire)")
    return {"TOTAL": total}


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def charger_corpus(path):
    paires = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            if row[0].startswith("#") or row[0] == "type_erreur":
                continue
            paires.append((row[0].strip(), row[1].strip(), row[2].strip()))
    return paires


def afficher_tableau(titre, resultats_par_outil, categories=None):
    if not resultats_par_outil:
        return
    print(f"\n{'='*70}")
    print(titre)
    print(f"{'='*70}\n")

    if categories is None:
        categories = sorted(set(
            cat for r in resultats_par_outil.values()
            for cat in r if cat != "TOTAL"
        ))
        categories.append("TOTAL")

    for cat in categories:
        print(f"--- {cat} ---")
        header = (f"{'Correcteur':15s} {'OK':>6s} {'Total':>5s}"
                  f" {'FN':>5s} {'Wrong':>5s} {'FP':>4s}"
                  f" | {'P':>5s} {'R':>5s} {'F1':>5s}")
        print(header)
        print("-" * len(header))
        for name, res_dict in resultats_par_outil.items():
            r = res_dict.get(cat)
            if r is None:
                continue
            print(f"{name:15s} {r.corrects:>6d} {r.total:>5d}"
                  f" {r.fn:>5d} {r.wrong:>5d} {r.fp:>4d}"
                  f" | {r.precision:>.3f} {r.recall:>.3f} {r.f1:>.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Comparatif multi-corpus")
    parser.add_argument("--max-wicopaco", type=int, default=200)
    parser.add_argument("--max-negatif", type=int, default=200)
    parser.add_argument("--skip-grammalecte", action="store_true")
    parser.add_argument("--skip-langtool", action="store_true")
    parser.add_argument("--skip-lectura", action="store_true")
    parser.add_argument("--only-fle", action="store_true")
    parser.add_argument("--only-wicopaco", action="store_true")
    parser.add_argument("--only-negatif", action="store_true")
    args = parser.parse_args()

    run_all = not (args.only_fle or args.only_wicopaco or args.only_negatif)
    data_dir = os.path.join(_PROJECT_ROOT, "data")

    # Charger les outils
    outils: dict[str, object] = {}
    lt_instance = None

    if not args.skip_lectura:
        print("Chargement Lectura...")
        outils["Lectura"] = AdaptateurLectura()

    if not args.skip_grammalecte:
        print("Chargement Grammalecte...")
        try:
            outils["Grammalecte"] = AdaptateurGrammalecte()
        except Exception as e:
            print(f"  Grammalecte indisponible: {e}")

    if not args.skip_langtool:
        print("Chargement LanguageTool...")
        try:
            lt_instance = AdaptateurLanguageTool()
            outils["LanguageTool"] = lt_instance
        except Exception as e:
            print(f"  LanguageTool indisponible: {e}")

    print(f"\nOutils: {', '.join(outils.keys())}\n")

    # --- 1. FLE synthetique (mode phrase) ---
    fle_path = os.path.join(data_dir, "grammaire_fle.tsv")
    if (run_all or args.only_fle) and os.path.exists(fle_path):
        paires = charger_corpus(fle_path)
        print(f"FLE: {len(paires)} paires")
        resultats = {}
        for name, outil in outils.items():
            resultats[name] = evaluer_phrase(paires, outil, name)
        afficher_tableau(
            "FLE SYNTHETIQUE (mode phrase, phrases courtes)", resultats)

    # --- 2. WiCoPaCo grammaire (mode mot) ---
    wico_path = os.path.join(data_dir, "grammaire_wicopaco.tsv")
    if (run_all or args.only_wicopaco) and os.path.exists(wico_path):
        paires = charger_corpus(wico_path)
        if args.max_wicopaco > 0:
            par_cat: dict[str, list] = {}
            for p in paires:
                par_cat.setdefault(p[0], []).append(p)
            paires = []
            for cat, items in sorted(par_cat.items()):
                paires.extend(items[:args.max_wicopaco])
        print(f"\nWiCoPaCo: {len(paires)} paires")
        resultats = {}
        for name, outil in outils.items():
            resultats[name] = evaluer_mot(paires, outil, name)
        afficher_tableau(
            f"WICOPACO GRAMMAIRE (mode mot, max {args.max_wicopaco}/cat)",
            resultats)

    # --- 3. Negatif (phrases correctes) ---
    neg_path = os.path.join(data_dir, "negatif_wicopaco.tsv")
    if (run_all or args.only_negatif) and os.path.exists(neg_path):
        paires = charger_corpus(neg_path)
        if args.max_negatif > 0:
            paires = paires[:args.max_negatif]
        print(f"\nNegatif: {len(paires)} phrases correctes")
        resultats = {}
        for name, outil in outils.items():
            resultats[name] = evaluer_negatif(paires, outil, name)
        afficher_tableau(
            f"NEGATIF (phrases correctes, {len(paires)} phrases)",
            resultats, categories=["TOTAL"])

    if lt_instance is not None:
        lt_instance.close()


if __name__ == "__main__":
    main()
