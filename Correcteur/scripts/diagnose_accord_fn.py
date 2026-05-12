#!/usr/bin/env python3
"""Diagnostic: analyse les 20 premiers exemples 'accord' de WiCoPaCo
pour comprendre pourquoi le correcteur les manque.

Affiche pour chaque paire:
  - La phrase erronee (tronquee)
  - La phrase attendue
  - Ce que le correcteur a produit
  - Match ou pas + details des corrections detectees
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time

# --- sys.path setup ---
_MODULES = "/data/work/projets/lectura/workspace/Modules"
sys.path.insert(0, os.path.join(_MODULES, "Correcteur", "src"))
sys.path.insert(0, os.path.join(_MODULES, "Lexique", "src"))

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura.db"
CORPUS_TSV = "/data/work/projets/lectura/workspace/Corpus/Correcteur/grammaire_wicopaco.tsv"

# --- Truncation logic (from benchmark_unifie.py) ---
_MOT_RE = re.compile(r"[\w]+(?:['\u2019\-][\w]+)*", re.UNICODE)


def _extraire_mots(texte: str) -> list[str]:
    return [m.group().lower() for m in _MOT_RE.finditer(texte)]


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


def _trouver_mot_cible(erronee: str, attendue: str):
    mots_err = _extraire_mots(erronee)
    mots_att = _extraire_mots(attendue)
    if len(mots_err) != len(mots_att):
        return None
    for i, (a, b) in enumerate(zip(mots_err, mots_att)):
        if a != b:
            return (a, b, i)
    return None


def _normaliser(texte: str) -> str:
    return " ".join(texte.strip().lower().split())


def main():
    # Load lexique
    print("Loading Lexique from", LEXIQUE_DB, "...")
    t0 = time.time()
    from lectura_lexique import Lexique
    lex = Lexique(LEXIQUE_DB)
    print(f"  Lexique loaded in {time.time() - t0:.1f}s")

    # Create correcteur with default config
    from lectura_correcteur import Correcteur, CorrecteurConfig
    config = CorrecteurConfig()
    correcteur = Correcteur(lex, config=config)
    print(f"  Correcteur ready (config defaults)")

    # Load corpus - first 50 accord lines (we process 20, count skips in 50)
    accord_rows = []
    with open(CORPUS_TSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) >= 3 and row[0] == "accord":
                accord_rows.append((row[1], row[2]))
                if len(accord_rows) >= 50:
                    break

    print(f"\nLoaded {len(accord_rows)} 'accord' rows (first 50)")

    # Count skips in first 50
    skipped_50 = 0
    for erronee, attendue in accord_rows:
        tokens_err = erronee.split()
        tokens_att = attendue.split()
        if len(tokens_err) != len(tokens_att):
            skipped_50 += 1

    print(f"  Skipped (token count mismatch) in first 50: {skipped_50}/{len(accord_rows)}")
    print()

    # Process first 20 accord rows
    n_process = 0
    n_match = 0
    n_detected_wrong = 0
    n_not_detected = 0
    n_skipped = 0

    print("=" * 100)
    print("DIAGNOSTIC: First 20 processable 'accord' examples")
    print("=" * 100)

    for idx, (erronee_raw, attendue_raw) in enumerate(accord_rows):
        if n_process >= 20:
            break

        # Truncate context
        result_trunc = _tronquer_contexte(erronee_raw, attendue_raw)
        if result_trunc is None:
            n_skipped += 1
            continue

        erronee, attendue = result_trunc
        cible = _trouver_mot_cible(erronee, attendue)

        n_process += 1
        print(f"\n--- Example {n_process} (row {idx}) ---")
        if cible:
            mot_err, mot_att, mot_idx = cible
            print(f"  Mot cible: '{mot_err}' -> '{mot_att}' (word idx {mot_idx})")
        else:
            print(f"  Mot cible: could not identify")

        print(f"  Erronee:  {erronee}")
        print(f"  Attendue: {attendue}")

        # Run correcteur
        t1 = time.time()
        resultat = correcteur.corriger(erronee)
        dt = time.time() - t1

        obtenu = resultat.phrase_corrigee
        print(f"  Obtenu:   {obtenu}")
        print(f"  Time:     {dt*1000:.0f}ms")

        # Check match
        norm_att = _normaliser(attendue)
        norm_obt = _normaliser(obtenu)

        if norm_att == norm_obt:
            print(f"  Result:   MATCH")
            n_match += 1
        else:
            # Determine: did it detect the error at all?
            if cible:
                mot_err, mot_att, mot_idx = cible
                mots_obtenu = _extraire_mots(obtenu)

                # Check if the target word was changed
                target_in_output = None
                for offset in (0, -1, 1, -2, 2):
                    j = mot_idx + offset
                    if 0 <= j < len(mots_obtenu):
                        if mots_obtenu[j] == mot_att:
                            target_in_output = "correct"
                            break
                        elif mots_obtenu[j] == mot_err:
                            target_in_output = "unchanged"
                            break

                if target_in_output == "unchanged":
                    print(f"  Result:   MISS - error NOT detected (word unchanged)")
                    n_not_detected += 1
                elif target_in_output == "correct":
                    print(f"  Result:   MISS - target word OK but other differences")
                    n_detected_wrong += 1
                else:
                    print(f"  Result:   MISS - target word changed to something else")
                    n_detected_wrong += 1
            else:
                print(f"  Result:   MISS (cannot identify target word)")
                n_not_detected += 1

        # Show all corrections made
        if resultat.corrections:
            print(f"  Corrections ({len(resultat.corrections)}):")
            for c in resultat.corrections:
                print(f"    [{c.index}] '{c.original}' -> '{c.corrige}' ({c.regle}: {c.explication})")
        else:
            print(f"  Corrections: NONE")

        # Show word-level detail around target
        if cible and resultat.mots:
            mot_err, mot_att, mot_idx = cible
            print(f"  Words around target (idx {mot_idx}):")
            for m in resultat.mots:
                if m.original.lower() == mot_err or m.corrige.lower() == mot_att:
                    flag = " <<<" if m.original.lower() != m.corrige.lower() else ""
                    print(f"    '{m.original}' -> '{m.corrige}' "
                          f"[pos={m.pos}, lexique={m.dans_lexique}, "
                          f"conf={m.confiance:.2f}]{flag}")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Processed:          {n_process}")
    print(f"  Skipped (mismatch): {n_skipped}")
    print(f"  MATCH (correct):    {n_match}")
    print(f"  MISS total:         {n_process - n_match}")
    print(f"    - Not detected:   {n_not_detected}")
    print(f"    - Wrong fix:      {n_detected_wrong}")
    print(f"  Skip rate (50):     {skipped_50}/50 = {skipped_50/50*100:.0f}%")


if __name__ == "__main__":
    main()
