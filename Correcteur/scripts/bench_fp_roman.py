#!/usr/bin/env python3
"""Benchmark faux positifs — Passe le correcteur sur du texte bien orthographie.

Utilise un texte litteraire (roman) comme corpus negatif : le texte est
suppose correct, donc toute modification du correcteur est un faux positif.

Usage :
    python scripts/bench_fp_roman.py
    python scripts/bench_fp_roman.py --v1          # comparer avec V1
    python scripts/bench_fp_roman.py --limit 100   # limiter le nombre de phrases
    python scripts/bench_fp_roman.py --export fp_roman.tsv  # exporter les FP en TSV
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Phonemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Graphemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Tokeniseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/G2P-Pipeline/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Formules/src")

LEXIQUE_DB = os.path.join(
    _PROJECT_ROOT, "src", "lectura_correcteur", "data", "lexique_correcteur.db",
)

ROMAN_PATH = "/home/moi/Documents/work/projets/humanuscrit/productions/roman_humanuscrit/texte_complet/v3_text.txt"


def extraire_phrases(texte: str) -> list[str]:
    """Extrait les phrases du texte du roman en filtrant les elements de mise en page."""
    lignes = texte.split("\n")
    phrases = []
    buffer = []

    for ligne in lignes:
        ligne_strip = ligne.strip()

        # Sauter les lignes vides, separateurs de page, numeros de page, headers
        if not ligne_strip:
            # Fin de paragraphe : flush buffer
            if buffer:
                paragraphe = " ".join(buffer)
                phrases.extend(_decouper_en_phrases(paragraphe))
                buffer = []
            continue

        # Separateur de page "--- PAGE N ---"
        if re.match(r"^---\s*PAGE\s+\d+\s*---$", ligne_strip):
            if buffer:
                paragraphe = " ".join(buffer)
                phrases.extend(_decouper_en_phrases(paragraphe))
                buffer = []
            continue

        # Numero de page seul (ex: "12", "143")
        if re.match(r"^\d{1,4}$", ligne_strip):
            continue

        # Header "Humanuscrit — System Down"
        if ligne_strip.startswith("Humanuscrit"):
            continue

        # Titre de chapitre repete en bas de page (mot identique au debut)
        if re.match(r"^(Chapitre\s+\d+|Avant-propos|Epilogue|Prologue)\s*$", ligne_strip):
            continue

        # Lignes tres courtes qui sont des titres (pas une phrase)
        # On les garde si elles finissent par de la ponctuation de phrase
        if len(ligne_strip) < 5 and not any(ligne_strip.endswith(p) for p in ".!?…»"):
            continue

        buffer.append(ligne_strip)

    # Flush final
    if buffer:
        paragraphe = " ".join(buffer)
        phrases.extend(_decouper_en_phrases(paragraphe))

    return phrases


def _decouper_en_phrases(texte: str) -> list[str]:
    """Decoupe un paragraphe en phrases individuelles."""
    # Decoupage sur ponctuation forte suivie d'une majuscule ou fin
    # On garde les phrases dialoguees intactes
    phrases = []

    # Split sur . ! ? … suivi d'un espace et majuscule
    # Mais pas sur les abreviations courantes (M., Dr., etc.)
    parts = re.split(r'(?<=[.!?…»])\s+(?=[A-ZÀ-ÿ«—])', texte)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Ignorer les phrases trop courtes (< 3 mots) sauf dialogues
        mots = part.split()
        if len(mots) < 2 and not part.startswith("—"):
            continue
        phrases.append(part)

    return phrases


def analyser_fp(correcteur, phrases: list[str], verbose: bool = True):
    """Passe chaque phrase dans le correcteur et detecte les faux positifs."""
    fp_list = []
    n_ok = 0
    n_erreur = 0
    regles_fp = Counter()

    for i, phrase in enumerate(phrases):
        try:
            resultat = correcteur.corriger(phrase)
        except Exception as e:
            n_erreur += 1
            if verbose:
                print(f"  [ERREUR] phrase {i}: {e}")
            continue

        # Comparer : si le correcteur a modifie la phrase, c'est un FP
        original_n = " ".join(phrase.strip().lower().split())
        corrige_n = " ".join(resultat.phrase_corrigee.strip().lower().split())

        if original_n != corrige_n:
            # Identifier les corrections specifiques
            corrections = []
            for c in resultat.corrections:
                orig = c.original if hasattr(c, 'original') else ""
                corr = c.corrige if hasattr(c, 'corrige') else ""
                regle = c.regle if hasattr(c, 'regle') else "?"
                if orig.lower() != corr.lower():
                    corrections.append({
                        "original": orig,
                        "corrige": corr,
                        "regle": regle,
                    })
                    regles_fp[regle] += 1

            fp_list.append({
                "index": i,
                "phrase": phrase,
                "corrige": resultat.phrase_corrigee,
                "corrections": corrections,
            })
        else:
            n_ok += 1

        # Progression
        if verbose and (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(phrases)} phrases traitees, {len(fp_list)} FP")

    return fp_list, n_ok, n_erreur, regles_fp


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP sur roman")
    parser.add_argument("--fichier", default=ROMAN_PATH,
                        help="Chemin du fichier texte")
    parser.add_argument("--v1", action="store_true",
                        help="Utiliser V1 au lieu de V5")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limiter le nombre de phrases (0 = toutes)")
    parser.add_argument("--export", type=str, default="",
                        help="Exporter les FP en TSV")
    args = parser.parse_args()

    # Charger le texte
    print(f"Chargement : {args.fichier}")
    with open(args.fichier, encoding="utf-8") as f:
        texte = f.read()

    phrases = extraire_phrases(texte)
    print(f"Phrases extraites : {len(phrases)}")

    if args.limit > 0:
        phrases = phrases[:args.limit]
        print(f"  (limite a {args.limit} phrases)")

    # Charger le correcteur
    from lectura_correcteur._lexique_lite import LexiqueLite
    lexique = LexiqueLite(LEXIQUE_DB)

    if args.v1:
        from lectura_correcteur import Correcteur
        correcteur = Correcteur(lexique)
        label = "V1"
    else:
        from lectura_correcteur import CorrecteurV5, CorrecteurV5Config
        cfg = CorrecteurV5Config()
        correcteur = CorrecteurV5(lexique, config=cfg)
        label = f"V5 (P2G={'ON' if correcteur.p2g_disponible else 'OFF'})"

    print(f"Correcteur : {label}")
    print()

    # Lancer l'analyse
    t0 = time.time()
    fp_list, n_ok, n_erreur, regles_fp = analyser_fp(correcteur, phrases)
    elapsed = time.time() - t0

    # Afficher les resultats
    n_total = len(phrases)
    n_fp = len(fp_list)
    taux_fp = n_fp / n_total * 100 if n_total > 0 else 0

    print(f"\n{'='*70}")
    print(f"  Resultats — {label}")
    print(f"{'='*70}")
    print(f"  Phrases totales : {n_total}")
    print(f"  OK (inchangees) : {n_ok} ({n_ok/n_total*100:.1f}%)")
    print(f"  Faux positifs   : {n_fp} ({taux_fp:.1f}%)")
    if n_erreur:
        print(f"  Erreurs exec    : {n_erreur}")
    print(f"  Temps           : {elapsed:.1f}s ({n_total/elapsed:.0f} phrases/s)")

    # Top regles FP
    if regles_fp:
        print(f"\n--- Top regles generant des FP ---")
        for regle, count in regles_fp.most_common(20):
            print(f"  {regle:40s} : {count}")

    # Detail des FP
    if fp_list:
        print(f"\n--- Detail des faux positifs ({n_fp}) ---")
        for fp in fp_list:
            print(f"\n  [{fp['index']}] ORIGINAL : {fp['phrase'][:120]}")
            print(f"       CORRIGE : {fp['corrige'][:120]}")
            for c in fp["corrections"]:
                print(f"       {c['regle']:30s} : {c['original']!r} -> {c['corrige']!r}")

    # Export TSV
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            f.write("index\tregle\toriginal\tcorrige\tphrase\n")
            for fp in fp_list:
                for c in fp["corrections"]:
                    f.write(f"{fp['index']}\t{c['regle']}\t{c['original']}\t{c['corrige']}\t{fp['phrase']}\n")
        print(f"\nExporte : {args.export} ({sum(len(fp['corrections']) for fp in fp_list)} lignes)")


if __name__ == "__main__":
    main()
