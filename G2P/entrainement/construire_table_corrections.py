#!/usr/bin/env python3
"""Construit la table de corrections g2p_corrections_unifie.json.

1. Charge corrections_candidates.json (sortie de evaluer_mots_isoles.py)
2. Simule les règles sur chaque mot → exclut si la règle corrige correctement
3. Exclut les voyelles mi-ouvertes (tolérées)
4. Ajoute élisions + hard-coded
5. Sauvegarde g2p_corrections_unifie.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lectura_nlp.posttraitement import appliquer_regles_g2p  # noqa: E402

# ── Chemins ──────────────────────────────────────────────────────────────

CANDIDATES_PATH = PROJECT_ROOT / "evaluation" / "corrections_candidates.json"
OUTPUT_PATH = PROJECT_ROOT / "modeles" / "g2p_corrections_unifie.json"

# ── Élisions standard ────────────────────────────────────────────────────

ELISIONS = {
    "j'": "ʒ", "t'": "t", "m'": "m", "l'": "l",
    "d'": "d", "n'": "n", "s'": "s", "c'": "s",
    "qu'": "k",
    "jusqu'": "ʒysk", "lorsqu'": "lɔʁsk", "puisqu'": "pɥisk",
    "quelqu'": "kɛlk", "quoiqu'": "kwak", "presqu'": "pʁɛsk",
}

# ── Hard-coded irréguliers ───────────────────────────────────────────────

HARDCODED = {
    "monsieur": "məsjø",
    "messieurs": "mesjø",
    "madame": "madam",
    "mesdames": "medam",
    "oignon": "ɔɲɔ̃",
    "oignons": "ɔɲɔ̃",
    "femme": "fam",
    "femmes": "fam",
    "fils": "fis",
}

# ── Catégories à exclure (tolérées) ─────────────────────────────────────

CATEGORIES_EXCLUES = {
    "voyelle mi-ouverte (ɛ/e, ɔ/o)",
}


def main() -> None:
    if not CANDIDATES_PATH.exists():
        print(f"ERREUR : {CANDIDATES_PATH} non trouvé")
        print("Lancez d'abord evaluer_mots_isoles.py")
        sys.exit(1)

    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"Chargement : {len(candidates)} candidats")

    # Compteurs
    stats = Counter()
    table: dict[str, str] = {}

    for word, info in candidates.items():
        category = info.get("category", info.get("source", ""))
        gold = info["gold"]
        pred = info["pred"]

        # Exclure les catégories tolérées
        if category in CATEGORIES_EXCLUES:
            stats["exclu_mid_vowel"] += 1
            continue

        # Simuler les règles
        corrected = appliquer_regles_g2p(word, pred)
        if corrected == gold:
            stats["corrige_par_regle"] += 1
            continue

        # Ce mot a besoin d'une entrée dans la table
        table[word] = gold
        stats["table"] += 1

    # Ajouter élisions (toujours, même si déjà dans candidates)
    for form, ipa in ELISIONS.items():
        if form not in table:
            stats["ajout_elision"] += 1
        table[form] = ipa

    # Ajouter hard-coded
    for word, ipa in HARDCODED.items():
        if word not in table:
            stats["ajout_hardcoded"] += 1
        table[word] = ipa

    # Trier par clé
    table_sorted = dict(sorted(table.items()))

    # Sauvegarder
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(table_sorted, f, ensure_ascii=False, indent=1)

    print(f"\nStatistiques :")
    for key, count in stats.most_common():
        print(f"  {key:25s}: {count:>5}")
    print(f"\nTable finale : {len(table_sorted)} entrées")
    print(f"Sauvegardé : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
