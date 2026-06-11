#!/usr/bin/env python3
"""Analyse des mots composes dans le phone_lexicon.

Identifie les entrees du lexique qui contiennent un tiret ou apostrophe,
et verifie si leur IPA peut etre decompose en IPA de mots individuels.

Objectif: construire un dictionnaire fiable de fusions IPA -> mot compose.

Usage:
    python scripts/analyser_composes.py
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Graphemiseur" / "src"))

from lectura_graphemiseur._phone_lexicon import PhoneLexicon


def analyser_composes():
    """Analyse les mots composes du phone_lexicon."""

    db_path = (
        Path(__file__).resolve().parent.parent.parent
        / "Graphemiseur" / "src" / "lectura_graphemiseur" / "modeles" / "phone_lexicon.db"
    )
    lex = PhoneLexicon(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 1. Trouver toutes les entrees composees (avec tiret ou apostrophe)
    table = lex._table
    cursor = conn.execute(
        f"SELECT DISTINCT ortho, phone, freq, cgram FROM {table} "
        "WHERE (ortho LIKE '%-%' OR ortho LIKE '%''%') "
        "AND phone IS NOT NULL AND phone != '' "
        "ORDER BY freq DESC"
    )

    composes_tiret = []
    composes_apo = []
    for row in cursor:
        ortho = row["ortho"] or ""
        phone = row["phone"] or ""
        freq = row["freq"] or 0
        cgram = row["cgram"] or ""

        if "-" in ortho:
            composes_tiret.append((ortho, phone, freq, cgram))
        if "'" in ortho:
            composes_apo.append((ortho, phone, freq, cgram))

    print(f"Entrees composees avec tiret: {len(composes_tiret)}")
    print(f"Entrees composees avec apostrophe: {len(composes_apo)}")

    # 2. Top composes par frequence
    print(f"\n{'='*80}")
    print("Top 50 composes avec tiret (par freq)")
    print(f"{'='*80}")
    seen = set()
    count = 0
    for ortho, phone, freq, cgram in composes_tiret:
        if ortho.lower() in seen:
            continue
        seen.add(ortho.lower())
        print(f"  {ortho:35s} {phone:25s} freq={freq:>8.1f}  {cgram}")
        count += 1
        if count >= 50:
            break

    print(f"\n{'='*80}")
    print("Top 50 composes avec apostrophe (par freq)")
    print(f"{'='*80}")
    seen = set()
    count = 0
    for ortho, phone, freq, cgram in composes_apo:
        if ortho.lower() in seen:
            continue
        seen.add(ortho.lower())
        print(f"  {ortho:35s} {phone:25s} freq={freq:>8.1f}  {cgram}")
        count += 1
        if count >= 50:
            break

    # 3. Identifier les composes "interessants" pour la fusion P2G
    #    = ceux qu'un STT/tokeniseur pourrait splitter en mots individuels
    print(f"\n{'='*80}")
    print("Composes interessants pour fusion P2G")
    print(f"{'='*80}")
    print("(entrees freq >= 5, ou mots courants)")

    # Composes courants avec tiret
    composes_courants_tiret = set()
    for ortho, phone, freq, cgram in composes_tiret:
        key = ortho.lower()
        if key in composes_courants_tiret:
            continue
        if freq >= 5.0:
            composes_courants_tiret.add(key)
            parts = key.split("-")
            if len(parts) >= 2:
                print(f"  [tiret]  {key:35s} {phone:25s} freq={freq:>8.1f}  parts={parts}")

    # Composes courants avec apostrophe
    composes_courants_apo = set()
    for ortho, phone, freq, cgram in composes_apo:
        key = ortho.lower()
        if key in composes_courants_apo:
            continue
        if freq >= 5.0:
            composes_courants_apo.add(key)
            parts = key.split("'")
            if len(parts) >= 2:
                print(f"  [apo]    {key:35s} {phone:25s} freq={freq:>8.1f}  parts={parts}")

    # 4. Pour les cas cles mentionnes par l'utilisateur
    print(f"\n{'='*80}")
    print("Verification des cas specifiques")
    print(f"{'='*80}")

    cas_cibles = [
        "aujourd'hui", "n'est-ce pas", "est-ce que",
        "états-unis", "peut-être", "c'est-à-dire",
        "lorsque", "parce que", "quelqu'un",
        "jusqu'à", "jusqu'au", "jusqu'en",
        "d'abord", "d'accord", "d'autres",
    ]

    for mot in cas_cibles:
        cursor2 = conn.execute(
            f"SELECT phone, freq, cgram FROM {table} WHERE LOWER(ortho) = ? ORDER BY freq DESC LIMIT 5",
            (mot,),
        )
        rows = cursor2.fetchall()
        if rows:
            for r in rows:
                print(f"  {mot:25s} phone={r['phone']:25s} freq={r['freq']:>8.1f}  {r['cgram']}")
        else:
            print(f"  {mot:25s} NON TROUVE dans le lexique")

    # 5. Compter combien de phones fusionne matcheraient par erreur
    print(f"\n{'='*80}")
    print("Analyse des collisions: phones de 1 lettre combines")
    print(f"{'='*80}")

    # Verifier les phones courts tres courants qui causent des collisions
    phones_courts = ["a", "l", "d", "s", "n", "k", "ɛ", "e", "o", "y", "ɑ̃", "ɔ̃"]
    for p1 in phones_courts[:6]:
        for p2 in phones_courts[:6]:
            fused = p1 + p2
            if lex.exists(fused):
                best = lex.best_ortho(fused)
                freq = lex.best_freq(fused)
                entries = lex.all_entries(fused)
                orthos = set((e.get("ortho") or "").lower() for e in entries[:5])
                has_special = any("'" in o or "-" in o for o in orthos)
                print(f"  {p1}+{p2} = {fused:10s} -> {best:15s} freq={freq:>8.1f}  special={has_special}  {orthos}")

    # 6. Statistiques sur les composes par longueur
    print(f"\n{'='*80}")
    print("Distribution des composes par nb de composants")
    print(f"{'='*80}")

    by_parts_tiret = Counter()
    by_parts_apo = Counter()
    for ortho, phone, freq, cgram in composes_tiret:
        by_parts_tiret[len(ortho.split("-"))] += 1
    for ortho, phone, freq, cgram in composes_apo:
        by_parts_apo[len(ortho.split("'"))] += 1

    print("  Tiret:")
    for k in sorted(by_parts_tiret):
        print(f"    {k} composants: {by_parts_tiret[k]}")
    print("  Apostrophe:")
    for k in sorted(by_parts_apo):
        print(f"    {k} composants: {by_parts_apo[k]}")

    conn.close()


if __name__ == "__main__":
    analyser_composes()
