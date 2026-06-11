#!/usr/bin/env python3
"""Extrait un dictionnaire d'entites notables depuis le lexique V6.

Script one-shot qui produit ``entites.json`` embarquable dans lectura-p2g.

Usage::

    python scripts/extraire_entites.py [--db PATH] [--out PATH] [--seuil N]

Par defaut :
    --db   : /data/work/projets/lectura/workspace/Lexique/bases/lexique_lectura_v6.db
    --out  : src/lectura_p2g/data/entites.json
    --seuil: 50
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import unicodedata
from pathlib import Path

DEFAULT_DB = "/data/work/projets/lectura/workspace/Lexique/bases/lexique_lectura_v6.db"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "src" / "lectura_p2g" / "data" / "entites.json"
DEFAULT_SEUIL = 50


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def extraire(db_path: str, seuil: int) -> list[dict]:
    """Extrait les entites de type 'personne' avec notoriete >= seuil.

    Pour chaque entite, reconstruit l'IPA concatene a partir des phones
    des composants (lemmes NOM PROPRE) tries par position dans le label.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Etape 1 : recuperer toutes les entites eligibles
    cur.execute(
        """
        SELECT e.id, e.label, CAST(p.valeur AS INTEGER) AS notoriete
        FROM entites e
        JOIN entite_proprietes p ON p.entite_id = e.id AND p.cle = 'notoriete'
        WHERE CAST(p.valeur AS INTEGER) >= ?
          AND e.type_entite = 'personne'
        ORDER BY CAST(p.valeur AS INTEGER) DESC
        """,
        (seuil,),
    )
    entites_raw = cur.fetchall()

    # Etape 2 : pour chaque entite, recuperer les phones des composants
    # tries par position (important pour l'IPA concatene)
    query_phones = """
        SELECT el.position, l.lemme, f.phone
        FROM entite_lemmes el
        JOIN lemmes l ON l.id = el.lemme_id AND l.cgram = 'NOM PROPRE'
        JOIN formes f ON f.lemme_id = l.id AND f.phone IS NOT NULL AND f.phone != ''
        WHERE el.entite_id = ?
        ORDER BY el.position, f.id
    """

    entites = []
    for row in entites_raw:
        eid = row["id"]
        label = row["label"]
        notoriete = row["notoriete"]

        cur.execute(query_phones, (eid,))
        composants = cur.fetchall()

        if not composants:
            # Pas de phones -> on ne peut pas matcher en IPA, on skip
            continue

        # Grouper par position : prendre le premier phone par position
        # position peut etre None -> utiliser un compteur comme fallback
        phones_by_pos: dict[int, str] = {}
        fallback_pos = 0
        for comp in composants:
            pos = comp["position"]
            if pos is None:
                pos = fallback_pos
            if pos not in phones_by_pos:
                phones_by_pos[pos] = comp["phone"]
            fallback_pos = pos + 1

        # Trier par position et concatener
        sorted_phones = [phones_by_pos[p] for p in sorted(phones_by_pos)]
        ipa = _nfc("".join(sorted_phones))

        if not ipa:
            continue

        nb_mots = len(label.split())

        entites.append({
            "label": label,
            "ipa": ipa,
            "notoriete": notoriete,
            "nb_mots": nb_mots,
        })

    conn.close()
    return entites


def construire_index_bigrammes(entites: list[dict]) -> dict[str, list[int]]:
    """Construit un index de pre-filtrage par bigrammes IPA.

    Pour chaque bigramme (2 caracteres consecutifs) present dans l'IPA
    d'une entite, on stocke l'index de l'entite dans la liste.
    """
    index: dict[str, list[int]] = {}
    for idx, ent in enumerate(entites):
        ipa = ent["ipa"]
        seen: set[str] = set()
        for i in range(len(ipa) - 1):
            bigram = ipa[i : i + 2]
            if bigram not in seen:
                seen.add(bigram)
                index.setdefault(bigram, []).append(idx)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrait le dictionnaire d'entites notables.")
    parser.add_argument("--db", default=DEFAULT_DB, help="Chemin vers lexique_lectura_v6.db")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Chemin de sortie (JSON)")
    parser.add_argument("--seuil", type=int, default=DEFAULT_SEUIL, help="Seuil de notoriete minimum")
    args = parser.parse_args()

    print(f"Extraction depuis : {args.db}")
    print(f"Seuil notoriete   : {args.seuil}")

    entites = extraire(args.db, args.seuil)
    print(f"Entites extraites : {len(entites)}")

    index_ipa = construire_index_bigrammes(entites)
    print(f"Bigrammes indexes : {len(index_ipa)}")

    data = {
        "version": 1,
        "seuil_notoriete": args.seuil,
        "entites": entites,
        "index_ipa": index_ipa,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    size_mo = out_path.stat().st_size / (1024 * 1024)
    print(f"Fichier genere    : {out_path}")
    print(f"Taille            : {size_mo:.2f} Mo")


if __name__ == "__main__":
    main()
