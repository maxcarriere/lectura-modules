#!/usr/bin/env python3
"""Construit une version light de phone_lexicon.db (freq > 0 seulement).

La version complete fait ~76 Mo (1.25M lignes, dont ~82% avec freq=0).
La version light ne garde que les lignes freq > 0 (~227k lignes, ~14 Mo),
suffisante pour la publication publique (PyPI/GitHub).

Usage :
    python build_phone_lexicon_light.py [--input PATH] [--output PATH]

Par defaut :
    input  = ../src/lectura_graphemiseur/modeles/phone_lexicon.db
    output = phone_lexicon_light.db
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def build_light(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"ERREUR : {input_path} introuvable")
        sys.exit(1)

    if output_path.exists():
        output_path.unlink()

    # Stats source
    conn_src = sqlite3.connect(str(input_path))
    total_rows = conn_src.execute("SELECT COUNT(*) FROM phones").fetchone()[0]
    total_phones = conn_src.execute("SELECT COUNT(DISTINCT phone) FROM phones").fetchone()[0]
    src_size = input_path.stat().st_size

    # Creer la DB light
    conn_dst = sqlite3.connect(str(output_path))
    conn_dst.execute("""
        CREATE TABLE phones (
            phone TEXT NOT NULL,
            ortho TEXT NOT NULL,
            cgram TEXT DEFAULT '',
            genre TEXT DEFAULT '',
            nombre TEXT DEFAULT '',
            freq REAL DEFAULT 0
        )
    """)

    # Copier les lignes freq > 0
    cursor = conn_src.execute(
        "SELECT phone, ortho, cgram, genre, nombre, freq "
        "FROM phones WHERE freq > 0"
    )
    batch = []
    copied = 0
    for row in cursor:
        batch.append(row)
        if len(batch) >= 10000:
            conn_dst.executemany(
                "INSERT INTO phones (phone, ortho, cgram, genre, nombre, freq) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
            copied += len(batch)
            batch.clear()
    if batch:
        conn_dst.executemany(
            "INSERT INTO phones (phone, ortho, cgram, genre, nombre, freq) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )
        copied += len(batch)

    # Index pour les lookups par phone
    conn_dst.execute("CREATE INDEX idx_phones_phone ON phones (phone)")
    conn_dst.commit()

    # Stats destination
    light_phones = conn_dst.execute("SELECT COUNT(DISTINCT phone) FROM phones").fetchone()[0]
    conn_dst.close()
    conn_src.close()

    dst_size = output_path.stat().st_size

    # Afficher le resume
    print(f"phone_lexicon.db light construite avec succes")
    print()
    print(f"  Source     : {input_path}")
    print(f"  Sortie     : {output_path}")
    print()
    print(f"  Lignes     : {total_rows:>10,} → {copied:>10,}  ({copied/total_rows*100:.1f}%)")
    print(f"  Phones     : {total_phones:>10,} → {light_phones:>10,}  ({light_phones/total_phones*100:.1f}%)")
    print(f"  Taille     : {src_size/1024/1024:>10.1f} Mo → {dst_size/1024/1024:>10.1f} Mo  ({dst_size/src_size*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Construit phone_lexicon_light.db (freq > 0 seulement)"
    )
    default_input = (
        Path(__file__).resolve().parent.parent
        / "src" / "lectura_graphemiseur" / "modeles" / "phone_lexicon.db"
    )
    parser.add_argument(
        "--input", type=Path, default=default_input,
        help="Chemin vers phone_lexicon.db source",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("phone_lexicon_light.db"),
        help="Chemin de sortie (defaut: phone_lexicon_light.db)",
    )
    args = parser.parse_args()
    build_light(args.input, args.output)


if __name__ == "__main__":
    main()
