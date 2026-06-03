#!/usr/bin/env python3
"""Extrait un lexique leger du lexique complet pour le Correcteur.

Source : lexique_lectura.db (schema v5 : tables formes + lemmes)
Sorties :
  - lexique_correcteur.db     (~50 Mo, SQLite avec index)
  - lexique_correcteur.csv.gz (~14 Mo, CSV compresse, optionnel)

Colonnes : ortho, phone, cgram, genre, nombre, freq, lemme, multext, is_lemme, syllabes

Usage :
    python build_csv.py                          # depuis lexique_lectura.db
    python build_csv.py --db path/to/lexique.db  # BDD specifique
    python build_csv.py --output-dir ./out/      # repertoire de sortie
    python build_csv.py --csv                    # generer aussi le CSV gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sqlite3
import sys
from pathlib import Path

# Colonnes
COLUMNS = [
    "ortho", "phone", "cgram", "genre", "nombre",
    "freq", "lemme", "multext", "is_lemme", "syllabes",
]

# Positions genre/nombre dans les tags multext (depend de la categorie)
# Standard Multext-GRACE : A=ADJ est Afpms (5 car), mais 6386 entrees
# ont des tags a 4 car (Afms) ou le degre est omis.
_GENRE_POS = {
    "N": 2,  # Ncms → pos 2
    "A": 3,  # Afpms → pos 3 (5 car standard)
    "D": 3,  # Da-ms → pos 3
    "P": 3,  # Pp3ms → pos 3
}

_NOMBRE_POS = {
    "N": 3,  # Ncms → pos 3
    "A": 4,  # Afpfs → pos 4 (5 car standard)
    "D": 4,  # Da-ms → pos 4
    "P": 4,  # Pp3ms → pos 4
    "V": 5,  # Vmip3s → pos 5
}

_DEGRE_VALUES = frozenset("pcs")  # positif, comparatif, superlatif


def _adj_genre_nombre_pos(multext: str) -> tuple[int, int]:
    """Positions genre/nombre pour un ADJ, selon que le degre est present ou non."""
    # Standard (5 car) : Afpms → degre en pos 2, genre pos 3, nombre pos 4
    # Tronque (4 car)  : Afms  → pas de degre, genre pos 2, nombre pos 3
    if len(multext) >= 3 and multext[2] in _DEGRE_VALUES:
        return 3, 4  # standard
    return 2, 3  # degre omis


def _genre_from_multext(multext: str) -> str:
    """Extrait le genre (m/f) depuis un tag multext."""
    if not multext:
        return ""
    cat = multext[0]
    if cat == "A":
        g_pos, _ = _adj_genre_nombre_pos(multext)
        if len(multext) > g_pos:
            c = multext[g_pos]
            if c in ("m", "f"):
                return c
        return ""
    pos = _GENRE_POS.get(cat)
    if pos is not None and len(multext) > pos:
        c = multext[pos]
        if c in ("m", "f"):
            return c
    return ""


def _nombre_from_multext(multext: str) -> str:
    """Extrait le nombre (s/p) depuis un tag multext."""
    if not multext:
        return ""
    cat = multext[0]
    if cat == "A":
        _, n_pos = _adj_genre_nombre_pos(multext)
        if len(multext) > n_pos:
            c = multext[n_pos]
            if c in ("s", "p"):
                return c
        return ""
    pos = _NOMBRE_POS.get(cat)
    if pos is not None and len(multext) > pos:
        c = multext[pos]
        if c in ("s", "p"):
            return c
    return ""


def _iter_rows(db_path: Path):
    """Itere sur les lignes du lexique source (JOIN formes + lemmes)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "lemmes" not in tables:
        print("ERREUR : table 'lemmes' absente — schema v3 non supporte",
              file=sys.stderr)
        sys.exit(1)

    # Detecter la presence de la colonne syllabes
    cols_formes = {
        r[1] for r in conn.execute("PRAGMA table_info(formes)")
    }
    has_syllabes = "syllabes" in cols_formes
    syl_col = ", f.syllabes" if has_syllabes else ""

    query = f"""
        SELECT
            f.ortho,
            f.phone,
            l.cgram,
            l.genre,
            f.multext,
            f.freq_composite,
            f.freq_opensubs,
            l.lemme
            {syl_col}
        FROM formes f
        JOIN lemmes l ON f.lemme_id = l.id
        ORDER BY lower(f.ortho), f.freq_composite DESC
    """

    for row in conn.execute(query):
        ortho = row["ortho"] or ""
        phone = row["phone"] or ""
        cgram = row["cgram"] or ""
        multext = row["multext"] or ""
        genre = _genre_from_multext(multext) or row["genre"] or ""
        lemme = row["lemme"] or ""
        syllabes = (row["syllabes"] or "") if has_syllabes else ""

        freq_raw = row["freq_composite"]
        if freq_raw is None or freq_raw == 0:
            freq_raw = row["freq_opensubs"]
        freq = round(float(freq_raw or 0), 2)

        nombre = _nombre_from_multext(multext)
        is_lemme = 1 if ortho.lower() == lemme.lower() else 0

        yield (ortho, phone, cgram, genre, nombre, freq, lemme, multext, is_lemme, syllabes)

    conn.close()


def build_sqlite(db_path: Path, output_path: Path) -> int:
    """Genere le SQLite leger depuis la BDD source."""
    if not db_path.exists():
        print(f"ERREUR : base introuvable : {db_path}", file=sys.stderr)
        sys.exit(1)

    # Supprimer l'ancien fichier si present
    if output_path.exists():
        output_path.unlink()

    print(f"Lecture de {db_path} ...")

    out = sqlite3.connect(str(output_path))
    out.execute("PRAGMA journal_mode=WAL")
    out.execute("PRAGMA synchronous=OFF")

    out.execute("""
        CREATE TABLE lexique (
            ortho     TEXT NOT NULL,
            phone     TEXT DEFAULT '',
            cgram     TEXT DEFAULT '',
            genre     TEXT DEFAULT '',
            nombre    TEXT DEFAULT '',
            freq      REAL DEFAULT 0,
            lemme     TEXT DEFAULT '',
            multext   TEXT DEFAULT '',
            is_lemme  INTEGER DEFAULT 0,
            syllabes  TEXT DEFAULT ''
        )
    """)

    row_count = 0
    batch: list[tuple] = []
    BATCH_SIZE = 50_000

    for row in _iter_rows(db_path):
        batch.append(row)
        row_count += 1
        if len(batch) >= BATCH_SIZE:
            out.executemany(
                "INSERT INTO lexique VALUES (?,?,?,?,?,?,?,?,?,?)", batch
            )
            batch.clear()

    if batch:
        out.executemany(
            "INSERT INTO lexique VALUES (?,?,?,?,?,?,?,?,?,?)", batch
        )

    # Index
    print("Creation des index ...")
    out.execute("CREATE INDEX idx_ortho ON lexique(ortho COLLATE NOCASE)")
    out.execute("CREATE INDEX idx_phone ON lexique(phone)")
    out.execute("CREATE INDEX idx_lemme ON lexique(lemme COLLATE NOCASE)")

    out.commit()
    out.execute("PRAGMA journal_mode=DELETE")
    out.execute("VACUUM")
    out.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Ecrit {row_count:,} lignes dans {output_path} ({size_mb:.1f} Mo)")
    return row_count


def build_csv(db_path: Path, output_path: Path) -> int:
    """Genere le CSV compresse depuis la BDD source."""
    if not db_path.exists():
        print(f"ERREUR : base introuvable : {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Lecture de {db_path} (CSV) ...")

    row_count = 0
    with gzip.open(str(output_path), "wt", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(COLUMNS)

        for row in _iter_rows(db_path):
            writer.writerow(row)
            row_count += 1

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Ecrit {row_count:,} lignes dans {output_path} ({size_mb:.1f} Mo)")
    return row_count


def main():
    parser = argparse.ArgumentParser(
        description="Extrait un lexique leger pour le Correcteur"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "lexique_lectura.db",
        help="Chemin vers la BDD SQLite source (defaut: ../lexique_lectura.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repertoire de sortie (defaut: .)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Generer aussi le CSV gz (en plus du SQLite)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # SQLite (toujours)
    build_sqlite(args.db, args.output_dir / "lexique_correcteur.db")

    # CSV (optionnel)
    if args.csv:
        build_csv(args.db, args.output_dir / "lexique_correcteur.csv.gz")


if __name__ == "__main__":
    main()
