#!/usr/bin/env python3
"""Convertit le lexique CSV en base SQLite optimisee.

Usage :
    python construire_bdd.py [chemin_csv] [chemin_db]

Par defaut :
    chemin_csv = ../lexique_lectura.csv
    chemin_db  = ../lexique_lectura.db
"""

from __future__ import annotations

import csv
import sqlite3
import sys
import time
from pathlib import Path

# Ajouter le parent pour importer lectura_lexique
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _SCRIPT_DIR.parent / "src"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from lectura_lexique._utils import _tokenize_ipa, reverse_phone_ipa

# --- Schema ---

SCHEMA_FORMES = """
CREATE TABLE IF NOT EXISTS formes (
    id INTEGER PRIMARY KEY,
    ortho TEXT NOT NULL,
    lemme TEXT,
    cgram TEXT,
    multext TEXT,
    genre TEXT,
    nombre TEXT,
    mode TEXT,
    temps TEXT,
    personne TEXT,
    phone TEXT,
    phone_reversed TEXT,
    variantes TEXT,
    nb_syllabes INTEGER,
    syllabes TEXT,
    freq_frantext REAL,
    freq_lm10 REAL,
    freq_frwac REAL,
    freq_opensubs REAL,
    source TEXT,
    synonymes TEXT,
    antonymes TEXT,
    domaine TEXT,
    definition TEXT,
    registre TEXT,
    etymologie TEXT,
    exemple TEXT
)
"""

SCHEMA_RELATIONS = """
CREATE TABLE IF NOT EXISTS relations (
    lemme TEXT NOT NULL,
    type TEXT NOT NULL,
    cible TEXT NOT NULL
)
"""

SCHEMA_NOMS_PROPRES = """
CREATE TABLE IF NOT EXISTS noms_propres (
    id INTEGER PRIMARY KEY,
    lemme TEXT NOT NULL,
    cgram TEXT,
    sous_type TEXT,
    definition TEXT,
    etymologie TEXT,
    phone TEXT
)
"""

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_ortho ON formes(ortho COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_lemme ON formes(lemme COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_phone ON formes(phone)",
    "CREATE INDEX IF NOT EXISTS idx_cgram ON formes(cgram)",
    "CREATE INDEX IF NOT EXISTS idx_nb_syllabes ON formes(nb_syllabes)",
    "CREATE INDEX IF NOT EXISTS idx_phone_rev ON formes(phone_reversed)",
    "CREATE INDEX IF NOT EXISTS idx_rel_lemme ON relations(lemme, type)",
    "CREATE INDEX IF NOT EXISTS idx_rel_cible ON relations(cible, type)",
    "CREATE INDEX IF NOT EXISTS idx_np_lemme ON noms_propres(lemme COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_np_sous_type ON noms_propres(sous_type)",
    "CREATE INDEX IF NOT EXISTS idx_np_phone ON noms_propres(phone)",
]

# Colonnes du CSV dans l'ordre attendu
CSV_COLUMNS = [
    "ortho", "lemme", "cgram", "multext", "genre", "nombre",
    "mode", "temps", "personne", "phone", "variantes",
    "nb_syllabes", "syllabes", "freq_frantext", "freq_lm10",
    "freq_frwac", "freq_opensubs", "source", "synonymes",
    "antonymes", "domaine", "definition", "registre",
    "etymologie", "exemple",
]

# Colonnes numeriques
FLOAT_COLS = {"freq_frantext", "freq_lm10", "freq_frwac", "freq_opensubs"}
INT_COLS = {"nb_syllabes"}

BATCH_SIZE = 10_000


def _parse_value(col: str, val: str):
    """Convertit une valeur CSV en type Python appropriate."""
    if not val:
        return None
    if col in FLOAT_COLS:
        try:
            return float(val)
        except ValueError:
            return None
    if col in INT_COLS:
        try:
            return int(val)
        except ValueError:
            return None
    return val


def _parse_relations(lemme: str, raw: str, rel_type: str) -> list[tuple[str, str, str]]:
    """Parse une colonne synonymes/antonymes en tuples (lemme, type, cible)."""
    if not raw:
        return []
    return [
        (lemme, rel_type, cible.strip())
        for cible in raw.split(";")
        if cible.strip()
    ]


NP_COLUMNS = ["lemme", "cgram", "sous_type", "definition", "etymologie"]


def ingerer_noms_propres(conn: sqlite3.Connection, np_csv_path: Path) -> int:
    """Ingere le fichier noms_propres.csv dans la table noms_propres.

    Retourne le nombre de lignes inserees.
    """
    if not np_csv_path.exists():
        print(f"  (fichier noms propres introuvable : {np_csv_path})")
        return 0

    conn.execute(SCHEMA_NOMS_PROPRES)
    conn.commit()

    insert_sql = (
        "INSERT INTO noms_propres (lemme, cgram, sous_type, definition, etymologie) "
        "VALUES (?, ?, ?, ?, ?)"
    )

    nb = 0
    batch: list[tuple] = []
    with open(np_csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = tuple(row.get(col, "") or None for col in NP_COLUMNS)
            batch.append(values)
            nb += 1
            if len(batch) >= BATCH_SIZE:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()
                if nb % 100_000 == 0:
                    print(f"  {nb:>10,} noms propres traites...")

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()

    return nb


def construire_bdd(csv_path: Path, db_path: Path, np_csv_path: Path | None = None) -> None:
    """Convertit un CSV lexique en base SQLite."""
    print(f"Source CSV : {csv_path}")
    print(f"Sortie DB  : {db_path}")
    if np_csv_path:
        print(f"Noms propres : {np_csv_path}")
    print()

    if db_path.exists():
        db_path.unlink()
        print("(ancien fichier .db supprime)")

    t0 = time.time()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")

    # Creer les tables
    conn.execute(SCHEMA_FORMES)
    conn.execute(SCHEMA_RELATIONS)
    conn.execute(SCHEMA_NOMS_PROPRES)
    conn.commit()

    # Colonnes pour INSERT
    insert_cols = [
        "ortho", "lemme", "cgram", "multext", "genre", "nombre",
        "mode", "temps", "personne", "phone", "phone_reversed",
        "variantes", "nb_syllabes", "syllabes", "freq_frantext",
        "freq_lm10", "freq_frwac", "freq_opensubs", "source",
        "synonymes", "antonymes", "domaine", "definition",
        "registre", "etymologie", "exemple",
    ]
    placeholders = ",".join("?" for _ in insert_cols)
    insert_sql = f"INSERT INTO formes ({','.join(insert_cols)}) VALUES ({placeholders})"
    insert_rel_sql = "INSERT INTO relations (lemme, type, cible) VALUES (?, ?, ?)"

    nb_lignes = 0
    nb_relations = 0
    formes_batch: list[tuple] = []
    relations_batch: list[tuple[str, str, str]] = []

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERREUR : CSV vide ou sans header")
            conn.close()
            return

        print(f"Colonnes CSV : {len(reader.fieldnames)}")
        print(f"Colonnes attendues : {len(CSV_COLUMNS)}")
        print()

        for row in reader:
            nb_lignes += 1

            # Construire le tuple de valeurs
            phone = row.get("phone", "")
            phone_rev = reverse_phone_ipa(phone) if phone else None

            values = []
            for col in insert_cols:
                if col == "phone_reversed":
                    values.append(phone_rev)
                else:
                    values.append(_parse_value(col, row.get(col, "")))

            formes_batch.append(tuple(values))

            # Relations
            lemme = row.get("lemme", "")
            if lemme:
                relations_batch.extend(
                    _parse_relations(lemme, row.get("synonymes", ""), "synonyme")
                )
                relations_batch.extend(
                    _parse_relations(lemme, row.get("antonymes", ""), "antonyme")
                )

            # Flush par batch
            if len(formes_batch) >= BATCH_SIZE:
                conn.executemany(insert_sql, formes_batch)
                if relations_batch:
                    conn.executemany(insert_rel_sql, relations_batch)
                    nb_relations += len(relations_batch)
                conn.commit()
                formes_batch.clear()
                relations_batch.clear()
                if nb_lignes % 100_000 == 0:
                    print(f"  {nb_lignes:>10,} lignes traitees...")

    # Flush restant
    if formes_batch:
        conn.executemany(insert_sql, formes_batch)
    if relations_batch:
        conn.executemany(insert_rel_sql, relations_batch)
        nb_relations += len(relations_batch)
    conn.commit()

    print(f"  {nb_lignes:>10,} lignes inserees au total")
    print(f"  {nb_relations:>10,} relations (synonymes/antonymes)")
    print()

    # Noms propres
    if np_csv_path:
        print("Ingestion des noms propres...")
        nb_np = ingerer_noms_propres(conn, np_csv_path)
        print(f"  {nb_np:>10,} noms propres inseres")
        print()

    # Index
    print("Creation des index...")
    for stmt in INDEX_STATEMENTS:
        conn.execute(stmt)
    conn.commit()

    # VACUUM
    print("VACUUM...")
    conn.execute("VACUUM")
    conn.commit()

    conn.close()

    t1 = time.time()
    taille = db_path.stat().st_size / (1024 * 1024)

    print()
    print(f"Termine en {t1 - t0:.1f}s")
    print(f"Fichier : {db_path} ({taille:.1f} Mo)")
    print(f"Lignes  : {nb_lignes:,}")


def main() -> None:
    default_csv = _SCRIPT_DIR.parent / "lexique_lectura.csv"
    default_db = _SCRIPT_DIR.parent / "lexique_lectura.db"
    default_np = _SCRIPT_DIR.parent.parent.parent / "Lexique" / "noms_propres.csv"

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv
    db_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_db
    np_csv_path = Path(sys.argv[3]) if len(sys.argv) > 3 else default_np

    if not csv_path.exists():
        print(f"ERREUR : fichier introuvable : {csv_path}")
        sys.exit(1)

    construire_bdd(csv_path, db_path, np_csv_path if np_csv_path.exists() else None)


if __name__ == "__main__":
    main()
