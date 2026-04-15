"""Backends de chargement pour differents formats de lexique.

Chaque loader est un iterateur qui produit des ``EntreeLexicale`` avec
des noms de champs canoniques (via ``_aliases.resoudre_colonnes``).
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from lectura_lexique._aliases import FREQ_PRIORITE, resoudre_colonnes
from lectura_lexique._types import EntreeLexicale


def _normaliser_entree(row: dict[str, str], mapping: dict[str, str]) -> EntreeLexicale:
    """Applique le mapping d'alias et resout la frequence prioritaire."""
    entree: dict[str, Any] = {}
    freq_trouvee = False

    for col_source, valeur in row.items():
        col_canon = mapping.get(col_source, col_source.lower().strip())
        if col_canon == "freq" and not freq_trouvee:
            # C'est un alias de frequence -> stocker comme "freq"
            try:
                entree["freq"] = float(valeur) if valeur else 0.0
            except (ValueError, TypeError):
                entree["freq"] = 0.0
            freq_trouvee = True
        elif col_canon in (
            "freq_opensubs", "freqfilms2", "freq_frwac_forme_pmw",
            "freq_frwac", "freq_lm10", "freq_frantext",
        ):
            # Stocker la colonne brute et aussi comme "freq" si prioritaire
            try:
                val = float(valeur) if valeur else 0.0
            except (ValueError, TypeError):
                val = 0.0
            entree[col_canon] = val
        else:
            entree[col_canon] = valeur

    # Si pas de "freq" explicite, choisir la meilleure frequence disponible
    if "freq" not in entree:
        for col_freq in FREQ_PRIORITE:
            if col_freq in entree:
                entree["freq"] = entree[col_freq]
                break
        else:
            entree["freq"] = 0.0

    return entree  # type: ignore[return-value]


def iter_csv(
    path: str | Path,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
) -> Iterator[EntreeLexicale]:
    """Itere sur un fichier CSV/TSV et produit des EntreeLexicale."""
    path = Path(path)
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return
        mapping = resoudre_colonnes(list(reader.fieldnames))
        for row in reader:
            yield _normaliser_entree(row, mapping)


def iter_tsv(path: str | Path) -> Iterator[EntreeLexicale]:
    """Raccourci pour ``iter_csv`` avec delimiter tabulation."""
    yield from iter_csv(path, delimiter="\t")


def iter_sqlite(
    path: str | Path,
    table: str = "formes",
) -> Iterator[EntreeLexicale]:
    """Itere sur une table SQLite et produit des EntreeLexicale."""
    path = Path(path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(f"SELECT * FROM {table}")  # noqa: S608
        colonnes = [desc[0] for desc in cur.description]
        mapping = resoudre_colonnes(colonnes)
        for row in cur:
            yield _normaliser_entree(dict(row), mapping)
    finally:
        conn.close()
