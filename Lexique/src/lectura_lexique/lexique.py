"""Classe concrete Lexique avec chargement lazy a 3 niveaux.

- **Niveau 1** (init) : ``frozenset`` de toutes les formes -> ``existe()`` O(1)
- **Niveau 2** (lazy) : index ``phone -> [entrees]`` -> ``homophones()``
- **Niveau 3** (lazy) : index ``ortho -> [entrees]`` -> ``info()``, ``frequence()``, ``phone_de()``

Pour SQLite, les niveaux 2-3 utilisent des requetes SQL directes (pas de RAM).
Pour CSV/TSV, un seul parcours du fichier construit les deux index a la demande.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from lectura_lexique._aliases import resoudre_colonnes
from lectura_lexique._loaders import iter_csv, iter_tsv
from lectura_lexique._types import EntreeLexicale
from lectura_lexique._utils import normaliser_ortho


class Lexique:
    """Interface de requetage du lexique Lectura.

    Supports CSV, TSV et SQLite. Detection automatique par extension.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        table: str = "formes",
        precharger: bool = False,
    ) -> None:
        """Initialise le lexique a partir d'un fichier.

        Args:
            source: Chemin vers un fichier .db/.sqlite, .tsv, ou .csv
            table: Nom de la table (pour SQLite uniquement)
            precharger: Si True, charge tous les index immediatement
        """
        self._source = Path(source)
        self._table = table
        self._backend = self._detecter_backend()

        # Niveau 1 : set de formes (toujours charge)
        self._formes: frozenset[str] = frozenset()

        # Niveau 2 : index phone -> entrees (lazy)
        self._index_phone: dict[str, list[dict[str, Any]]] | None = None

        # Niveau 3 : index ortho -> entrees (lazy)
        self._index_ortho: dict[str, list[dict[str, Any]]] | None = None

        # SQLite : connexion lazy
        self._conn: sqlite3.Connection | None = None
        self._col_mapping: dict[str, str] | None = None

        # Charger le niveau 1
        self._charger_formes()

        if precharger:
            self._charger_index()

    def _detecter_backend(self) -> str:
        """Detecte le backend par extension de fichier."""
        suffix = self._source.suffix.lower()
        if suffix in (".db", ".sqlite", ".sqlite3"):
            return "sqlite"
        if suffix == ".tsv":
            return "tsv"
        return "csv"

    def _charger_formes(self) -> None:
        """Niveau 1 : charge le set de toutes les formes."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            cur = conn.execute(f"SELECT DISTINCT lower(ortho) FROM {self._table}")  # noqa: S608
            self._formes = frozenset(row[0] for row in cur)
        else:
            formes: set[str] = set()
            for entree in self._iter_source():
                ortho = entree.get("ortho", "")
                if ortho:
                    formes.add(normaliser_ortho(ortho))
            self._formes = frozenset(formes)

    def _iter_source(self):
        """Itere sur la source CSV/TSV."""
        if self._backend == "tsv":
            yield from iter_tsv(self._source)
        else:
            yield from iter_csv(self._source)

    def _get_conn(self) -> sqlite3.Connection:
        """Ouvre la connexion SQLite a la demande."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._source))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _charger_index(self) -> None:
        """Niveaux 2-3 : construit les index phone et ortho (CSV/TSV)."""
        if self._backend == "sqlite":
            return  # Les index SQLite sont geres via requetes
        if self._index_phone is not None:
            return  # Deja charge

        index_phone: dict[str, list[dict[str, Any]]] = {}
        index_ortho: dict[str, list[dict[str, Any]]] = {}

        for entree in self._iter_source():
            ortho = entree.get("ortho", "")
            if not ortho:
                continue
            phone = entree.get("phone", "")
            freq = entree.get("freq", 0.0)
            try:
                freq = float(freq)
            except (ValueError, TypeError):
                freq = 0.0

            entry_dict: dict[str, Any] = dict(entree)
            entry_dict["freq"] = freq

            # Index ortho
            key_ortho = normaliser_ortho(ortho)
            if key_ortho not in index_ortho:
                index_ortho[key_ortho] = []
            index_ortho[key_ortho].append(entry_dict)

            # Index phone
            if phone:
                if phone not in index_phone:
                    index_phone[phone] = []
                index_phone[phone].append(entry_dict)

        self._index_phone = index_phone
        self._index_ortho = index_ortho

    # --- API publique (LexiqueProtocol) ---

    def existe(self, mot: str) -> bool:
        """Test d'appartenance O(1) via le set en memoire."""
        return normaliser_ortho(mot) in self._formes

    def homophones(self, phone: str) -> list[dict[str, Any]]:
        """Retourne tous les mots ayant cette prononciation.

        Chaque dict contient au minimum : ortho, cgram, freq.
        """
        if self._backend == "sqlite":
            conn = self._get_conn()
            cur = conn.execute(
                f"SELECT ortho, cgram, freq_opensubs AS freq, phone "
                f"FROM {self._table} WHERE phone = ?",  # noqa: S608
                (phone,),
            )
            return [dict(row) for row in cur.fetchall()]

        # CSV/TSV : charger les index si necessaire
        if self._index_phone is None:
            self._charger_index()
        assert self._index_phone is not None
        return self._index_phone.get(phone, [])

    def info(self, mot: str) -> list[dict[str, Any]]:
        """Retourne les entrees lexicales completes pour un mot."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            cur = conn.execute(
                f"SELECT * FROM {self._table} WHERE lower(ortho) = ?",  # noqa: S608
                (normaliser_ortho(mot),),
            )
            colonnes = [desc[0] for desc in cur.description]
            mapping = resoudre_colonnes(colonnes)
            results = []
            for row in cur.fetchall():
                entry: dict[str, Any] = {}
                for col, val in zip(colonnes, row):
                    canon = mapping.get(col, col)
                    entry[canon] = val
                results.append(entry)
            return results

        # CSV/TSV
        if self._index_ortho is None:
            self._charger_index()
        assert self._index_ortho is not None
        return self._index_ortho.get(normaliser_ortho(mot), [])

    def frequence(self, mot: str) -> float:
        """Frequence max parmi toutes les entrees de ce mot."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            cur = conn.execute(
                f"SELECT MAX(freq_opensubs) FROM {self._table} "  # noqa: S608
                f"WHERE lower(ortho) = ?",
                (normaliser_ortho(mot),),
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0

        # CSV/TSV
        if self._index_ortho is None:
            self._charger_index()
        assert self._index_ortho is not None
        entries = self._index_ortho.get(normaliser_ortho(mot), [])
        if not entries:
            return 0.0
        return max(float(e.get("freq", 0.0)) for e in entries)

    def phone_de(self, mot: str) -> str | None:
        """Retourne la prononciation la plus frequente d'un mot."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            cur = conn.execute(
                f"SELECT phone FROM {self._table} "  # noqa: S608
                f"WHERE lower(ortho) = ? AND phone != '' "
                f"ORDER BY freq_opensubs DESC LIMIT 1",
                (normaliser_ortho(mot),),
            )
            row = cur.fetchone()
            return row[0] if row else None

        # CSV/TSV
        if self._index_ortho is None:
            self._charger_index()
        assert self._index_ortho is not None
        entries = self._index_ortho.get(normaliser_ortho(mot), [])
        phones = [
            (e.get("phone", ""), float(e.get("freq", 0.0)))
            for e in entries
            if e.get("phone")
        ]
        if not phones:
            return None
        phones.sort(key=lambda x: -x[1])
        return phones[0][0]

    # --- Context manager & lifecycle ---

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs) -> Lexique:
        """Auto-decouvre un fichier lexique dans le repertoire.

        Cherche dans l'ordre : .db, .sqlite, .sqlite3, .tsv, .csv.
        """
        d = Path(directory)
        for pattern in ("*.db", "*.sqlite", "*.sqlite3", "*.tsv", "*.csv"):
            matches = sorted(d.glob(pattern))
            if matches:
                return cls(matches[0], **kwargs)
        raise FileNotFoundError(
            f"Aucun fichier lexique trouve dans {directory}"
        )

    def close(self) -> None:
        """Ferme la connexion SQLite si ouverte."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Lexique:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
