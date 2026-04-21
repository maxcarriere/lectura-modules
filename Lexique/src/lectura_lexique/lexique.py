"""Classe concrete Lexique avec chargement lazy a 4 niveaux.

- **Niveau 1** (init) : ``frozenset`` de toutes les formes -> ``existe()`` O(1)
- **Niveau 2** (lazy) : index ``phone -> [entrees]`` -> ``homophones()``
- **Niveau 3** (lazy) : index ``ortho -> [entrees]`` -> ``info()``, ``frequence()``, ``phone_de()``
- **Niveau 4** (lazy) : index ``lemme -> [entrees]`` -> ``conjuguer()``, ``formes_de()``

Pour SQLite, les niveaux 2-4 utilisent des requetes SQL directes (pas de RAM).
Pour CSV/TSV, un seul parcours du fichier construit les trois index a la demande.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Callable

from lectura_lexique._aliases import resoudre_colonnes
from lectura_lexique._loaders import iter_csv, iter_tsv
from lectura_lexique._multext import decoder_multext as _decoder_multext
from lectura_lexique._multext import filtre_multext as _filtre_multext
from lectura_lexique._types import Concept, EntreeForme, EntreeLemme, EntreeLexicale
from lectura_lexique._utils import normaliser_ortho

from lectura_lexique import _morphologie
from lectura_lexique import _phonetique
from lectura_lexique import _semantique
from lectura_lexique import _recherche


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

        # Niveau 4 : index lemme -> entrees (lazy)
        self._index_lemme: dict[str, list[dict[str, Any]]] | None = None

        # SQLite : connexion lazy
        self._conn: sqlite3.Connection | None = None
        self._col_mapping: dict[str, str] | None = None

        # Schema version (3 or 4) — detected at first SQLite access
        self._schema_version: int = 3

        # Detecter la version du schema AVANT de charger les formes
        if self._backend == "sqlite":
            self._detect_schema_version()

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
            # Creer les index NOCASE si absents
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table}_ortho_nocase "  # noqa: S608
                f"ON {self._table}(ortho COLLATE NOCASE)"
            )
            if self._schema_version < 4:
                # v3 : lemme est une colonne de formes
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_lemme_nocase "  # noqa: S608
                    f"ON {self._table}(lemme COLLATE NOCASE)"
                )
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
            self._conn = sqlite3.connect(
                str(self._source), check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _detect_schema_version(self) -> None:
        """Detecte si la BDD utilise le schema v4 (table lemmes presente)."""
        conn = self._get_conn()
        try:
            conn.execute("SELECT 1 FROM lemmes LIMIT 1")
            self._schema_version = 4
        except sqlite3.OperationalError:
            self._schema_version = 3

    @property
    def schema_version(self) -> int:
        """Version du schema BDD (3 ou 4)."""
        return self._schema_version

    def _charger_index(self) -> None:
        """Niveaux 2-4 : construit les index phone, ortho et lemme (CSV/TSV)."""
        if self._backend == "sqlite":
            return  # Les index SQLite sont geres via requetes
        if self._index_phone is not None:
            return  # Deja charge

        index_phone: dict[str, list[dict[str, Any]]] = {}
        index_ortho: dict[str, list[dict[str, Any]]] = {}
        index_lemme: dict[str, list[dict[str, Any]]] = {}

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

            # Index lemme
            lemme = entree.get("lemme", "")
            if lemme:
                key_lemme = normaliser_ortho(lemme)
                if key_lemme not in index_lemme:
                    index_lemme[key_lemme] = []
                index_lemme[key_lemme].append(entry_dict)

        self._index_phone = index_phone
        self._index_ortho = index_ortho
        self._index_lemme = index_lemme

    def _get_index_ortho(self) -> dict[str, list[dict[str, Any]]]:
        """Retourne l'index ortho, le chargeant si necessaire."""
        if self._index_ortho is None:
            self._charger_index()
        assert self._index_ortho is not None
        return self._index_ortho

    def _get_index_phone(self) -> dict[str, list[dict[str, Any]]]:
        """Retourne l'index phone, le chargeant si necessaire."""
        if self._index_phone is None:
            self._charger_index()
        assert self._index_phone is not None
        return self._index_phone

    def _get_index_lemme(self) -> dict[str, list[dict[str, Any]]]:
        """Retourne l'index lemme, le chargeant si necessaire."""
        if self._index_lemme is None:
            self._charger_index()
        assert self._index_lemme is not None
        return self._index_lemme

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
            if self._schema_version >= 4:
                cur = conn.execute(
                    "SELECT f.ortho, l.cgram, f.freq_opensubs AS freq, f.phone "
                    "FROM formes f "
                    "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                    "WHERE f.phone = ?",
                    (phone,),
                )
            else:
                cur = conn.execute(
                    f"SELECT ortho, cgram, freq_opensubs AS freq, phone "
                    f"FROM {self._table} WHERE phone = ?",  # noqa: S608
                    (phone,),
                )
            return [dict(row) for row in cur.fetchall()]

        # CSV/TSV : charger les index si necessaire
        return self._get_index_phone().get(phone, [])

    def info(self, mot: str) -> list[dict[str, Any]]:
        """Retourne les entrees lexicales completes pour un mot."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            if self._schema_version >= 4:
                return self._info_v4(conn, mot)
            cur = conn.execute(
                f"SELECT * FROM {self._table} WHERE ortho = ? COLLATE NOCASE",  # noqa: S608
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
        return self._get_index_ortho().get(normaliser_ortho(mot), [])

    def _info_v4(self, conn: sqlite3.Connection, mot: str) -> list[dict[str, Any]]:
        """info() pour schema v4 : JOIN formes + lemmes."""
        cur = conn.execute(
            "SELECT f.id, f.ortho, f.multext, f.phone, f.phone_reversed, "
            "f.nb_syllabes, f.syllabes, f.freq_opensubs AS freq_opensubs, "
            "f.freq_frantext, f.freq_lm10, f.freq_frwac, f.freq_composite, f.source, "
            "l.id AS lemme_id, l.lemme, l.cgram, l.genre, l.contrainte_nombre, "
            "l.etymologie, l.freq_opensubs AS lemme_freq, l.age "
            "FROM formes f "
            "LEFT JOIN lemmes l ON f.lemme_id = l.id "
            "WHERE f.ortho = ? COLLATE NOCASE",
            (normaliser_ortho(mot),),
        )
        results = []
        for row in cur.fetchall():
            entry: dict[str, Any] = dict(row)
            # Decoder multext en traits lisibles
            multext = entry.get("multext") or ""
            if multext:
                traits = _decoder_multext(multext)
                entry["genre"] = entry.get("genre") or traits.get("genre", "")
                entry["nombre"] = traits.get("nombre", "")
                entry["mode"] = traits.get("mode", "")
                entry["temps"] = traits.get("temps", "")
                entry["personne"] = traits.get("personne", "")
            entry["freq"] = entry.get("freq_composite") or entry.get("freq_opensubs", 0.0) or 0.0
            results.append(entry)
        return results

    def frequence(self, mot: str) -> float:
        """Frequence max parmi toutes les entrees de ce mot."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            cur = conn.execute(
                f"SELECT MAX(freq_opensubs) FROM {self._table} "  # noqa: S608
                f"WHERE ortho = ? COLLATE NOCASE",
                (normaliser_ortho(mot),),
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0

        # CSV/TSV
        entries = self._get_index_ortho().get(normaliser_ortho(mot), [])
        if not entries:
            return 0.0
        return max(float(e.get("freq", 0.0)) for e in entries)

    def phone_de(self, mot: str) -> str | None:
        """Retourne la prononciation la plus frequente d'un mot."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            cur = conn.execute(
                f"SELECT phone FROM {self._table} "  # noqa: S608
                f"WHERE ortho = ? COLLATE NOCASE AND phone != '' "
                f"ORDER BY freq_opensubs DESC LIMIT 1",
                (normaliser_ortho(mot),),
            )
            row = cur.fetchone()
            return row[0] if row else None

        # CSV/TSV
        entries = self._get_index_ortho().get(normaliser_ortho(mot), [])
        phones = [
            (e.get("phone", ""), float(e.get("freq", 0.0)))
            for e in entries
            if e.get("phone")
        ]
        if not phones:
            return None
        phones.sort(key=lambda x: -x[1])
        return phones[0][0]

    def verbes_par_phone_et_personne(
        self,
        phones: list[str],
        personne: str,
        nombre: str,
    ) -> list[dict[str, Any]]:
        """Trouve les formes verbales correspondant a une liste de phones et personne.

        Args:
            phones: Liste de transcriptions IPA (original + variantes proches)
            personne: Personne grammaticale ("1", "2", "3")
            nombre: Nombre grammatical ("s", "p")

        Returns:
            Liste de dicts avec cles : ortho, lemme, cgram, personne, nombre, freq
        """
        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        if self._backend == "sqlite":
            conn = self._get_conn()
            placeholders = ",".join("?" for _ in phones)

            if self._schema_version >= 4:
                # v4 : traits dans multext, lemme dans table lemmes
                multext_pattern = _filtre_multext(
                    pos="VER", personne=personne,
                    nombre={"s": "singulier", "p": "pluriel"}.get(nombre, nombre) if nombre else None,
                )
                query = (
                    "SELECT f.ortho, l.lemme, l.cgram, f.multext, "
                    "f.freq_composite, f.freq_opensubs AS freq, f.phone "
                    "FROM formes f "
                    "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                    f"WHERE f.phone IN ({placeholders}) "
                    "AND l.cgram IN ('VER', 'AUX') "
                )
                params: list[Any] = list(phones)
                if multext_pattern and multext_pattern != "%":
                    query += "AND f.multext LIKE ? "
                    params.append(multext_pattern)
                query += "ORDER BY f.freq_composite DESC"
                cur = conn.execute(query, params)
                for row in cur.fetchall():
                    ortho = row["ortho"]
                    if ortho.lower() not in seen:
                        seen.add(ortho.lower())
                        entry = dict(row)
                        multext = entry.pop("multext", "") or ""
                        if multext:
                            traits = _decoder_multext(multext)
                            entry["personne"] = traits.get("personne", "")
                            entry["nombre"] = traits.get("nombre", "")
                            entry["mode"] = traits.get("mode", "")
                            entry["temps"] = traits.get("temps", "")
                        results.append(entry)
            else:
                query = (
                    f"SELECT ortho, lemme, cgram, personne, nombre, mode, temps, "
                    f"freq_opensubs AS freq, phone "
                    f"FROM {self._table} "
                    f"WHERE phone IN ({placeholders}) "
                    f"AND cgram IN ('VER', 'AUX') "
                    f"AND personne = ?"
                )
                params = list(phones) + [personne]
                if nombre:
                    query += " AND nombre = ?"
                    params.append(nombre)
                query += " ORDER BY freq_opensubs DESC"

                cur = conn.execute(query, params)
                for row in cur.fetchall():
                    ortho = row["ortho"]
                    if ortho.lower() not in seen:
                        seen.add(ortho.lower())
                        results.append(dict(row))
        else:
            # CSV/TSV : chercher dans l'index phone
            phone_index = self._get_index_phone()

            for phone in phones:
                entries = phone_index.get(phone, [])
                for e in entries:
                    if e.get("cgram", "") not in ("VER", "AUX"):
                        continue
                    if e.get("personne", "") != personne:
                        continue
                    if nombre and e.get("nombre", "") != nombre:
                        continue
                    ortho = e.get("ortho", "")
                    if ortho.lower() not in seen:
                        seen.add(ortho.lower())
                        results.append(e)

            results.sort(key=lambda x: -float(x.get("freq", 0)))

        return results

    # --- Methodes facades : Morphologie ---

    def conjuguer(self, verbe: str) -> dict[str, dict[str, dict[str, str]]]:
        """Table de conjugaison d'un verbe.

        Retour : {mode: {temps: {"1s": forme, "2s": forme, ...}}}
        """
        if self._backend == "sqlite":
            conn = self._get_conn()

            if self._schema_version >= 4:
                cur = conn.execute(
                    "SELECT f.ortho, f.multext, f.phone, f.freq_opensubs, "
                    "l.lemme, l.cgram "
                    "FROM formes f "
                    "JOIN lemmes l ON f.lemme_id = l.id "
                    "WHERE l.lemme = ? COLLATE NOCASE "
                    "AND l.cgram IN ('VER', 'AUX')",
                    (normaliser_ortho(verbe),),
                )
                entries = []
                for row in cur.fetchall():
                    entry: dict[str, Any] = dict(row)
                    multext = entry.get("multext") or ""
                    if multext:
                        traits = _decoder_multext(multext)
                        entry["mode"] = traits.get("mode", "")
                        entry["temps"] = traits.get("temps", "")
                        entry["personne"] = traits.get("personne", "")
                        entry["nombre"] = traits.get("nombre", "")
                        entry["genre"] = traits.get("genre", "")
                    entries.append(entry)
                return _morphologie.conjuguer(entries)

            cur = conn.execute(
                f"SELECT * FROM {self._table} "  # noqa: S608
                f"WHERE lemme = ? COLLATE NOCASE "
                f"AND cgram IN ('VER', 'AUX')",
                (normaliser_ortho(verbe),),
            )
            colonnes = [desc[0] for desc in cur.description]
            mapping = resoudre_colonnes(colonnes)
            entries = []
            for row in cur.fetchall():
                entry: dict[str, Any] = {}
                for col, val in zip(colonnes, row):
                    canon = mapping.get(col, col)
                    entry[canon] = val
                entries.append(entry)
            return _morphologie.conjuguer(entries)

        entries = self._get_index_lemme().get(normaliser_ortho(verbe), [])
        return _morphologie.conjuguer(entries)

    def formes_de(self, lemme: str, cgram: str | None = None) -> list[dict[str, Any]]:
        """Toutes les formes flechies d'un lemme, avec filtre POS optionnel."""
        if self._backend == "sqlite":
            conn = self._get_conn()

            if self._schema_version >= 4:
                return self._formes_de_v4(conn, lemme, cgram)

            query = f"SELECT * FROM {self._table} WHERE lemme = ? COLLATE NOCASE"  # noqa: S608
            params: list[Any] = [normaliser_ortho(lemme)]
            if cgram is not None:
                query += " AND cgram LIKE ?"
                params.append(f"{cgram}%")
            cur = conn.execute(query, params)
            colonnes = [desc[0] for desc in cur.description]
            mapping = resoudre_colonnes(colonnes)
            entries = []
            for row in cur.fetchall():
                entry: dict[str, Any] = {}
                for col, val in zip(colonnes, row):
                    canon = mapping.get(col, col)
                    entry[canon] = val
                entries.append(entry)
            return _morphologie.formes_de(entries, cgram)

        entries = self._get_index_lemme().get(normaliser_ortho(lemme), [])
        return _morphologie.formes_de(entries, cgram)

    def _formes_de_v4(
        self, conn: sqlite3.Connection, lemme: str, cgram: str | None = None,
    ) -> list[dict[str, Any]]:
        """formes_de() pour schema v4."""
        query = (
            "SELECT f.ortho, f.multext, f.phone, f.nb_syllabes, "
            "f.freq_opensubs, l.lemme, l.cgram, l.genre "
            "FROM formes f "
            "JOIN lemmes l ON f.lemme_id = l.id "
            "WHERE l.lemme = ? COLLATE NOCASE"
        )
        params: list[Any] = [normaliser_ortho(lemme)]
        if cgram is not None:
            query += " AND l.cgram LIKE ?"
            params.append(f"{cgram}%")
        cur = conn.execute(query, params)
        results = []
        seen: set[str] = set()
        for row in cur.fetchall():
            entry: dict[str, Any] = dict(row)
            multext = entry.get("multext") or ""
            if multext:
                traits = _decoder_multext(multext)
                entry["genre"] = entry.get("genre") or traits.get("genre", "")
                entry["nombre"] = traits.get("nombre", "")
                entry["mode"] = traits.get("mode", "")
                entry["temps"] = traits.get("temps", "")
                entry["personne"] = traits.get("personne", "")
            ortho = str(entry.get("ortho", "")).lower()
            if ortho not in seen:
                seen.add(ortho)
                results.append(entry)
        return results

    def lemme_de(self, mot: str) -> str | None:
        """Lemme le plus frequent parmi les entrees d'un mot."""
        entries = self.info(mot)
        return _morphologie.lemme_de(entries)

    # --- Methodes facades : Phonetique ---

    def rimes(self, mot: str, nb_phonemes: int = 2, limite: int = 50) -> list[dict[str, Any]]:
        """Mots partageant les N derniers phonemes avec le mot donne."""
        phone = self.phone_de(mot)
        if not phone:
            return []

        if self._backend == "sqlite":
            from lectura_lexique._utils import _tokenize_ipa
            segments = _tokenize_ipa(phone)
            if len(segments) < nb_phonemes:
                return []
            # Requete via phone_reversed si colonne existe
            suffixe = segments[-nb_phonemes:]
            suffixe_rev = "".join(reversed(suffixe))
            conn = self._get_conn()
            try:
                if self._schema_version >= 4:
                    cur = conn.execute(
                        "SELECT f.ortho, f.phone, f.phone_reversed, f.multext, "
                        "f.nb_syllabes, f.freq_opensubs, f.freq_composite, "
                        "l.lemme, l.cgram, l.genre "
                        "FROM formes f "
                        "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                        "WHERE f.phone_reversed LIKE ? "
                        "ORDER BY f.freq_composite DESC",
                        (f"{suffixe_rev}%",),
                    )
                    results = []
                    seen: set[str] = set()
                    for row in cur.fetchall():
                        ortho = row["ortho"].lower()
                        if ortho in seen:
                            continue
                        seen.add(ortho)
                        entry: dict[str, Any] = dict(row)
                        entry["freq"] = entry.get("freq_composite") or entry.get("freq_opensubs", 0.0) or 0.0
                        results.append(entry)
                        if len(results) >= limite:
                            break
                    return results
                else:
                    cur = conn.execute(
                        f"SELECT * FROM {self._table} "  # noqa: S608
                        f"WHERE phone_reversed LIKE ? "
                        f"ORDER BY freq_opensubs DESC LIMIT ?",
                        (f"{suffixe_rev}%", limite),
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
            except sqlite3.OperationalError:
                # phone_reversed n'existe pas, scan complet
                pass

        phone_index = self._get_index_phone()
        return _phonetique.rimes(phone_index, phone, nb_phonemes, limite)

    def contient_son(self, son: str, limite: int = 50) -> list[dict[str, Any]]:
        """Mots contenant une sequence phonetique."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            if self._schema_version >= 4:
                cur = conn.execute(
                    "SELECT f.ortho, f.phone, f.multext, f.nb_syllabes, "
                    "f.freq_opensubs, f.freq_composite, l.lemme, l.cgram "
                    "FROM formes f "
                    "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                    "WHERE f.phone LIKE ? "
                    "ORDER BY f.freq_composite DESC LIMIT ?",
                    (f"%{son}%", limite),
                )
                results = []
                for row in cur.fetchall():
                    entry: dict[str, Any] = dict(row)
                    entry["freq"] = entry.get("freq_composite") or entry.get("freq_opensubs", 0.0) or 0.0
                    results.append(entry)
                return results
            else:
                cur = conn.execute(
                    f"SELECT * FROM {self._table} "  # noqa: S608
                    f"WHERE phone LIKE ? "
                    f"ORDER BY freq_opensubs DESC LIMIT ?",
                    (f"%{son}%", limite),
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

        phone_index = self._get_index_phone()
        return _phonetique.contient_son(phone_index, son, limite)

    def mots_par_syllabes(
        self, n: int, cgram: str | None = None, limite: int = 50,
    ) -> list[dict[str, Any]]:
        """Mots avec exactement N syllabes."""
        if self._backend == "sqlite":
            conn = self._get_conn()

            if self._schema_version >= 4:
                query = (
                    "SELECT f.ortho, f.multext, f.phone, f.nb_syllabes, "
                    "f.freq_opensubs, f.freq_composite, l.lemme, l.cgram, l.genre "
                    "FROM formes f "
                    "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                    "WHERE f.nb_syllabes = ?"
                )
                params: list[Any] = [n]
                if cgram is not None:
                    query += " AND l.cgram LIKE ?"
                    params.append(f"{cgram}%")
                query += " ORDER BY f.freq_composite DESC LIMIT ?"
                params.append(limite)
                try:
                    cur = conn.execute(query, params)
                    results = []
                    for row in cur.fetchall():
                        entry: dict[str, Any] = dict(row)
                        multext = entry.get("multext") or ""
                        if multext:
                            traits = _decoder_multext(multext)
                            entry["genre"] = entry.get("genre") or traits.get("genre", "")
                            entry["nombre"] = traits.get("nombre", "")
                        entry["freq"] = entry.get("freq_composite") or entry.get("freq_opensubs", 0.0) or 0.0
                        results.append(entry)
                    return results
                except sqlite3.OperationalError:
                    pass
            else:
                query = (
                    f"SELECT * FROM {self._table} "  # noqa: S608
                    f"WHERE nb_syllabes = ?"
                )
                params = [n]
                if cgram is not None:
                    query += " AND cgram LIKE ?"
                    params.append(f"{cgram}%")
                query += " ORDER BY freq_opensubs DESC LIMIT ?"
                params.append(limite)
                try:
                    cur = conn.execute(query, params)
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
                except sqlite3.OperationalError:
                    pass  # colonne absente, fallback

        ortho_index = self._get_index_ortho()
        return _phonetique.mots_par_syllabes(ortho_index, n, cgram, limite)

    # --- Methodes : Definitions multi-sens ---

    def definitions(self, mot: str, cgram: str | None = None) -> list[dict[str, Any]]:
        """Definitions multiples depuis la table definitions (v3) ou concepts (v4).

        Retourne [] si la table est absente (compatibilite anciennes BDD).
        Ordonne par cgram, sens_num.
        """
        if self._backend != "sqlite":
            return []

        if self._schema_version >= 4:
            # v4 : deleguer vers concepts_de
            concepts = self.concepts_de(mot, cgram)
            results: list[dict[str, Any]] = []
            for c in concepts:
                d: dict[str, Any] = dict(c)
                d["lemme"] = mot
                d["domaine"] = c.get("theme") or ""
                # Recuperer exemples, synonymes, antonymes pour compat v3
                cid = c.get("id")
                if cid is not None:
                    d["exemples"] = self.exemples_de(cid)
                    syn_concepts = self.synonymes_de(cid)
                    # Retourner les noms de lemmes (dedupliques)
                    seen_syn: set[str] = set()
                    syn_words: list[str] = []
                    for s in syn_concepts:
                        w = s.get("_lemme") or ""
                        if w and w.lower() not in seen_syn:
                            seen_syn.add(w.lower())
                            syn_words.append(w)
                    d["synonymes"] = syn_words
                    ant_concepts = self.antonymes_de(cid)
                    seen_ant: set[str] = set()
                    ant_words: list[str] = []
                    for a in ant_concepts:
                        w = a.get("_lemme") or ""
                        if w and w.lower() not in seen_ant:
                            seen_ant.add(w.lower())
                            ant_words.append(w)
                    d["antonymes"] = ant_words
                else:
                    d["exemples"] = []
                    d["synonymes"] = []
                    d["antonymes"] = []
                d["tags"] = []
                results.append(d)
            return results

        conn = self._get_conn()
        try:
            if cgram:
                cur = conn.execute(
                    "SELECT * FROM definitions "
                    "WHERE lemme = ? COLLATE NOCASE AND cgram = ? "
                    "ORDER BY cgram, sens_num",
                    (mot, cgram),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM definitions "
                    "WHERE lemme = ? COLLATE NOCASE "
                    "ORDER BY cgram, sens_num",
                    (mot,),
                )
            rows = cur.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                # Convertir les champs separes par ; en listes
                for key in ("exemples", "synonymes", "antonymes", "tags"):
                    val = d.get(key)
                    if val:
                        d[key] = [x.strip() for x in val.split(";") if x.strip()]
                    else:
                        d[key] = []
                results.append(d)
            return results
        except sqlite3.OperationalError:
            # Table definitions absente
            return []

    # --- Methodes facades : Semantique ---

    def synonymes(self, mot: str) -> list[str]:
        """Synonymes d'un mot (colonne 'synonymes', separateur ;)."""
        entries = self.info(mot)
        return _semantique.synonymes(entries)

    def antonymes(self, mot: str) -> list[str]:
        """Antonymes d'un mot (colonne 'antonymes', separateur ;)."""
        entries = self.info(mot)
        return _semantique.antonymes(entries)

    def definition(self, mot: str) -> list[str]:
        """Definitions d'un mot."""
        entries = self.info(mot)
        return _semantique.definition(entries)

    # --- Methodes facades : Recherche ---

    def rechercher(
        self, pattern: str, champ: str = "ortho", limite: int = 50,
    ) -> list[dict[str, Any]]:
        """Recherche par regex sur ortho ou phone."""
        if self._backend == "sqlite":
            import re as _re
            try:
                regex = _re.compile(pattern, _re.IGNORECASE)
            except _re.error:
                return []
            conn = self._get_conn()
            conn.create_function("REGEXP", 2, lambda p, s: bool(regex.search(s)) if s else False)
            col = "phone" if champ == "phone" else "ortho"

            if self._schema_version >= 4:
                col_qualified = f"f.{col}"
                cur = conn.execute(
                    "SELECT f.ortho, f.phone, f.multext, f.nb_syllabes, "
                    "f.freq_opensubs, f.freq_composite, l.lemme, l.cgram "
                    "FROM formes f "
                    "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                    f"WHERE {col_qualified} REGEXP ? "
                    "ORDER BY f.freq_composite DESC LIMIT ?",
                    (pattern, limite),
                )
                results: list[dict[str, Any]] = []
                seen: set[str] = set()
                for row in cur.fetchall():
                    entry: dict[str, Any] = dict(row)
                    entry["freq"] = entry.get("freq_composite") or entry.get("freq_opensubs", 0.0) or 0.0
                    ortho = str(entry.get("ortho", "")).lower()
                    if ortho not in seen:
                        seen.add(ortho)
                        results.append(entry)
                return results
            else:
                cur = conn.execute(
                    f"SELECT * FROM {self._table} "  # noqa: S608
                    f"WHERE {col} REGEXP ? "
                    f"ORDER BY freq_opensubs DESC LIMIT ?",
                    (pattern, limite),
                )
                colonnes = [desc[0] for desc in cur.description]
                mapping = resoudre_colonnes(colonnes)
                results = []
                seen = set()
                for row in cur.fetchall():
                    entry: dict[str, Any] = {}
                    for col_name, val in zip(colonnes, row):
                        canon = mapping.get(col_name, col_name)
                        entry[canon] = val
                    ortho = str(entry.get("ortho", "")).lower()
                    if ortho not in seen:
                        seen.add(ortho)
                        results.append(entry)
                return results

        if champ == "phone":
            index = self._get_index_phone()
        else:
            index = self._get_index_ortho()
        return _recherche.rechercher(index, pattern, champ, limite)

    def _build_filter_clauses(
        self,
        *,
        cgram: str | None = None,
        genre: str | None = None,
        nombre: str | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        nb_syllabes: int | None = None,
        age_min: float | None = None,
        age_max: float | None = None,
        illustrable_min: float | None = None,
        categorie: str | None = None,
        has_age: bool | None = None,
        has_phone: bool | None = None,
    ) -> tuple[list[str], list[Any]]:
        """Construit les clauses WHERE et parametres pour les filtres."""
        conditions: list[str] = []
        params: list[Any] = []
        if cgram is not None:
            conditions.append("cgram LIKE ?")
            params.append(f"{cgram}%")
        if genre is not None:
            conditions.append("genre = ?")
            params.append(genre)
        if nombre is not None:
            conditions.append("nombre = ?")
            params.append(nombre)
        if freq_min is not None:
            conditions.append("freq_opensubs >= ?")
            params.append(freq_min)
        if freq_max is not None:
            conditions.append("freq_opensubs <= ?")
            params.append(freq_max)
        if nb_syllabes is not None:
            conditions.append("nb_syllabes = ?")
            params.append(nb_syllabes)
        if has_age is True:
            conditions.append("age IS NOT NULL")
        elif has_age is False:
            conditions.append("age IS NULL")
        if age_min is not None:
            conditions.append("age >= ?")
            params.append(age_min)
        if age_max is not None:
            conditions.append("age <= ?")
            params.append(age_max)
        if illustrable_min is not None:
            conditions.append("illustrable >= ?")
            params.append(illustrable_min)
        if categorie is not None:
            conditions.append("categorie = ?")
            params.append(categorie)
        if has_phone is True:
            conditions.append("phone IS NOT NULL AND phone != ''")
        elif has_phone is False:
            conditions.append("(phone IS NULL OR phone = '')")
        return conditions, params

    def filtrer(
        self,
        *,
        cgram: str | None = None,
        genre: str | None = None,
        nombre: str | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        nb_syllabes: int | None = None,
        age_min: float | None = None,
        age_max: float | None = None,
        illustrable_min: float | None = None,
        categorie: str | None = None,
        has_age: bool | None = None,
        has_phone: bool | None = None,
        mode: str | None = None,
        temps: str | None = None,
        personne: str | None = None,
        limite: int = 100,
    ) -> list[dict[str, Any]]:
        """Filtre multi-critere sur l'ensemble du lexique."""
        if self._backend == "sqlite":
            conn = self._get_conn()

            if self._schema_version >= 4:
                return self._filtrer_v4(
                    conn, cgram=cgram, genre=genre, nombre=nombre,
                    freq_min=freq_min, freq_max=freq_max, nb_syllabes=nb_syllabes,
                    age_min=age_min, age_max=age_max,
                    has_age=has_age, has_phone=has_phone,
                    mode=mode, temps=temps, personne=personne,
                    limite=limite,
                )

            conditions, params = self._build_filter_clauses(
                cgram=cgram, genre=genre, nombre=nombre,
                freq_min=freq_min, freq_max=freq_max, nb_syllabes=nb_syllabes,
                age_min=age_min, age_max=age_max,
                illustrable_min=illustrable_min, categorie=categorie,
                has_age=has_age, has_phone=has_phone,
            )

            where = " AND ".join(conditions) if conditions else "1=1"
            query = (
                f"SELECT * FROM {self._table} "  # noqa: S608
                f"WHERE {where} LIMIT ?"
            )
            params.append(limite)
            cur = conn.execute(query, params)
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

        # CSV/TSV : iterer sur tout l'index ortho
        ortho_index = self._get_index_ortho()
        all_entries = (e for entries in ortho_index.values() for e in entries)
        return _recherche.filtrer(
            all_entries,
            cgram=cgram,
            genre=genre,
            nombre=nombre,
            freq_min=freq_min,
            freq_max=freq_max,
            nb_syllabes=nb_syllabes,
            limite=limite,
        )

    def _filtrer_v4(
        self,
        conn: sqlite3.Connection,
        *,
        cgram: str | None = None,
        genre: str | None = None,
        nombre: str | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        nb_syllabes: int | None = None,
        age_min: float | None = None,
        age_max: float | None = None,
        has_age: bool | None = None,
        has_phone: bool | None = None,
        mode: str | None = None,
        temps: str | None = None,
        personne: str | None = None,
        limite: int = 100,
    ) -> list[dict[str, Any]]:
        """Filtre v4 avec multext patterns et JOIN lemmes."""
        conditions: list[str] = []
        params: list[Any] = []

        # Multext pattern pour genre/nombre/mode/temps/personne
        multext_pattern = _filtre_multext(
            pos=cgram, mode=mode, temps=temps,
            personne=personne, nombre=nombre, genre=genre,
        )
        if multext_pattern and multext_pattern != "%":
            conditions.append("f.multext LIKE ?")
            params.append(multext_pattern)
        elif cgram:
            conditions.append("l.cgram LIKE ?")
            params.append(f"{cgram}%")

        if freq_min is not None:
            conditions.append("f.freq_opensubs >= ?")
            params.append(freq_min)
        if freq_max is not None:
            conditions.append("f.freq_opensubs <= ?")
            params.append(freq_max)
        if nb_syllabes is not None:
            conditions.append("f.nb_syllabes = ?")
            params.append(nb_syllabes)
        if has_age is True:
            conditions.append("l.age IS NOT NULL")
        elif has_age is False:
            conditions.append("l.age IS NULL")
        if age_min is not None:
            conditions.append("l.age >= ?")
            params.append(age_min)
        if age_max is not None:
            conditions.append("l.age <= ?")
            params.append(age_max)
        if has_phone is True:
            conditions.append("f.phone IS NOT NULL AND f.phone != ''")
        elif has_phone is False:
            conditions.append("(f.phone IS NULL OR f.phone = '')")

        where = " AND ".join(conditions) if conditions else "1=1"
        query = (
            "SELECT f.id, f.ortho, f.multext, f.phone, f.nb_syllabes, "
            "f.freq_opensubs, l.lemme, l.cgram, l.genre, l.age "
            "FROM formes f "
            "LEFT JOIN lemmes l ON f.lemme_id = l.id "
            f"WHERE {where} LIMIT ?"
        )
        params.append(limite)
        cur = conn.execute(query, params)
        results = []
        for row in cur.fetchall():
            entry: dict[str, Any] = dict(row)
            multext = entry.get("multext") or ""
            if multext:
                traits = _decoder_multext(multext)
                entry["genre"] = entry.get("genre") or traits.get("genre", "")
                entry["nombre"] = traits.get("nombre", "")
                entry["mode"] = traits.get("mode", "")
                entry["temps"] = traits.get("temps", "")
                entry["personne"] = traits.get("personne", "")
            entry["freq"] = entry.get("freq_opensubs", 0.0) or 0.0
            results.append(entry)
        return results

    def compter(
        self,
        *,
        cgram: str | None = None,
        genre: str | None = None,
        nombre: str | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        nb_syllabes: int | None = None,
        age_min: float | None = None,
        age_max: float | None = None,
        illustrable_min: float | None = None,
        categorie: str | None = None,
        has_age: bool | None = None,
        has_phone: bool | None = None,
    ) -> int:
        """Compte les formes matchant les filtres (sans les charger)."""
        if self._backend == "sqlite":
            conn = self._get_conn()

            if self._schema_version >= 4:
                # v4 : utiliser le meme schema que _filtrer_v4
                conditions: list[str] = []
                params: list[Any] = []
                multext_pattern = _filtre_multext(
                    pos=cgram, nombre=nombre, genre=genre,
                )
                if multext_pattern and multext_pattern != "%":
                    conditions.append("f.multext LIKE ?")
                    params.append(multext_pattern)
                elif cgram:
                    conditions.append("l.cgram LIKE ?")
                    params.append(f"{cgram}%")
                if freq_min is not None:
                    conditions.append("f.freq_opensubs >= ?")
                    params.append(freq_min)
                if freq_max is not None:
                    conditions.append("f.freq_opensubs <= ?")
                    params.append(freq_max)
                if nb_syllabes is not None:
                    conditions.append("f.nb_syllabes = ?")
                    params.append(nb_syllabes)
                if has_age is True:
                    conditions.append("l.age IS NOT NULL")
                elif has_age is False:
                    conditions.append("l.age IS NULL")
                if age_min is not None:
                    conditions.append("l.age >= ?")
                    params.append(age_min)
                if age_max is not None:
                    conditions.append("l.age <= ?")
                    params.append(age_max)
                if has_phone is True:
                    conditions.append("f.phone IS NOT NULL AND f.phone != ''")
                elif has_phone is False:
                    conditions.append("(f.phone IS NULL OR f.phone = '')")
                where = " AND ".join(conditions) if conditions else "1=1"
                query = (
                    "SELECT COUNT(*) FROM formes f "
                    "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                    f"WHERE {where}"
                )
                cur = conn.execute(query, params)
                row = cur.fetchone()
                return int(row[0]) if row else 0

            conditions_v3, params_v3 = self._build_filter_clauses(
                cgram=cgram, genre=genre, nombre=nombre,
                freq_min=freq_min, freq_max=freq_max, nb_syllabes=nb_syllabes,
                age_min=age_min, age_max=age_max,
                illustrable_min=illustrable_min, categorie=categorie,
                has_age=has_age, has_phone=has_phone,
            )
            where = " AND ".join(conditions_v3) if conditions_v3 else "1=1"
            query = f"SELECT COUNT(*) FROM {self._table} WHERE {where}"  # noqa: S608
            cur = conn.execute(query, params_v3)
            row = cur.fetchone()
            return int(row[0]) if row else 0

        # CSV/TSV : compter via filtrer() sans limite
        ortho_index = self._get_index_ortho()
        all_entries = (e for entries in ortho_index.values() for e in entries)
        return len(_recherche.filtrer(
            all_entries,
            cgram=cgram, genre=genre, nombre=nombre,
            freq_min=freq_min, freq_max=freq_max, nb_syllabes=nb_syllabes,
            limite=0,
        ))

    def exporter(
        self,
        output_path: str | Path,
        *,
        format: str = "csv",
        colonnes: list[str] | None = None,
        cgram: str | None = None,
        genre: str | None = None,
        nombre: str | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        nb_syllabes: int | None = None,
        age_min: float | None = None,
        age_max: float | None = None,
        illustrable_min: float | None = None,
        categorie: str | None = None,
        has_age: bool | None = None,
        has_phone: bool | None = None,
        limite: int = 0,
        callback: Callable[[int], None] | None = None,
    ) -> int:
        """Exporte un sous-ensemble filtre du lexique.

        Args:
            output_path: Chemin du fichier de sortie
            format: "csv" ou "sqlite"
            colonnes: Liste des colonnes a inclure (None = toutes)
            cgram..has_phone: Filtres (memes que filtrer())
            limite: 0 = pas de limite
            callback: Appele tous les 5000 lignes avec le nombre courant

        Returns:
            Nombre de lignes exportees
        """
        output_path = Path(output_path)
        filter_kwargs = dict(
            cgram=cgram, genre=genre, nombre=nombre,
            freq_min=freq_min, freq_max=freq_max, nb_syllabes=nb_syllabes,
            age_min=age_min, age_max=age_max,
            illustrable_min=illustrable_min, categorie=categorie,
            has_age=has_age, has_phone=has_phone,
        )

        if format == "sqlite":
            return self._exporter_sqlite(
                output_path, colonnes=colonnes, limite=limite,
                callback=callback, **filter_kwargs,
            )
        return self._exporter_csv(
            output_path, colonnes=colonnes, limite=limite,
            callback=callback, **filter_kwargs,
        )

    def _iter_filtered_rows(
        self,
        *,
        colonnes: list[str] | None = None,
        limite: int = 0,
        callback: Callable[[int], None] | None = None,
        **filter_kwargs: Any,
    ):
        """Itere sur les lignes filtrees (generateur)."""
        count = 0
        batch_size = 5000

        if self._backend == "sqlite":
            conn = self._get_conn()

            if self._schema_version >= 4:
                # v4 : utiliser un JOIN formes+lemmes
                conditions: list[str] = []
                params: list[Any] = []
                cgram = filter_kwargs.get("cgram")
                if cgram:
                    conditions.append("l.cgram LIKE ?")
                    params.append(f"{cgram}%")
                genre = filter_kwargs.get("genre")
                if genre:
                    conditions.append("l.genre = ?")
                    params.append(genre)
                freq_min = filter_kwargs.get("freq_min")
                if freq_min is not None:
                    conditions.append("f.freq_opensubs >= ?")
                    params.append(freq_min)
                freq_max = filter_kwargs.get("freq_max")
                if freq_max is not None:
                    conditions.append("f.freq_opensubs <= ?")
                    params.append(freq_max)
                nb_syllabes = filter_kwargs.get("nb_syllabes")
                if nb_syllabes is not None:
                    conditions.append("f.nb_syllabes = ?")
                    params.append(nb_syllabes)
                has_phone = filter_kwargs.get("has_phone")
                if has_phone is True:
                    conditions.append("f.phone IS NOT NULL AND f.phone != ''")
                elif has_phone is False:
                    conditions.append("(f.phone IS NULL OR f.phone = '')")

                where = " AND ".join(conditions) if conditions else "1=1"
                query = (
                    "SELECT f.id, f.ortho, f.multext, f.phone, f.phone_reversed, "
                    "f.nb_syllabes, f.syllabes, f.freq_opensubs, f.freq_frantext, "
                    "f.freq_lm10, f.freq_frwac, f.freq_composite, f.source, "
                    "l.lemme, l.cgram, l.genre "
                    "FROM formes f "
                    "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                    f"WHERE {where}"
                )
                if limite:
                    query += " LIMIT ?"
                    params.append(limite)

                cur = conn.execute(query, params)
                col_names = [desc[0] for desc in cur.description]
                for row in cur:
                    entry = dict(zip(col_names, row))
                    if colonnes:
                        entry = {k: v for k, v in entry.items() if k in colonnes}
                    yield entry
                    count += 1
                    if callback and count % batch_size == 0:
                        callback(count)
            else:
                conditions_v3, params_v3 = self._build_filter_clauses(**filter_kwargs)
                where = " AND ".join(conditions_v3) if conditions_v3 else "1=1"

                # Determiner les colonnes disponibles
                cur_cols = conn.execute(f"PRAGMA table_info({self._table})")  # noqa: S608
                all_cols = [row[1] for row in cur_cols.fetchall()]

                if colonnes:
                    select_cols = [c for c in colonnes if c in all_cols]
                else:
                    select_cols = all_cols

                select = ", ".join(select_cols)
                query = f"SELECT {select} FROM {self._table} WHERE {where}"  # noqa: S608
                if limite:
                    query += " LIMIT ?"
                    params_v3.append(limite)

                cur = conn.execute(query, params_v3)
                for row in cur:
                    entry = dict(zip(select_cols, row))
                    yield entry
                    count += 1
                    if callback and count % batch_size == 0:
                        callback(count)
        else:
            ortho_index = self._get_index_ortho()
            all_entries = (e for entries in ortho_index.values() for e in entries)
            for e in _recherche.filtrer(
                all_entries, limite=limite, **filter_kwargs,
            ):
                if colonnes:
                    entry = {k: v for k, v in e.items() if k in colonnes}
                else:
                    entry = dict(e)
                yield entry
                count += 1
                if callback and count % batch_size == 0:
                    callback(count)

    def _exporter_csv(
        self,
        output_path: Path,
        *,
        colonnes: list[str] | None = None,
        limite: int = 0,
        callback: Callable[[int], None] | None = None,
        **filter_kwargs: Any,
    ) -> int:
        """Exporte en CSV."""
        # Determiner les noms de colonnes
        if colonnes:
            fieldnames = list(colonnes)
        elif self._backend == "sqlite":
            conn = self._get_conn()
            cur = conn.execute(f"PRAGMA table_info({self._table})")  # noqa: S608
            fieldnames = [row[1] for row in cur.fetchall()]
        else:
            # Prendre les cles de la premiere entree
            ortho_index = self._get_index_ortho()
            first_entries = next(iter(ortho_index.values()), [])
            fieldnames = list(first_entries[0].keys()) if first_entries else []

        count = 0
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in self._iter_filtered_rows(
                colonnes=colonnes, limite=limite,
                callback=callback, **filter_kwargs,
            ):
                writer.writerow(entry)
                count += 1

        if callback:
            callback(count)
        return count

    def _exporter_sqlite(
        self,
        output_path: Path,
        *,
        colonnes: list[str] | None = None,
        limite: int = 0,
        callback: Callable[[int], None] | None = None,
        **filter_kwargs: Any,
    ) -> int:
        """Exporte en SQLite."""
        if output_path.exists():
            output_path.unlink()

        # Determiner le schema des colonnes
        if self._backend == "sqlite":
            conn_src = self._get_conn()
            cur = conn_src.execute(f"PRAGMA table_info({self._table})")  # noqa: S608
            col_info = cur.fetchall()
            # col_info: (cid, name, type, notnull, dflt_value, pk)
            all_col_defs = [(row[1], row[2]) for row in col_info]
        else:
            # Fallback : toutes les colonnes en TEXT
            ortho_index = self._get_index_ortho()
            first_entries = next(iter(ortho_index.values()), [])
            all_col_defs = [(k, "TEXT") for k in first_entries[0].keys()] if first_entries else []

        if colonnes:
            col_defs = [(name, typ) for name, typ in all_col_defs if name in colonnes]
        else:
            col_defs = all_col_defs

        col_names = [name for name, _ in col_defs]
        col_schema = ", ".join(f"{name} {typ}" for name, typ in col_defs)

        conn_out = sqlite3.connect(str(output_path))
        conn_out.execute("PRAGMA journal_mode=WAL")
        conn_out.execute("PRAGMA synchronous=OFF")
        conn_out.execute(f"CREATE TABLE formes ({col_schema})")
        conn_out.commit()

        placeholders = ", ".join("?" for _ in col_names)
        insert_sql = f"INSERT INTO formes ({', '.join(col_names)}) VALUES ({placeholders})"

        count = 0
        batch: list[tuple] = []
        batch_size = 10_000
        for entry in self._iter_filtered_rows(
            colonnes=colonnes, limite=limite,
            callback=callback, **filter_kwargs,
        ):
            values = tuple(entry.get(c) for c in col_names)
            batch.append(values)
            count += 1
            if len(batch) >= batch_size:
                conn_out.executemany(insert_sql, batch)
                conn_out.commit()
                batch.clear()

        if batch:
            conn_out.executemany(insert_sql, batch)
            conn_out.commit()

        # Creer les index utiles
        for col in ("ortho", "lemme", "phone", "cgram"):
            if col in col_names:
                collate = " COLLATE NOCASE" if col in ("ortho", "lemme") else ""
                conn_out.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{col} ON formes({col}{collate})"
                )
        if "illustrable" in col_names:
            conn_out.execute(
                "CREATE INDEX IF NOT EXISTS idx_illustrable ON formes(illustrable)"
            )
        if "nb_syllabes" in col_names:
            conn_out.execute(
                "CREATE INDEX IF NOT EXISTS idx_nb_syllabes ON formes(nb_syllabes)"
            )
        conn_out.commit()

        conn_out.execute("VACUUM")
        conn_out.close()

        if callback:
            callback(count)
        return count

    def anagrammes(self, mot: str, limite: int = 50) -> list[dict[str, Any]]:
        """Mots avec les memes lettres rearrangees."""
        if self._backend == "sqlite":
            conn = self._get_conn()
            mot_lower = mot.strip().lower()
            mot_sorted = "".join(sorted(mot_lower))
            # SQLite n'a pas de fonction sort-letters; on scanne les mots
            # de meme longueur pour limiter le scan
            cur = conn.execute(
                f"SELECT DISTINCT ortho FROM {self._table} "  # noqa: S608
                f"WHERE length(ortho) = ? AND lower(ortho) != ?",
                (len(mot_lower), mot_lower),
            )
            candidats = [row[0] for row in cur.fetchall()
                         if "".join(sorted(row[0].lower())) == mot_sorted]

            if not candidats:
                return []

            # Recuperer les entrees completes pour les candidats
            placeholders = ",".join("?" for _ in candidats)

            if self._schema_version >= 4:
                cur2 = conn.execute(
                    "SELECT f.ortho, f.phone, f.multext, f.nb_syllabes, "
                    "f.freq_opensubs, f.freq_composite, l.lemme, l.cgram "
                    "FROM formes f "
                    "LEFT JOIN lemmes l ON f.lemme_id = l.id "
                    f"WHERE f.ortho IN ({placeholders}) "
                    "ORDER BY f.freq_composite DESC LIMIT ?",
                    candidats + [limite],
                )
                results: list[dict[str, Any]] = []
                seen: set[str] = set()
                for row in cur2.fetchall():
                    entry: dict[str, Any] = dict(row)
                    entry["freq"] = entry.get("freq_composite") or entry.get("freq_opensubs", 0.0) or 0.0
                    ortho = str(entry.get("ortho", "")).lower()
                    if ortho not in seen:
                        seen.add(ortho)
                        results.append(entry)
                return results
            else:
                cur2 = conn.execute(
                    f"SELECT * FROM {self._table} "  # noqa: S608
                    f"WHERE ortho IN ({placeholders}) "
                    f"ORDER BY freq_opensubs DESC LIMIT ?",
                    candidats + [limite],
                )
                colonnes = [desc[0] for desc in cur2.description]
                mapping = resoudre_colonnes(colonnes)
                results = []
                seen = set()
                for row in cur2.fetchall():
                    entry: dict[str, Any] = {}
                    for col, val in zip(colonnes, row):
                        canon = mapping.get(col, col)
                        entry[canon] = val
                    ortho = str(entry.get("ortho", "")).lower()
                    if ortho not in seen:
                        seen.add(ortho)
                        results.append(entry)
                return results

        ortho_index = self._get_index_ortho()
        return _recherche.anagrammes(ortho_index, mot, limite)

    # --- Methodes v4 : lemmes, concepts, relations ---

    def info_lemme(self, lemme: str, cgram: str | None = None) -> EntreeLemme | None:
        """Retourne les infos d'un lemme (v4 uniquement).

        Args:
            lemme: Le lemme a chercher
            cgram: Filtre categorie grammaticale optionnel

        Returns:
            EntreeLemme ou None si introuvable
        """
        if self._backend != "sqlite" or self._schema_version < 4:
            return None
        conn = self._get_conn()
        if cgram:
            cur = conn.execute(
                "SELECT * FROM lemmes WHERE lemme = ? COLLATE NOCASE AND cgram = ?",
                (normaliser_ortho(lemme), cgram),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM lemmes WHERE lemme = ? COLLATE NOCASE",
                (normaliser_ortho(lemme),),
            )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)  # type: ignore[return-value]

    def concepts_de(self, lemme: str, cgram: str | None = None) -> list[Concept]:
        """Retourne les concepts (sens) d'un lemme (v4 uniquement).

        Args:
            lemme: Le lemme a chercher
            cgram: Filtre categorie optionnel

        Returns:
            Liste de Concept ordonnee par sens_num
        """
        if self._backend != "sqlite" or self._schema_version < 4:
            return []
        conn = self._get_conn()
        if cgram:
            cur = conn.execute(
                "SELECT c.* FROM concepts c "
                "JOIN lemmes l ON c.lemme_id = l.id "
                "WHERE l.lemme = ? COLLATE NOCASE AND l.cgram = ? "
                "ORDER BY c.sens_num",
                (normaliser_ortho(lemme), cgram),
            )
        else:
            cur = conn.execute(
                "SELECT c.* FROM concepts c "
                "JOIN lemmes l ON c.lemme_id = l.id "
                "WHERE l.lemme = ? COLLATE NOCASE "
                "ORDER BY c.sens_num",
                (normaliser_ortho(lemme),),
            )
        return [dict(row) for row in cur.fetchall()]  # type: ignore[misc]

    def synonymes_de(self, concept_id: int) -> list[Concept]:
        """Retourne les concepts synonymes d'un concept (v4 uniquement).

        Chaque concept retourne inclut un champ ``_lemme`` avec le nom du lemme.
        """
        if self._backend != "sqlite" or self._schema_version < 4:
            return []
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT c.*, l.lemme AS _lemme FROM concepts c "
            "LEFT JOIN lemmes l ON c.lemme_id = l.id "
            "JOIN concept_synonymes cs ON "
            "  (cs.concept_b = c.id AND cs.concept_a = ?) OR "
            "  (cs.concept_a = c.id AND cs.concept_b = ?)",
            (concept_id, concept_id),
        )
        return [dict(row) for row in cur.fetchall()]  # type: ignore[misc]

    def antonymes_de(self, concept_id: int) -> list[Concept]:
        """Retourne les concepts antonymes d'un concept (v4 uniquement).

        Chaque concept retourne inclut un champ ``_lemme`` avec le nom du lemme.
        """
        if self._backend != "sqlite" or self._schema_version < 4:
            return []
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT c.*, l.lemme AS _lemme FROM concepts c "
            "LEFT JOIN lemmes l ON c.lemme_id = l.id "
            "JOIN concept_antonymes ca ON "
            "  (ca.concept_b = c.id AND ca.concept_a = ?) OR "
            "  (ca.concept_a = c.id AND ca.concept_b = ?)",
            (concept_id, concept_id),
        )
        return [dict(row) for row in cur.fetchall()]  # type: ignore[misc]

    def exemples_de(self, concept_id: int) -> list[str]:
        """Retourne les exemples d'usage d'un concept (v4 uniquement)."""
        if self._backend != "sqlite" or self._schema_version < 4:
            return []
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT exemple FROM concept_exemples WHERE concept_id = ?",
            (concept_id,),
        )
        return [row[0] for row in cur.fetchall() if row[0]]

    def categories_de(self, concept_id: int) -> list[str]:
        """Retourne les categories d'un concept (v4 uniquement)."""
        if self._backend != "sqlite" or self._schema_version < 4:
            return []
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT cat.label FROM categories cat "
            "JOIN concept_categories cc ON cc.categorie_id = cat.id "
            "WHERE cc.concept_id = ?",
            (concept_id,),
        )
        return [row[0] for row in cur.fetchall() if row[0]]

    def decoder_multext(self, tag: str) -> dict[str, str]:
        """Decode un tag multext en traits lisibles. Delegue a _multext.py."""
        return _decoder_multext(tag)

    # --- Methodes NP unifiees (v4 : alias vers methodes standard) ---

    def info_nom_propre(self, mot: str) -> list[dict[str, Any]]:
        """Retourne les entrees noms propres (SQLite uniquement)."""
        if self._backend != "sqlite":
            return []
        conn = self._get_conn()

        if self._schema_version >= 4:
            # NP unifies : chercher dans formes+lemmes avec cgram="NOM PROPRE"
            cur = conn.execute(
                "SELECT f.id, f.ortho, f.multext, f.phone, f.phone_reversed, "
                "f.nb_syllabes, f.syllabes, f.freq_opensubs, f.source, "
                "l.id AS lemme_id, l.lemme, l.cgram, l.genre, "
                "l.etymologie, l.age "
                "FROM formes f "
                "JOIN lemmes l ON f.lemme_id = l.id "
                "WHERE l.cgram = 'NOM PROPRE' AND f.ortho = ? COLLATE NOCASE",
                (mot,),
            )
            return [dict(row) for row in cur.fetchall()]

        try:
            cur = conn.execute(
                "SELECT * FROM noms_propres WHERE lemme = ? COLLATE NOCASE",
                (mot,),
            )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    def rechercher_nom_propre(self, pattern: str, limite: int = 50) -> list[dict[str, Any]]:
        """Recherche regex sur les noms propres."""
        if self._backend != "sqlite":
            return []
        import re as _re
        try:
            regex = _re.compile(pattern, _re.IGNORECASE)
        except _re.error:
            return []
        conn = self._get_conn()
        conn.create_function("REGEXP", 2, lambda p, s: bool(regex.search(s)) if s else False)

        if self._schema_version >= 4:
            try:
                cur = conn.execute(
                    "SELECT f.ortho AS lemme, l.cgram, f.phone, f.freq_opensubs, "
                    "l.etymologie, l.age "
                    "FROM formes f "
                    "JOIN lemmes l ON f.lemme_id = l.id "
                    "WHERE l.cgram = 'NOM PROPRE' AND f.ortho REGEXP ? LIMIT ?",
                    (pattern, limite),
                )
                return [dict(row) for row in cur.fetchall()]
            except sqlite3.OperationalError:
                return []

        try:
            cur = conn.execute(
                "SELECT * FROM noms_propres WHERE lemme REGEXP ? LIMIT ?",
                (pattern, limite),
            )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    def existe_nom_propre(self, mot: str) -> bool:
        """Test d'existence dans les noms propres."""
        if self._backend != "sqlite":
            return False
        conn = self._get_conn()

        if self._schema_version >= 4:
            cur = conn.execute(
                "SELECT 1 FROM formes f "
                "JOIN lemmes l ON f.lemme_id = l.id "
                "WHERE l.cgram = 'NOM PROPRE' AND f.ortho = ? COLLATE NOCASE LIMIT 1",
                (mot,),
            )
            return cur.fetchone() is not None

        try:
            cur = conn.execute(
                "SELECT 1 FROM noms_propres WHERE lemme = ? COLLATE NOCASE LIMIT 1",
                (mot,),
            )
            return cur.fetchone() is not None
        except sqlite3.OperationalError:
            return False

    def phone_nom_propre(self, mot: str) -> str | None:
        """Retourne la phonetique d'un nom propre."""
        if self._backend != "sqlite":
            return None
        conn = self._get_conn()

        if self._schema_version >= 4:
            cur = conn.execute(
                "SELECT f.phone FROM formes f "
                "JOIN lemmes l ON f.lemme_id = l.id "
                "WHERE l.cgram = 'NOM PROPRE' AND f.ortho = ? COLLATE NOCASE "
                "AND f.phone IS NOT NULL AND f.phone != '' LIMIT 1",
                (mot,),
            )
            row = cur.fetchone()
            return row[0] if row else None

        try:
            cur = conn.execute(
                "SELECT phone FROM noms_propres "
                "WHERE lemme = ? COLLATE NOCASE AND phone IS NOT NULL AND phone != '' "
                "LIMIT 1",
                (mot,),
            )
            row = cur.fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None

    def homophones_nom_propre(self, phone: str) -> list[dict[str, Any]]:
        """Noms propres ayant cette prononciation."""
        if self._backend != "sqlite":
            return []
        conn = self._get_conn()

        if self._schema_version >= 4:
            cur = conn.execute(
                "SELECT f.ortho AS lemme, l.cgram, f.phone "
                "FROM formes f "
                "JOIN lemmes l ON f.lemme_id = l.id "
                "WHERE l.cgram = 'NOM PROPRE' AND f.phone = ?",
                (phone,),
            )
            return [dict(row) for row in cur.fetchall()]

        try:
            cur = conn.execute(
                "SELECT * FROM noms_propres WHERE phone = ?",
                (phone,),
            )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    def rimes_nom_propre(self, phone_suffix: str, limite: int = 50) -> list[dict[str, Any]]:
        """Noms propres dont la phonetique se termine par le suffixe donne."""
        if self._backend != "sqlite":
            return []
        conn = self._get_conn()

        if self._schema_version >= 4:
            cur = conn.execute(
                "SELECT f.ortho AS lemme, l.cgram, f.phone "
                "FROM formes f "
                "JOIN lemmes l ON f.lemme_id = l.id "
                "WHERE l.cgram = 'NOM PROPRE' "
                "AND f.phone LIKE ? AND f.phone IS NOT NULL AND f.phone != '' "
                "LIMIT ?",
                (f"%{phone_suffix}", limite),
            )
            return [dict(row) for row in cur.fetchall()]

        try:
            cur = conn.execute(
                "SELECT * FROM noms_propres "
                "WHERE phone LIKE ? AND phone IS NOT NULL AND phone != '' "
                "LIMIT ?",
                (f"%{phone_suffix}", limite),
            )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

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
