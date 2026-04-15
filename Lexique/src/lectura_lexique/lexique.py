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
from lectura_lexique._types import EntreeLexicale
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
            query = (
                f"SELECT ortho, lemme, cgram, personne, nombre, mode, temps, "
                f"freq_opensubs AS freq, phone "
                f"FROM {self._table} "
                f"WHERE phone IN ({placeholders}) "
                f"AND cgram IN ('VER', 'AUX') "
                f"AND personne = ?"
            )
            params: list[Any] = list(phones) + [personne]
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
            query = (
                f"SELECT * FROM {self._table} "  # noqa: S608
                f"WHERE nb_syllabes = ?"
            )
            params: list[Any] = [n]
            if cgram is not None:
                query += " AND cgram LIKE ?"
                params.append(f"{cgram}%")
            query += f" ORDER BY freq_opensubs DESC LIMIT ?"
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
        """Definitions multiples depuis la table definitions.

        Retourne [] si la table est absente (compatibilite anciennes BDD).
        Ordonne par cgram, sens_num.
        """
        if self._backend != "sqlite":
            return []
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
            results: list[dict[str, Any]] = []
            for row in rows:
                d: dict[str, Any] = dict(row)
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
            cur = conn.execute(
                f"SELECT * FROM {self._table} "  # noqa: S608
                f"WHERE {col} REGEXP ? "
                f"ORDER BY freq_opensubs DESC LIMIT ?",
                (pattern, limite),
            )
            colonnes = [desc[0] for desc in cur.description]
            mapping = resoudre_colonnes(colonnes)
            results: list[dict[str, Any]] = []
            seen: set[str] = set()
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
        limite: int = 100,
    ) -> list[dict[str, Any]]:
        """Filtre multi-critere sur l'ensemble du lexique."""
        if self._backend == "sqlite":
            conn = self._get_conn()
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
            age_min=age_min,
            age_max=age_max,
            illustrable_min=illustrable_min,
            categorie=categorie,
            has_age=has_age,
            has_phone=has_phone,
            limite=limite,
        )

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
            conditions, params = self._build_filter_clauses(
                cgram=cgram, genre=genre, nombre=nombre,
                freq_min=freq_min, freq_max=freq_max, nb_syllabes=nb_syllabes,
                age_min=age_min, age_max=age_max,
                illustrable_min=illustrable_min, categorie=categorie,
                has_age=has_age, has_phone=has_phone,
            )
            where = " AND ".join(conditions) if conditions else "1=1"
            query = f"SELECT COUNT(*) FROM {self._table} WHERE {where}"  # noqa: S608
            cur = conn.execute(query, params)
            row = cur.fetchone()
            return int(row[0]) if row else 0

        # CSV/TSV : compter via filtrer() sans limite
        ortho_index = self._get_index_ortho()
        all_entries = (e for entries in ortho_index.values() for e in entries)
        return len(_recherche.filtrer(
            all_entries,
            cgram=cgram, genre=genre, nombre=nombre,
            freq_min=freq_min, freq_max=freq_max, nb_syllabes=nb_syllabes,
            age_min=age_min, age_max=age_max,
            illustrable_min=illustrable_min, categorie=categorie,
            has_age=has_age, has_phone=has_phone,
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
            conditions, params = self._build_filter_clauses(**filter_kwargs)
            where = " AND ".join(conditions) if conditions else "1=1"

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
                params.append(limite)

            cur = conn.execute(query, params)
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
            cur2 = conn.execute(
                f"SELECT * FROM {self._table} "  # noqa: S608
                f"WHERE ortho IN ({placeholders}) "
                f"ORDER BY freq_opensubs DESC LIMIT ?",
                candidats + [limite],
            )
            colonnes = [desc[0] for desc in cur2.description]
            mapping = resoudre_colonnes(colonnes)
            results: list[dict[str, Any]] = []
            seen: set[str] = set()
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

    # --- Methodes : Noms propres ---

    def info_nom_propre(self, mot: str) -> list[dict[str, Any]]:
        """Retourne les entrees noms propres (SQLite uniquement)."""
        if self._backend != "sqlite":
            return []
        conn = self._get_conn()
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
