"""Backend SQLite leger pour le Correcteur.

Charge ``lexique_correcteur.db`` — un SQLite de ~50 Mo contenant
uniquement les colonnes necessaires au Correcteur (vs ~2 Go pour le
lexique complet).

Expose la meme interface que ``lectura_lexique.Lexique`` :
existe, info, frequence, phone_de, homophones, formes_de,
lemme_de, conjuguer.

Consommation memoire minimale : seul le set de formes (existe) est
en RAM (~50 Mo). Toutes les autres requetes passent par SQLite.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from lectura_correcteur._multext import decoder_multext


class LexiqueLite:
    """Lexique SQLite leger pour le Correcteur.

    Utilise un fichier ``lexique_correcteur.db`` contenant une seule
    table ``lexique`` avec index sur ortho, phone et lemme.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._path), check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

        # Set de formes en RAM pour existe() O(1)
        cur = self._conn.execute("SELECT DISTINCT lower(ortho) FROM lexique")
        self._formes: frozenset[str] = frozenset(row[0] for row in cur)

    def _decode_row(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convertit une Row SQLite en dict avec traits multext decodes."""
        entry: dict[str, Any] = dict(row)
        multext = entry.get("multext") or ""
        if multext:
            traits = decoder_multext(multext)
            entry["genre"] = entry.get("genre") or traits.get("genre", "")
            entry["nombre"] = traits.get("nombre", entry.get("nombre", ""))
            entry["mode"] = traits.get("mode", "")
            entry["temps"] = traits.get("temps", "")
            entry["personne"] = traits.get("personne", "")
        else:
            entry.setdefault("mode", "")
            entry.setdefault("temps", "")
            entry.setdefault("personne", "")
        return entry

    # ── API publique ──────────────────────────────────────────────────

    def existe(self, mot: str) -> bool:
        """Test d'appartenance O(1) via le set en memoire."""
        return mot.lower() in self._formes

    def info(self, mot: str) -> list[dict[str, Any]]:
        """Entrees lexicales completes pour un mot."""
        cur = self._conn.execute(
            "SELECT * FROM lexique WHERE ortho = ? COLLATE NOCASE",
            (mot,),
        )
        return [self._decode_row(row) for row in cur.fetchall()]

    def frequence(self, mot: str) -> float:
        """Frequence maximale parmi toutes les entrees du mot."""
        cur = self._conn.execute(
            "SELECT MAX(freq) FROM lexique WHERE ortho = ? COLLATE NOCASE",
            (mot,),
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def phone_de(self, mot: str) -> str | None:
        """Prononciation la plus frequente."""
        cur = self._conn.execute(
            "SELECT phone FROM lexique "
            "WHERE ortho = ? COLLATE NOCASE AND phone != '' "
            "ORDER BY freq DESC LIMIT 1",
            (mot,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def homophones(self, phone: str) -> list[dict[str, Any]]:
        """Tous les mots ayant cette prononciation."""
        cur = self._conn.execute(
            "SELECT * FROM lexique WHERE phone = ?",
            (phone,),
        )
        return [self._decode_row(row) for row in cur.fetchall()]

    def formes_de(self, lemme: str, cgram: str | None = None) -> list[dict[str, Any]]:
        """Toutes les formes flechies d'un lemme, avec filtre POS optionnel."""
        if cgram is not None:
            cur = self._conn.execute(
                "SELECT * FROM lexique "
                "WHERE lemme = ? COLLATE NOCASE AND cgram LIKE ?",
                (lemme, f"{cgram}%"),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM lexique WHERE lemme = ? COLLATE NOCASE",
                (lemme,),
            )

        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in cur.fetchall():
            entry = self._decode_row(row)
            ortho = entry.get("ortho", "")
            key = ortho.lower()
            if ortho and key not in seen:
                seen.add(key)
                result.append(entry)
        return result

    def lemme_de(self, mot: str) -> str | None:
        """Lemme le plus frequent parmi les entrees d'un mot."""
        entries = self.info(mot)
        if not entries:
            return None

        freq_par_lemme: dict[str, float] = {}
        for e in entries:
            lemme = e.get("lemme", "")
            if not lemme:
                continue
            freq_par_lemme[lemme] = (
                freq_par_lemme.get(lemme, 0.0) + float(e.get("freq", 0))
            )

        if not freq_par_lemme:
            return None

        return max(freq_par_lemme, key=lambda k: freq_par_lemme[k])

    def conjuguer(self, verbe: str) -> dict[str, dict[str, dict[str, str]]]:
        """Table de conjugaison d'un verbe.

        Retour : {mode: {temps: {"1s": forme, "2s": forme, ...}}}
        """
        cur = self._conn.execute(
            "SELECT * FROM lexique "
            "WHERE lemme = ? COLLATE NOCASE AND cgram IN ('VER', 'AUX')",
            (verbe,),
        )

        table: dict[str, dict[str, dict[str, str]]] = {}

        for row in cur.fetchall():
            entry = self._decode_row(row)

            mode = entry.get("mode", "")
            temps = entry.get("temps", "")
            ortho = entry.get("ortho", "")

            if not mode or not ortho:
                continue

            personne = entry.get("personne", "")
            nombre = entry.get("nombre", "")
            if isinstance(nombre, str) and len(nombre) > 1:
                nombre = nombre[0]

            if personne and nombre:
                cle = f"{personne}{nombre}"
            elif personne:
                cle = personne
            else:
                cle = ""

            if mode not in table:
                table[mode] = {}
            if temps not in table[mode]:
                table[mode][temps] = {}
            if cle not in table[mode][temps]:
                table[mode][temps][cle] = ortho

        return table

    def close(self) -> None:
        """Ferme la connexion SQLite."""
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
        self._formes = frozenset()
