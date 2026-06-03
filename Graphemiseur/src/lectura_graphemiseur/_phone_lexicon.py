"""PhoneLexicon autonome pour le Graphemiseur V6.

Fournit un acces direct a la DB SQLite pour les phone_lex_features (28d)
et les candidats lex_select, sans dependance sur LexiqueLite ou le Correcteur.

Supporte deux schemas :
  - Table ``phones`` (DB dediee phone_lexicon.db)
  - Table ``lexique`` (DB correcteur, meme colonnes)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class PhoneLexicon:
    """Lexique phonetique autonome pour le P2G V6.

    Charge depuis une DB SQLite contenant soit une table ``phones``
    (DB dediee) soit une table ``lexique`` (DB correcteur).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row

        # Detecter la table disponible
        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "phones" in tables:
            self._table = "phones"
        elif "lexique" in tables:
            self._table = "lexique"
        else:
            raise ValueError(
                f"DB {db_path} ne contient ni table 'phones' ni 'lexique'"
            )

        # Precharger phone_set et phone_to_best
        self.phone_set: set[str] = set()
        self.phone_to_best: dict[str, tuple[str, float]] = {}
        self._preload()

        logger.info(
            "PhoneLexicon charge: %d phones depuis %s (table=%s)",
            len(self.phone_set), db_path, self._table,
        )

    def _preload(self) -> None:
        cursor = self._conn.execute(
            f"SELECT phone, ortho, freq FROM {self._table} "
            "WHERE phone != '' AND phone IS NOT NULL"
        )
        for row in cursor:
            phone = (row[0] or "").strip()
            if not phone:
                continue
            ortho = row[1] or ""
            freq = row[2] or 0.0
            self.phone_set.add(phone)
            prev = self.phone_to_best.get(phone)
            if prev is None or freq > prev[1]:
                self.phone_to_best[phone] = (ortho, freq)

    def exists(self, phone: str) -> bool:
        return phone in self.phone_set

    def best_ortho(self, phone: str) -> str | None:
        entry = self.phone_to_best.get(phone)
        return entry[0] if entry else None

    def best_freq(self, phone: str) -> float:
        entry = self.phone_to_best.get(phone)
        return entry[1] if entry else 0.0

    def all_entries(self, phone: str) -> list[dict]:
        """Retourne toutes les entrees lexicales pour un phone donne.

        Returns:
            list[dict] avec cles: ortho, cgram, freq (+ genre, nombre si disponibles)
        """
        if not phone:
            return []
        cursor = self._conn.execute(
            f"SELECT * FROM {self._table} WHERE phone = ?",
            (phone,),
        )
        results: list[dict] = []
        for row in cursor:
            d = dict(row)
            # Normaliser les cles attendues par inference_onnx_v2
            d.setdefault("ortho", "")
            d.setdefault("cgram", "")
            d.setdefault("freq", 0.0)
            results.append(d)
        return results

    def all_entries_with_perturbations(
        self, phone: str, k_max: int = 20,
    ) -> tuple[list[dict], dict[str, str]]:
        """Version basique : exact only + dedup (pas de perturbations CTC).

        Le STT peut surcharger cette methode avec les perturbations CTC.
        Pour le Graphemiseur autonome, les entrees exactes suffisent.

        Returns:
            entries: liste de dicts dedupliques par ortho.lower(), tries par freq desc
            resolved_map: {ortho.lower(): "exact"}
        """
        raw = self.all_entries(phone)
        if not raw:
            return [], {}

        # Dedupliquer par ortho.lower() (garder la plus haute freq)
        by_ortho: dict[str, dict] = {}
        for e in raw:
            key = e["ortho"].lower()
            if key not in by_ortho or (e.get("freq", 0) or 0) > (
                by_ortho[key].get("freq", 0) or 0
            ):
                by_ortho[key] = e

        unique = sorted(
            by_ortho.values(), key=lambda e: -(e.get("freq", 0) or 0),
        )[:k_max]

        resolved_map = {e["ortho"].lower(): "exact" for e in unique}
        return unique, resolved_map

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"PhoneLexicon({self._db_path!r}, {len(self.phone_set)} phones)"
