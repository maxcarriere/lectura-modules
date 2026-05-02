"""Modele de langue n-gram avec backoff depuis SQLite.

Stocke des n-grams (1 a 4) avec probabilites log10 et poids de backoff
calcules par lissage Kneser-Ney modifie. Utilisé pour desambiguiser les
candidats de correction par le contexte.

Schema SQLite attendu :
    unigrams(word TEXT PK, count INT, prob REAL)
    bigrams(w1 TEXT, w2 TEXT, count INT, prob REAL, backoff REAL, PK(w1,w2))
    trigrams(w1, w2, w3, count, prob, backoff, PK(w1,w2,w3))
    fourgrams(w1, w2, w3, w4, count, prob, backoff, PK(w1,w2,w3,w4))
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any

# Log-probabilite pour les mots inconnus du LM
_LOG_PROB_OOV = -5.0

# Tokens speciaux
BOS = "<s>"
EOS = "</s>"


class ScorerNgram:
    """Modele de langue n-gram avec backoff, stocke dans SQLite."""

    __slots__ = ("_conn", "_max_order", "_cache")

    def __init__(self, db_path: str) -> None:
        """Ouvre le fichier SQLite en lecture seule.

        Args:
            db_path: Chemin vers le fichier ngram.db.
        """
        uri = f"file:{db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("PRAGMA cache_size=-8192")  # 8 Mo de cache
        self._max_order = self._detecter_ordre()
        self._cache: dict[tuple, float | None] = {}

    def _detecter_ordre(self) -> int:
        """Detecte l'ordre max du modele (4, 3 ou 2)."""
        cursor = self._conn.cursor()
        for table, order in [("fourgrams", 4), ("trigrams", 3), ("bigrams", 2)]:
            try:
                cursor.execute(
                    f"SELECT 1 FROM {table} LIMIT 1"  # noqa: S608
                )
                if cursor.fetchone() is not None:
                    return order
            except sqlite3.OperationalError:
                continue
        return 1

    def close(self) -> None:
        """Ferme la connexion SQLite."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # --- Lookups unitaires ---

    def _prob_unigram(self, w: str) -> float | None:
        """Log-prob P(w). Retourne None si absent."""
        key = ("u", w)
        if key in self._cache:
            return self._cache[key]
        row = self._conn.execute(
            "SELECT prob FROM unigrams WHERE word = ?", (w,)
        ).fetchone()
        val = row[0] if row else None
        self._cache[key] = val
        return val

    def _prob_bigram(self, w1: str, w2: str) -> float | None:
        """Log-prob P(w2|w1). Retourne None si absent."""
        key = ("b", w1, w2)
        if key in self._cache:
            return self._cache[key]
        row = self._conn.execute(
            "SELECT prob FROM bigrams WHERE w1 = ? AND w2 = ?", (w1, w2)
        ).fetchone()
        val = row[0] if row else None
        self._cache[key] = val
        return val

    def _backoff_bigram(self, w1: str) -> float:
        """Poids de backoff alpha(w1) pour bigram -> unigram."""
        key = ("bb", w1)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]
        row = self._conn.execute(
            "SELECT backoff FROM bigrams WHERE w1 = ? LIMIT 1", (w1,)
        ).fetchone()
        val = row[0] if row else 0.0
        self._cache[key] = val
        return val

    def _prob_trigram(self, w1: str, w2: str, w3: str) -> float | None:
        """Log-prob P(w3|w1,w2). Retourne None si absent."""
        key = ("t", w1, w2, w3)
        if key in self._cache:
            return self._cache[key]
        try:
            row = self._conn.execute(
                "SELECT prob FROM trigrams WHERE w1 = ? AND w2 = ? AND w3 = ?",
                (w1, w2, w3),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        val = row[0] if row else None
        self._cache[key] = val
        return val

    def _backoff_trigram(self, w1: str, w2: str) -> float:
        """Poids de backoff alpha(w1,w2) pour trigram -> bigram."""
        key = ("bt", w1, w2)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]
        try:
            row = self._conn.execute(
                "SELECT backoff FROM trigrams WHERE w1 = ? AND w2 = ? LIMIT 1",
                (w1, w2),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        val = row[0] if row else 0.0
        self._cache[key] = val
        return val

    def _prob_fourgram(
        self, w1: str, w2: str, w3: str, w4: str,
    ) -> float | None:
        """Log-prob P(w4|w1,w2,w3). Retourne None si absent."""
        key = ("f", w1, w2, w3, w4)
        if key in self._cache:
            return self._cache[key]
        try:
            row = self._conn.execute(
                "SELECT prob FROM fourgrams"
                " WHERE w1 = ? AND w2 = ? AND w3 = ? AND w4 = ?",
                (w1, w2, w3, w4),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        val = row[0] if row else None
        self._cache[key] = val
        return val

    def _backoff_fourgram(self, w1: str, w2: str, w3: str) -> float:
        """Poids de backoff alpha(w1,w2,w3) pour fourgram -> trigram."""
        key = ("bf", w1, w2, w3)
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]
        try:
            row = self._conn.execute(
                "SELECT backoff FROM fourgrams"
                " WHERE w1 = ? AND w2 = ? AND w3 = ? LIMIT 1",
                (w1, w2, w3),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        val = row[0] if row else 0.0
        self._cache[key] = val
        return val

    # --- Scoring avec backoff ---

    def score_mot_en_contexte(
        self,
        mot: str,
        contexte_gauche: list[str],
    ) -> float:
        """Calcule log10 P(mot | contexte_gauche) avec backoff n-gram.

        Args:
            mot: Le mot a scorer.
            contexte_gauche: Liste des mots precedents (les plus recents en dernier).

        Returns:
            Log10-probabilite du mot en contexte.
        """
        w = mot.lower()
        ctx = [c.lower() for c in contexte_gauche]

        # Essayer 4-gram
        if self._max_order >= 4 and len(ctx) >= 3:
            w1, w2, w3 = ctx[-3], ctx[-2], ctx[-1]
            p = self._prob_fourgram(w1, w2, w3, w)
            if p is not None:
                return p
            # Backoff vers trigram
            bo = self._backoff_fourgram(w1, w2, w3)
            return bo + self._score_trigram(ctx[-2], ctx[-1], w)

        # Essayer 3-gram
        if self._max_order >= 3 and len(ctx) >= 2:
            return self._score_trigram(ctx[-2], ctx[-1], w)

        # Essayer 2-gram
        if self._max_order >= 2 and len(ctx) >= 1:
            return self._score_bigram(ctx[-1], w)

        # Unigram
        return self._score_unigram(w)

    def _score_trigram(self, w1: str, w2: str, w3: str) -> float:
        """Score trigram avec backoff vers bigram."""
        p = self._prob_trigram(w1, w2, w3)
        if p is not None:
            return p
        bo = self._backoff_trigram(w1, w2)
        return bo + self._score_bigram(w2, w3)

    def _score_bigram(self, w1: str, w2: str) -> float:
        """Score bigram avec backoff vers unigram."""
        p = self._prob_bigram(w1, w2)
        if p is not None:
            return p
        bo = self._backoff_bigram(w1)
        return bo + self._score_unigram(w2)

    def _score_unigram(self, w: str) -> float:
        """Score unigram, OOV si absent."""
        p = self._prob_unigram(w)
        if p is not None:
            return p
        return _LOG_PROB_OOV

    # --- API de haut niveau ---

    def score_phrase(self, mots: list[str]) -> float:
        """Log-probabilite totale d'une phrase (somme des log-probs).

        Ajoute <s> au debut et </s> a la fin.

        Args:
            mots: Liste de mots de la phrase.

        Returns:
            Somme des log10-probabilites.
        """
        if not mots:
            return 0.0
        tokens = [BOS] + [m.lower() for m in mots] + [EOS]
        total = 0.0
        for i in range(1, len(tokens)):
            ctx = tokens[max(0, i - (self._max_order - 1)):i]
            total += self.score_mot_en_contexte(tokens[i], ctx)
        return total

    def scorer_candidats(
        self,
        candidats: list[str],
        contexte_gauche: list[str],
        contexte_droit: list[str],
    ) -> list[tuple[str, float]]:
        """Score les candidats par le modele de langue.

        Utilise le contexte gauche pour P(candidat | ctx_gauche) et
        le contexte droit pour P(ctx_droit[0] | ..., candidat).

        Args:
            candidats: Liste de mots candidats.
            contexte_gauche: Mots avant la position (les plus recents en dernier).
            contexte_droit: Mots apres la position.

        Returns:
            Liste de (candidat, score_lm) triee par score decroissant.
        """
        scores: list[tuple[str, float]] = []
        for c in candidats:
            # Score gauche : P(c | ctx_gauche)
            s = self.score_mot_en_contexte(c, contexte_gauche)
            # Score droit : P(ctx_droit[0] | ..., c)
            if contexte_droit:
                ctx_pour_droit = contexte_gauche + [c]
                s += self.score_mot_en_contexte(
                    contexte_droit[0], ctx_pour_droit,
                )
            scores.append((c, s))
        scores.sort(key=lambda x: -x[1])
        return scores
