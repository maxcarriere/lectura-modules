"""Scorer n-gram POS pour la desambiguation grammaticale.

Charge pos_ngram.db (n-grammes POS et POS+Morpho jusqu'a 4-grammes) et
expose des methodes pour scorer des sequences POS et choisir le meilleur
POS a une position.

Le modele utilise le lissage Kneser-Ney modifie avec backoff structure :
4-gram → 3-gram → 2-gram → 1-gram.

Utilisation principale :
- Arbitrer entre homophones en comparant les sequences POS resultantes
  (ex: "il a raison" → PRO AUX NOM vs "il a raison" → PRO PRE NOM)
- Valider les corrections grammaticales
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path


# Log-probabilite par defaut pour n-grammes inconnus (tres improbable)
_DEFAULT_LOGP = -15.0


class PosNgram:
    """Scorer n-gram POS (unigrammes, bigrammes, trigrammes, 4-grammes).

    Charge les tables pos_* et pm_* depuis pos_ngram.db.
    Supporte le backoff Kneser-Ney (logp + backoff weight).
    """

    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._cur = self._conn.cursor()

        # Cache pour eviter les requetes repetees
        self._cache_tri: dict[tuple[str, str, str], float] = {}
        self._cache_bi: dict[tuple[str, str], float] = {}
        self._cache_uni: dict[str, float] = {}
        self._cache_pm_four: dict[tuple[str, str, str, str], float] = {}
        self._cache_pm_tri: dict[tuple[str, str, str], float] = {}
        self._cache_pm_bi: dict[tuple[str, str], float] = {}
        self._cache_pm_uni: dict[str, float] = {}

        # Backoff weights
        self._cache_pm_tri_bo: dict[tuple[str, str, str], float] = {}
        self._cache_pm_bi_bo: dict[tuple[str, str], float] = {}
        self._cache_pm_uni_bo: dict[str, float] = {}

        # Detecter les tables disponibles
        self._cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        self._tables = {row[0] for row in self._cur.fetchall()}

        # Detecter si le schema a des colonnes backoff
        self._has_backoff = self._detect_backoff()

        # Detecter si 4-grammes sont disponibles
        self._has_fourgrams = "pm_fourgrams" in self._tables

        # Pre-charger les unigrammes POS (petit : ~20 entrees)
        if "pos_unigrams" in self._tables:
            self._cur.execute("SELECT w1, logp FROM pos_unigrams")
            for w, lp in self._cur.fetchall():
                self._cache_uni[w] = lp

        # Pre-charger les unigrammes POS+Morpho (petit : ~115 entrees)
        if "pm_unigrams" in self._tables:
            if self._has_backoff:
                self._cur.execute("SELECT w1, logp, backoff FROM pm_unigrams")
                for w, lp, bo in self._cur.fetchall():
                    self._cache_pm_uni[w] = lp
                    self._cache_pm_uni_bo[w] = bo
            else:
                self._cur.execute("SELECT w1, logp FROM pm_unigrams")
                for w, lp in self._cur.fetchall():
                    self._cache_pm_uni[w] = lp

        # Pre-charger les bigrammes PM (taille raisonnable)
        if "pm_bigrams" in self._tables:
            if self._has_backoff:
                self._cur.execute("SELECT w1, w2, logp, backoff FROM pm_bigrams")
                for w1, w2, lp, bo in self._cur.fetchall():
                    self._cache_pm_bi[(w1, w2)] = lp
                    self._cache_pm_bi_bo[(w1, w2)] = bo
            else:
                self._cur.execute("SELECT w1, w2, logp FROM pm_bigrams")
                for w1, w2, lp in self._cur.fetchall():
                    self._cache_pm_bi[(w1, w2)] = lp

        # Pre-charger les trigrammes PM (taille raisonnable ~20K)
        if "pm_trigrams" in self._tables:
            if self._has_backoff:
                self._cur.execute("SELECT w1, w2, w3, logp, backoff FROM pm_trigrams")
                for w1, w2, w3, lp, bo in self._cur.fetchall():
                    self._cache_pm_tri[(w1, w2, w3)] = lp
                    self._cache_pm_tri_bo[(w1, w2, w3)] = bo
            else:
                self._cur.execute("SELECT w1, w2, w3, logp FROM pm_trigrams")
                for w1, w2, w3, lp in self._cur.fetchall():
                    self._cache_pm_tri[(w1, w2, w3)] = lp

        # Pre-charger les 4-grammes PM (taille raisonnable ~66K)
        if self._has_fourgrams:
            self._cur.execute("SELECT w1, w2, w3, w4, logp FROM pm_fourgrams")
            for w1, w2, w3, w4, lp in self._cur.fetchall():
                self._cache_pm_four[(w1, w2, w3, w4)] = lp

    def _detect_backoff(self) -> bool:
        """Detecte si le schema inclut des colonnes backoff."""
        if "pm_bigrams" not in self._tables:
            return False
        try:
            self._cur.execute("PRAGMA table_info(pm_bigrams)")
            cols = {row[1] for row in self._cur.fetchall()}
            return "backoff" in cols
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Lookups de base POS
    # ------------------------------------------------------------------

    def logp_unigram(self, pos: str) -> float:
        """Log-probabilite P(pos)."""
        return self._cache_uni.get(pos, _DEFAULT_LOGP)

    def logp_bigram(self, pos1: str, pos2: str) -> float:
        """Log-probabilite P(pos2 | pos1)."""
        key = (pos1, pos2)
        if key in self._cache_bi:
            return self._cache_bi[key]
        self._cur.execute(
            "SELECT logp FROM pos_bigrams WHERE w1=? AND w2=?",
            (pos1, pos2),
        )
        row = self._cur.fetchone()
        val = row[0] if row else _DEFAULT_LOGP
        self._cache_bi[key] = val
        return val

    def logp_trigram(self, pos1: str, pos2: str, pos3: str) -> float:
        """Log-probabilite P(pos3 | pos1, pos2)."""
        key = (pos1, pos2, pos3)
        if key in self._cache_tri:
            return self._cache_tri[key]
        self._cur.execute(
            "SELECT logp FROM pos_trigrams WHERE w1=? AND w2=? AND w3=?",
            (pos1, pos2, pos3),
        )
        row = self._cur.fetchone()
        val = row[0] if row else _DEFAULT_LOGP
        self._cache_tri[key] = val
        return val

    # ------------------------------------------------------------------
    # Lookups POS+Morpho avec backoff Kneser-Ney
    # ------------------------------------------------------------------

    def logp_pm_unigram(self, pm: str) -> float:
        """Log-probabilite unigramme POS+Morpho P(pm)."""
        return self._cache_pm_uni.get(pm, _DEFAULT_LOGP)

    def logp_pm_bigram(self, pm1: str, pm2: str) -> float:
        """Log-probabilite bigramme POS+Morpho P(pm2 | pm1).

        Avec backoff KN : si le bigram existe, retourne logp.
        Sinon, retourne backoff(pm1) + logp_unigram(pm2).
        """
        key = (pm1, pm2)
        if key in self._cache_pm_bi:
            return self._cache_pm_bi[key]

        # Bigram absent — backoff vers unigram
        if self._has_backoff:
            bo = self._cache_pm_uni_bo.get(pm1, 0.0)
            uni = self.logp_pm_unigram(pm2)
            val = bo + uni
        else:
            val = _DEFAULT_LOGP
        self._cache_pm_bi[key] = val
        return val

    def logp_pm_trigram(self, pm1: str, pm2: str, pm3: str) -> float:
        """Log-probabilite trigramme POS+Morpho P(pm3 | pm1, pm2).

        Avec backoff KN : si le trigram existe, retourne logp.
        Sinon, retourne backoff(pm1,pm2) + logp_bigram(pm2, pm3).
        """
        key = (pm1, pm2, pm3)
        if key in self._cache_pm_tri:
            return self._cache_pm_tri[key]

        # Trigram absent — backoff vers bigram
        if self._has_backoff:
            bo = self._cache_pm_bi_bo.get((pm1, pm2), 0.0)
            bi = self.logp_pm_bigram(pm2, pm3)
            val = bo + bi
        else:
            val = _DEFAULT_LOGP
        self._cache_pm_tri[key] = val
        return val

    def logp_pm_fourgram(self, pm1: str, pm2: str, pm3: str, pm4: str) -> float:
        """Log-probabilite 4-gramme POS+Morpho P(pm4 | pm1, pm2, pm3).

        Avec backoff KN : si le 4-gram existe, retourne logp.
        Sinon, retourne backoff(pm1,pm2,pm3) + logp_trigram(pm2, pm3, pm4).
        """
        if not self._has_fourgrams:
            return self.logp_pm_trigram(pm2, pm3, pm4)

        key = (pm1, pm2, pm3, pm4)
        if key in self._cache_pm_four:
            return self._cache_pm_four[key]

        # 4-gram absent — backoff vers trigram
        if self._has_backoff:
            bo = self._cache_pm_tri_bo.get((pm1, pm2, pm3), 0.0)
            tri = self.logp_pm_trigram(pm2, pm3, pm4)
            val = bo + tri
        else:
            val = self.logp_pm_trigram(pm2, pm3, pm4)
        self._cache_pm_four[key] = val
        return val

    # ------------------------------------------------------------------
    # Scoring de sequences
    # ------------------------------------------------------------------

    def score_sequence(self, pos_tags: list[str]) -> float:
        """Score (log-probabilite) d'une sequence POS complete.

        Utilise les trigrammes avec backoff vers bigrammes.
        """
        if not pos_tags:
            return 0.0

        padded = [self.BOS, self.BOS] + pos_tags + [self.EOS]
        score = 0.0

        for i in range(2, len(padded)):
            tri = self.logp_trigram(padded[i - 2], padded[i - 1], padded[i])
            if tri > _DEFAULT_LOGP + 1.0:
                score += tri
            else:
                # Backoff : bigramme avec penalite
                bi = self.logp_bigram(padded[i - 1], padded[i])
                score += bi - 1.0  # penalite de backoff
        return score

    def score_position(
        self,
        pos_tags: list[str],
        idx: int,
        candidate_pos: str,
    ) -> float:
        """Score local (trigrammes) si on met candidate_pos a la position idx.

        Calcule la somme des log-probabilites des trigrammes affectes
        par le changement a la position idx (ceux qui contiennent idx).
        """
        if idx < 0 or idx >= len(pos_tags):
            return _DEFAULT_LOGP

        # Construire la sequence avec padding BOS/EOS
        padded = [self.BOS, self.BOS] + list(pos_tags) + [self.EOS]
        real_idx = idx + 2  # offset pour le padding

        # Remplacer temporairement
        old = padded[real_idx]
        padded[real_idx] = candidate_pos

        # Trigrammes affectes : ceux dont la fenetre inclut real_idx
        score = 0.0
        for i in range(max(2, real_idx), min(len(padded), real_idx + 3)):
            score += self.logp_trigram(padded[i - 2], padded[i - 1], padded[i])

        # Restaurer
        padded[real_idx] = old
        return score

    def best_pos_at(
        self,
        pos_tags: list[str],
        idx: int,
        candidates: list[str],
    ) -> tuple[str, float]:
        """Meilleur POS a la position idx parmi les candidats.

        Returns:
            (best_pos, score) — le candidat avec le meilleur score local.
        """
        best_pos = pos_tags[idx] if idx < len(pos_tags) else candidates[0]
        best_score = _DEFAULT_LOGP

        for cand in candidates:
            score = self.score_position(pos_tags, idx, cand)
            if score > best_score:
                best_score = score
                best_pos = cand
        return best_pos, best_score

    def compare_alternatives(
        self,
        pos_tags: list[str],
        idx: int,
        alternatives: list[tuple[str, str]],
    ) -> list[tuple[str, str, float]]:
        """Compare plusieurs alternatives (forme, pos) a une position.

        Utile pour la desambiguation d'homophones : chaque alternative
        a un mot et un POS different.

        Args:
            pos_tags: sequence POS courante
            idx: position a evaluer
            alternatives: liste de (forme, pos) a comparer

        Returns:
            liste de (forme, pos, score) triee par score decroissant
        """
        scored = []
        for forme, pos in alternatives:
            score = self.score_position(pos_tags, idx, pos)
            scored.append((forme, pos, score))
        scored.sort(key=lambda x: -x[2])
        return scored

    # ------------------------------------------------------------------
    # POS+Morpho scoring
    # ------------------------------------------------------------------

    @staticmethod
    def make_pm_tag(pos: str, number: str = "_", gender: str = "_", person: str = "_") -> str:
        """Construit un tag POS+Morpho (ex: "NOM|Plur|Masc|_")."""
        return f"{pos}|{number}|{gender}|{person}"

    def score_pm_position(
        self,
        pm_tags: list[str],
        idx: int,
        candidate_pm: str,
    ) -> float:
        """Score local POS+Morpho a une position."""
        if idx < 0 or idx >= len(pm_tags):
            return _DEFAULT_LOGP

        padded = [self.BOS, self.BOS] + list(pm_tags) + [self.EOS]
        real_idx = idx + 2

        old = padded[real_idx]
        padded[real_idx] = candidate_pm

        score = 0.0
        for i in range(max(2, real_idx), min(len(padded), real_idx + 3)):
            score += self.logp_pm_trigram(
                padded[i - 2], padded[i - 1], padded[i],
            )

        padded[real_idx] = old
        return score

    def close(self):
        self._conn.close()
