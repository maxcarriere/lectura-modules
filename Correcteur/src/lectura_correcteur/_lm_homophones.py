"""LM trigramme specialise pour la desambiguisation des homophones.

Charge homophones_trigrams.db (29 MB) et expose une interface simple
pour scorer un mot dans son contexte gauche/droit.

Schema DB:
  - trigrams(w_prev, w_target, w_next, count) — top-500 per target
  - bigrams_right(w_target, w_next, total) — sommes pre-calculees
  - bigrams_left(w_prev, w_target, total) — sommes pre-calculees
  - homophone_groups(phone, ortho, freq_lexique)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class LMHomophones:
    """LM trigramme specialise homophones (338 groupes, 740 formes)."""

    def __init__(self, db_path: str | Path, lexique=None):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._cur = self._conn.cursor()

        # Detecter si le schema inclut les tables bigrams pre-calculees
        self._cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bigrams_right'"
        )
        self._has_bigram_tables = self._cur.fetchone() is not None

        # Charger les groupes homophones
        self._ortho_to_phone: dict[str, str] = {}
        self._phone_to_orthos: dict[str, list[tuple[str, float]]] = {}

        self._cur.execute(
            "SELECT phone, ortho, freq_lexique FROM homophone_groups"
        )
        for phone, ortho, freq in self._cur.fetchall():
            self._ortho_to_phone[ortho.lower()] = phone
            if phone not in self._phone_to_orthos:
                self._phone_to_orthos[phone] = []
            self._phone_to_orthos[phone].append((ortho, freq))

        # Construire les paires de meme lemme pour filtrer les variantes
        # flexionnelles (ex: parti/partie/partis partagent le lemme "parti")
        self._same_lemma_pairs: set[tuple[str, str]] = set()
        if lexique is not None and hasattr(lexique, "lemme_de"):
            self._build_same_lemma_pairs(lexique)

    def _build_same_lemma_pairs(self, lexique) -> None:
        """Construit l'ensemble des paires (a, b) partageant un lemme commun.

        Verifie TOUS les lemmes de chaque forme (pas seulement le plus
        frequent), pour couvrir les cas ou deux formes partagent un lemme
        verbal mais ont des lemmes nominaux differents
        (ex: partis/partie partagent le lemme verbal "partir").
        """
        if not hasattr(lexique, "info"):
            return
        for phone, orthos in self._phone_to_orthos.items():
            if len(orthos) < 2:
                continue
            # Collecter TOUS les lemmes pour chaque forme du groupe
            lemma_map: dict[str, list[str]] = {}  # lemme -> [formes]
            for ortho, _freq in orthos:
                low = ortho.lower()
                infos = lexique.info(low)
                if not infos:
                    continue
                seen_lemmas: set[str] = set()
                for entry in infos:
                    lemme = entry.get("lemme", "")
                    if lemme and lemme.lower() not in seen_lemmas:
                        seen_lemmas.add(lemme.lower())
                        lemma_map.setdefault(lemme.lower(), []).append(low)
            # Pour chaque lemme avec 2+ formes distinctes, exclure mutuellement
            for _lemme, formes in lemma_map.items():
                unique_formes = list(dict.fromkeys(formes))  # deduplicate
                if len(unique_formes) < 2:
                    continue
                for fi in range(len(unique_formes)):
                    for fj in range(fi + 1, len(unique_formes)):
                        self._same_lemma_pairs.add(
                            (unique_formes[fi], unique_formes[fj])
                        )
                        self._same_lemma_pairs.add(
                            (unique_formes[fj], unique_formes[fi])
                        )

    @property
    def n_groups(self) -> int:
        return len(self._phone_to_orthos)

    def est_homophone(self, mot: str) -> bool:
        """True si le mot appartient a un groupe homophone connu."""
        return mot.lower() in self._ortho_to_phone

    def candidats(self, mot: str) -> list[tuple[str, float]] | None:
        """Retourne les variantes homophones (ortho, freq) ou None."""
        phone = self._ortho_to_phone.get(mot.lower())
        if phone is None:
            return None
        return self._phone_to_orthos.get(phone)

    def scorer(self, mot: str, ctx_gauche: str | None,
               ctx_droite: str | None) -> int:
        """Score un mot dans son contexte via trigrammes + bigrammes.

        Args:
            mot: le mot a scorer (orthographe)
            ctx_gauche: mot precedent (ou None)
            ctx_droite: mot suivant (ou None)

        Returns:
            score (0 si aucun n-gram trouve)
        """
        w = mot.lower()
        prev_w = ctx_gauche.lower() if ctx_gauche else None
        next_w = ctx_droite.lower() if ctx_droite else None
        score = 0

        # Trigram exact
        if prev_w and next_w:
            self._cur.execute(
                "SELECT count FROM trigrams "
                "WHERE w_prev=? AND w_target=? AND w_next=?",
                (prev_w, w, next_w),
            )
            row = self._cur.fetchone()
            if row:
                score += row[0] * 3

        # Bigrammes (tables pre-calculees si disponibles, sinon SUM sur trigrams)
        if self._has_bigram_tables:
            if prev_w:
                self._cur.execute(
                    "SELECT total FROM bigrams_left "
                    "WHERE w_prev=? AND w_target=?",
                    (prev_w, w),
                )
                row = self._cur.fetchone()
                if row:
                    score += row[0]

            if next_w:
                self._cur.execute(
                    "SELECT total FROM bigrams_right "
                    "WHERE w_target=? AND w_next=?",
                    (w, next_w),
                )
                row = self._cur.fetchone()
                if row:
                    score += row[0]
        else:
            # Fallback: somme sur trigrams (ancien schema)
            if prev_w:
                self._cur.execute(
                    "SELECT SUM(count) FROM trigrams "
                    "WHERE w_prev=? AND w_target=?",
                    (prev_w, w),
                )
                row = self._cur.fetchone()
                if row and row[0]:
                    score += row[0]

            if next_w:
                self._cur.execute(
                    "SELECT SUM(count) FROM trigrams "
                    "WHERE w_target=? AND w_next=?",
                    (w, next_w),
                )
                row = self._cur.fetchone()
                if row and row[0]:
                    score += row[0]

        return score

    def meilleur_homophone(
        self, mot: str, ctx_gauche: str | None, ctx_droite: str | None,
    ) -> tuple[str, str]:
        """Choisit le meilleur homophone selon le contexte.

        Returns:
            (best_ortho, source) — source est 'LM', 'FREQ' ou 'PASS'
        """
        mot_low = mot.lower()
        phone = self._ortho_to_phone.get(mot_low)
        if not phone:
            return mot, "PASS"

        candidates = self._phone_to_orthos.get(phone, [])
        if len(candidates) < 2:
            return mot, "PASS"

        scores = []
        for ortho, freq_lex in candidates:
            # Exclure les candidats partageant le meme lemme (variantes
            # flexionnelles, ex: parti/partie/partis)
            if (mot_low, ortho.lower()) in self._same_lemma_pairs:
                continue
            score = self.scorer(ortho, ctx_gauche, ctx_droite)
            scores.append((ortho, score, freq_lex))

        if not scores:
            return mot, "PASS"

        scores.sort(key=lambda x: (-x[1], -x[2]))
        best_ortho, best_score, _ = scores[0]

        if best_score > 0:
            return best_ortho, "LM"
        return best_ortho, "FREQ"

    def close(self):
        self._conn.close()
