"""Tagger fallback par lookup lexique — zero modele, zero dependance.

Pour chaque mot, interroge lexique.info() et retourne le POS/morpho
de l'entree la plus frequente. Pas de desambiguation contextuelle :
"mange" sera toujours tague selon son entree la plus frequente.

Utilise comme fallback quand aucun modele POS/Morpho n'est injecte.
"""

from __future__ import annotations

import re
from typing import Any

from lectura_correcteur._utils import normaliser_morpho

# Regex tokenisation française (elisions + mots + ponctuation)
_TOKEN_RE = re.compile(
    r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu|[Jj]usqu|[Qq]uelqu)"
    r"['\u2019]"
    r"|[\w]+(?:-[\w]+)*"
    r"|[^\s\w]+",
)


# Mots-outils dont le POS lexique est domine par une entree NOM/autre
# erronee (freq_composite aggrege les frequences de toutes les positions).
# On force le POS correct car ces mots sont quasi-toujours des mots
# grammaticaux, jamais des noms communs en contexte reel.
_FUNCTION_WORD_POS: dict[str, str] = {
    # Pronoms personnels (NOM domine par freq dans v4)
    "elle": "PRO:per",
    "elles": "PRO:per",
    "on": "PRO:per",
    # Articles / determinants (NOM domine par freq dans v4)
    "un": "ART:ind",
    "une": "ART:ind",
    "des": "ART",
    "du": "ART",
    "la": "ART",
    # Possessifs (NOM domine par freq dans v4)
    "mon": "ADJ:pos",
    "ton": "ADJ:pos",
    "sa": "ADJ:pos",
    "ma": "ADJ:pos",
    "ta": "ADJ:pos",
    # Prepositions / contractions
    "au": "PRE",
    "aux": "ART:def",
    # Verbe etre (ADJ "est"=east domine par freq egale dans v4)
    "est": "AUX",
}

# POS grammaticaux a preferer pour les mots courts (<=3 chars)
# quand le lexique retourne NOM mais qu'une entree grammaticale existe.
_GRAMMATICAL_POS_PREFIXES = frozenset({
    "ART", "PRE", "CON", "PRO", "DET", "ADV", "ADJ:pos", "ADJ:dem",
    "AUX",
})


class LexiqueTagger:
    """Tagger POS/Morpho par simple lookup lexique (plus frequente entree).

    Satisfait TaggerProtocol.
    """

    def __init__(self, lexique: Any) -> None:
        self._lexique = lexique

    def tokenize(self, text: str) -> list[tuple[str, bool]]:
        """Tokenise via regex."""
        result: list[tuple[str, bool]] = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group()
            is_word = tok[0].isalpha() or tok[0] == "_"
            result.append((tok, is_word))
        return result

    def _analyse_mot(self, word: str) -> tuple[dict, list[tuple[str, float]]]:
        """Analyse un mot : retourne (morpho_dict, pos_scores).

        pos_scores est une liste de (POS, probabilite) triee par prob
        decroissante, normalisee a somme 1.
        """
        d: dict[str, str] = {}
        low = word.lower()
        infos = self._lexique.info(low) if hasattr(self._lexique, "info") else []

        # Calculer pos_scores depuis toutes les entrees du lexique
        pos_freq: dict[str, float] = {}
        for entry in infos:
            cgram = entry.get("cgram")
            if cgram:
                freq = float(entry.get("freq") or 0)
                pos_freq[cgram] = pos_freq.get(cgram, 0) + max(freq, 0.01)

        # Override pour mots-outils mal classes par frequence
        override = _FUNCTION_WORD_POS.get(low)
        if override:
            d["pos"] = override
            # Boost l'override dans pos_scores
            if override not in pos_freq:
                pos_freq[override] = 100.0
            else:
                # Donner un poids dominant a l'override
                max_f = max(pos_freq.values()) if pos_freq else 1.0
                pos_freq[override] = max(pos_freq[override], max_f * 5)
            # Charger morpho depuis l'entree du bon POS
            matched = [e for e in infos if e.get("cgram") == override]
            if matched:
                best = max(matched, key=lambda e: float(e.get("freq") or 0))
                for feat in ("genre", "nombre", "temps", "mode", "personne"):
                    val = best.get(feat)
                    if val is not None:
                        d[feat] = normaliser_morpho(val)
        elif infos:
            best = max(infos, key=lambda e: float(e.get("freq") or 0))
            if (
                len(low) <= 3
                and best.get("cgram") == "NOM"
            ):
                gram_entries = [
                    e for e in infos
                    if any(
                        (e.get("cgram") or "").startswith(pfx)
                        for pfx in _GRAMMATICAL_POS_PREFIXES
                    )
                ]
                if gram_entries:
                    best = max(
                        gram_entries,
                        key=lambda e: float(e.get("freq") or 0),
                    )
            if best.get("cgram"):
                d["pos"] = best["cgram"]
            for feat in ("genre", "nombre", "temps", "mode", "personne"):
                val = best.get(feat)
                if val is not None:
                    d[feat] = normaliser_morpho(val)

        # Normaliser pos_scores
        pos_scores: list[tuple[str, float]] = []
        total = sum(pos_freq.values())
        if total > 0:
            pos_scores = sorted(
                [(pos, freq / total) for pos, freq in pos_freq.items()],
                key=lambda x: -x[1],
            )

        return d, pos_scores

    def tag_words(self, words: list[str]) -> list[dict]:
        """Tague chaque mot par lookup lexique (entree la plus frequente)."""
        results: list[dict] = []
        for word in words:
            d, _ = self._analyse_mot(word)
            results.append(d)
        return results

    def tag_words_rich(self, words: list[str]) -> list[dict]:
        """Tague avec scores POS pour Viterbi disambiguation.

        Retourne list[dict] avec en plus :
        - pos_scores: list[tuple[str, float]] (POS avec probabilites)
        - confiance_pos: float (confiance du top-1)
        """
        results: list[dict] = []
        for word in words:
            d, pos_scores = self._analyse_mot(word)
            d["pos_scores"] = pos_scores
            d["confiance_pos"] = pos_scores[0][1] if pos_scores else 1.0
            results.append(d)
        return results
