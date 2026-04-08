"""Verificateur orthographique simple (lookup lexique).

Pas de G2P/P2G : verifie simplement si chaque mot existe dans le lexique.
Les mots inconnus sont signales avec type_correction=HORS_LEXIQUE.
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur._types import MotAnalyse, TypeCorrection
from lectura_correcteur._utils import PUNCT_RE
from lectura_correcteur.orthographe._suggestions import suggerer


class VerificateurOrthographe:
    """Verification orthographique mot par mot via le lexique."""

    def __init__(
        self, lexique: Any, *, max_suggestions: int = 5, distance: int = 2,
        g2p: Any = None, scoring_actif: bool = False,
    ) -> None:
        self._lexique = lexique
        self._max_suggestions = max_suggestions
        self._distance = distance
        self._g2p = g2p
        self._scoring_actif = scoring_actif

    def verifier_phrase(
        self,
        mots: list[str],
        analyses_morpho: list[dict] | None = None,
    ) -> list[MotAnalyse]:
        """Verifie chaque mot de la phrase dans le lexique.

        Args:
            mots: Tokens mots (sans ponctuation).
            analyses_morpho: Resultats du MorphoTagger (optionnel).

        Returns:
            Liste de MotAnalyse avec dans_lexique et type_correction.
        """
        morpho_list = analyses_morpho or [{}] * len(mots)
        results: list[MotAnalyse] = []

        for i, mot in enumerate(mots):
            morpho_info = morpho_list[i] if i < len(morpho_list) else {}
            pos = morpho_info.get("pos", "")
            morpho_dict = {}
            for key in ("genre", "nombre", "temps", "mode", "personne"):
                val = morpho_info.get(key)
                if val is not None:
                    morpho_dict[key] = val

            # Tokens d'elision (j', l', n', etc.) : ne pas tenter de corriger
            if mot.endswith(("'", "\u2019")):
                results.append(MotAnalyse(
                    original=mot,
                    corrige=mot,
                    pos=pos,
                    morpho=morpho_dict,
                    dans_lexique=True,
                ))
                continue

            dans_lexique = self._lexique.existe(mot)
            type_corr = TypeCorrection.AUCUNE
            suggestions_list: list[str] = []
            corrige = mot
            if not dans_lexique and not PUNCT_RE.match(mot):
                type_corr = TypeCorrection.HORS_LEXIQUE
                suggestions_list = suggerer(
                    mot, self._lexique,
                    max_n=self._max_suggestions,
                    distance=self._distance,
                    g2p=self._g2p,
                )
                # Auto-correction: apply top suggestion when confident
                if suggestions_list:
                    top = suggestions_list[0]
                    top_freq = (
                        self._lexique.frequence(top)
                        if hasattr(self._lexique, "frequence") else 0.0
                    )
                    if top_freq >= 5.0:
                        corrige = top

            results.append(MotAnalyse(
                original=mot,
                corrige=corrige,
                pos=pos,
                morpho=morpho_dict,
                dans_lexique=dans_lexique,
                type_correction=type_corr,
                suggestions=suggestions_list,
            ))

        return results
