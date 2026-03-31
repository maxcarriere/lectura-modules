"""Verificateur orthographique simple (lookup lexique).

Pas de G2P/P2G : verifie simplement si chaque mot existe dans le lexique.
Les mots inconnus sont signales avec type_correction=HORS_LEXIQUE.
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur._types import MotAnalyse, TypeCorrection
from lectura_correcteur._utils import PUNCT_RE


class VerificateurOrthographe:
    """Verification orthographique mot par mot via le lexique."""

    def __init__(self, lexique: Any) -> None:
        self._lexique = lexique

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

            dans_lexique = self._lexique.existe(mot)
            type_corr = TypeCorrection.AUCUNE
            if not dans_lexique and not PUNCT_RE.match(mot):
                type_corr = TypeCorrection.HORS_LEXIQUE

            results.append(MotAnalyse(
                original=mot,
                corrige=mot,
                pos=pos,
                morpho=morpho_dict,
                dans_lexique=dans_lexique,
                type_correction=type_corr,
            ))

        return results
