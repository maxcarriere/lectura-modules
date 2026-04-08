"""Verificateur orthographique simple (lookup lexique).

Pas de G2P/P2G : verifie simplement si chaque mot existe dans le lexique.
Les mots inconnus sont signales avec type_correction=HORS_LEXIQUE.
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur._types import MotAnalyse, TypeCorrection
from lectura_correcteur._utils import PUNCT_RE
from lectura_correcteur.orthographe._suggestions import (
    _est_doublement_consonne,
    _est_variante_accent,
    _meilleure_variante_accent,
    suggerer,
)

# POS CRF -> cgram lexique compatibles (pour le re-ranking)
_POS_COMPAT: dict[str, set[str]] = {
    "VER": {"VER", "AUX"},
    "AUX": {"AUX", "VER"},
    "NOM": {"NOM"},
    "ADJ": {"ADJ"},
    "ADV": {"ADV"},
    "PRE": {"PRE"},
    "CON": {"CON"},
}


def _reclasser_par_pos(
    suggestions: list[str], pos_crf: str, lexique: Any,
    mot_original: str = "",
) -> list[str]:
    """Re-classe les suggestions en prioritisant celles compatibles avec le POS CRF.

    Les variantes accent-only gardent leur ordre par frequence (ne sont pas
    re-classees par POS), car elles sont tres fiables et le CRF se trompe
    souvent sur les mots OOV.
    """
    if not pos_crf:
        return suggestions
    cgrams_ok = _POS_COMPAT.get(pos_crf)
    if not cgrams_ok:
        return suggestions

    low = mot_original.lower()
    # Accent-only variants: preserve frequency order (safe, POS-independent)
    accents: list[str] = []
    non_accents: list[str] = []
    for s in suggestions:
        if low and _est_variante_accent(low, s.lower()):
            accents.append(s)
        else:
            non_accents.append(s)

    # POS re-ranking only on non-accent candidates
    compatibles: list[tuple[str, float]] = []
    autres: list[tuple[str, float]] = []
    for s in non_accents:
        freq = lexique.frequence(s) if hasattr(lexique, "frequence") else 0.0
        infos = lexique.info(s) if hasattr(lexique, "info") else []
        cand_cgrams = {e.get("cgram", "") for e in infos}
        if cand_cgrams & cgrams_ok:
            compatibles.append((s, freq))
        else:
            autres.append((s, freq))

    compatibles.sort(key=lambda x: -x[1])
    autres.sort(key=lambda x: -x[1])
    return accents + [s for s, _ in compatibles] + [s for s, _ in autres]


class VerificateurOrthographe:
    """Verification orthographique mot par mot via le lexique."""

    def __init__(
        self, lexique: Any, *, max_suggestions: int = 5, distance: int = 2,
        g2p: Any = None, scoring_actif: bool = False,
        symspell: Any = None,
    ) -> None:
        self._lexique = lexique
        self._max_suggestions = max_suggestions
        self._distance = distance
        self._g2p = g2p
        self._scoring_actif = scoring_actif
        self._symspell = symspell

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

            # Tokens avec trait d'union dont les parties sont connues
            # (ex: "vas-tu", "a-t-il") : considerer comme dans le lexique
            if (
                not dans_lexique
                and "-" in mot
                and not mot.startswith("-")
                and not mot.endswith("-")
            ):
                parts = mot.split("-")
                meaningful = [p for p in parts if p.lower() not in ("", "t")]
                if meaningful and all(
                    self._lexique.existe(p) for p in meaningful
                ):
                    dans_lexique = True

            type_corr = TypeCorrection.AUCUNE
            suggestions_list: list[str] = []
            corrige = mot

            # Accent disambiguation: mot in-lexique but rare,
            # and an accent variant is much more frequent
            if dans_lexique and not PUNCT_RE.match(mot):
                freq_actuelle = (
                    self._lexique.frequence(mot)
                    if hasattr(self._lexique, "frequence") else 999.0
                )
                accent_alt = _meilleure_variante_accent(
                    mot, self._lexique, freq_actuelle,
                )
                if accent_alt:
                    corrige = accent_alt
                    type_corr = TypeCorrection.HORS_LEXIQUE

            if not dans_lexique and not PUNCT_RE.match(mot):
                type_corr = TypeCorrection.HORS_LEXIQUE
                suggestions_list = suggerer(
                    mot, self._lexique,
                    max_n=self._max_suggestions,
                    distance=self._distance,
                    g2p=self._g2p,
                    symspell=self._symspell,
                )
                # Re-rank by POS CRF compatibility
                if suggestions_list and pos:
                    suggestions_list = _reclasser_par_pos(
                        suggestions_list, pos, self._lexique, mot,
                    )
                # Auto-correction: apply top suggestion when confident
                if suggestions_list:
                    top = suggestions_list[0]
                    low = mot.lower()
                    top_low = top.lower()
                    # Accent-only variants: always auto-correct (very safe)
                    if _est_variante_accent(low, top_low):
                        corrige = top
                    elif _est_doublement_consonne(low, top_low):
                        # Doublement/dedoublement de consonne: lower threshold
                        top_freq = (
                            self._lexique.frequence(top)
                            if hasattr(self._lexique, "frequence") else 0.0
                        )
                        if top_freq >= 1.0:
                            corrige = top
                    else:
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
