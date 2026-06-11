"""Passe 1 — Correction orthographique pure (OOV -> lexique).

Corrige les mots hors-lexique sans analyse grammaticale.
Chaque mot est traite independamment : pas de contexte inter-mots.

Algorithme :
  1. lexique.existe(mot) -> dans_lexique
  2. Si dans lexique : rien (sauf variantes accent via meilleure_variante_accent)
  3. Si OOV :
     a. Generer candidats : accents, edit d=1, AZERTY, G2P phonetique, edit d=2
     b. Classer par frequence
     c. Appliquer le meilleur si freq > seuil
"""

from __future__ import annotations

import logging

from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS
from lectura_correcteur._types import MotV2

logger = logging.getLogger(__name__)

# Seuil de frequence en dessous duquel un mot in-lexique est soumis
# aux corrections d'accent (mots rares potentiellement mal accentues).
# Desactive (0.0) : les mots in-lexique ne sont jamais corriges en accent.
# Le risque de FP (guidé→guide, pratiqué→pratique) est trop eleve.
_SEUIL_FREQ_ACCENT = 0.0

# Mots courts tres courants a ne jamais corriger (pronoms, negation, etc.)
# Meme si leur freq lexique est basse, ils sont quasi-toujours corrects.
_MOTS_PROTEGES = frozenset(_FUNCTION_WORD_POS.keys()) | frozenset({
    "ne", "y", "en", "que", "qui", "dont", "ou", "et", "si", "ni",
    "me", "te", "se", "le", "la", "de",
    # Radicaux elides (quelqu', lorsqu', etc.) quand apostrophe separee par espace
    "quelqu", "lorsqu", "puisqu", "jusqu", "quelque",
})

# Mots courants a ne pas corriger via edit d=1 (homographes proches d'autres mots)
# Ex: "metre" ne doit pas devenir "mettre", "cable" ne doit pas devenir "table"
_OOV_SKIP_EDIT_D1 = frozenset({
    "metre", "cable", "guide",
})


def passe1_orthographe(
    mots: list[MotV2],
    lexique,
    g2p=None,
) -> None:
    """Passe 1 : correction orthographique pure (in-place sur mots).

    Args:
        mots: liste de MotV2 a corriger
        lexique: objet avec existe(), info(), frequence(), homophones(), phone_de()
        g2p: objet optionnel avec prononcer() pour les suggestions phonetiques
    """
    for mot in mots:
        low = mot.forme.lower()

        # 1. Verifier la presence dans le lexique
        mot.dans_lexique = lexique.existe(low)

        # Mots proteges : ne jamais corriger
        if low in _MOTS_PROTEGES:
            mot.dans_lexique = True
            continue

        # Tokens elides (j', s', l', d', etc.) : ne pas corriger
        if mot.original.endswith(("'", "\u2019")):
            mot.dans_lexique = True
            continue

        # Mots capitalises (noms propres, sigles) : ne pas corriger
        if mot.original[0].isupper():
            mot.dans_lexique = True
            continue

        # Mots tres courts (<= 3 chars) : ne pas corriger en OOV
        # (risque trop eleve de faux positif ; homophones traites en passe 2)
        if len(low) <= 3:
            continue

        if mot.dans_lexique:
            freq = lexique.frequence(low) if hasattr(lexique, "frequence") else 100.0
            # Mots rares : tenter une correction d'accent seulement
            if freq < _SEUIL_FREQ_ACCENT:
                _corriger_accent(mot, lexique, freq)
            continue

        # 2. OOV : generer des candidats et appliquer le meilleur
        _corriger_oov(mot, lexique, g2p)


def _corriger_accent(mot: MotV2, lexique, freq_actuelle: float) -> None:
    """Tente une correction d'accent pour un mot in-lexique basse-frequence."""
    from lectura_correcteur.orthographe._suggestions import (
        _meilleure_variante_accent,
    )

    meilleure = _meilleure_variante_accent(mot.forme.lower(), lexique, freq_actuelle)
    if meilleure is not None:
        mot.forme = meilleure
        mot.dans_lexique = True
        mot.source_ortho = "accent"
        mot.corrections.append((1, "ortho.accent", f"{mot.original} -> {meilleure}"))


def _corriger_oov(mot: MotV2, lexique, g2p) -> None:
    """Corrige un mot OOV via variantes d'accent + edit-distance conservatif.

    Strategie conservative : les corrections accent-only sont toujours
    tentees (fiables). Les corrections edit d=1/d=2 ne sont faites que
    si le candidat est un doublement/dedoublement de consonne ou une
    variante tres proche (meme longueur +/- 1).
    """
    from lectura_correcteur.orthographe._suggestions import (
        _est_variante_accent,
        _variantes_accents,
        _edits_distance_1,
    )

    low = mot.forme.lower()
    best_forme: str | None = None
    best_freq: float = 0.0
    best_source: str = ""

    def _freq(c: str) -> float:
        return lexique.frequence(c) if hasattr(lexique, "frequence") else 0.0

    # a. Variantes d'accent (priorite la plus haute, toujours fiable)
    accents = _variantes_accents(low, lexique)
    for forme, freq in accents:
        if freq > best_freq:
            best_forme, best_freq, best_source = forme, freq, "accent"

    # b. Edit distance 1 — uniquement si pas de variante accent trouvee
    #    et mot suffisamment long (>= 5 chars) pour eviter les faux positifs
    #    sur mots courts (jiu→jeu, your→pour).
    if best_source != "accent" and low not in _OOV_SKIP_EDIT_D1 and len(low) >= 5:
        d1_candidats = _edits_distance_1(low)
        for c in d1_candidats:
            if not lexique.existe(c):
                continue
            freq = _freq(c)
            if freq <= best_freq:
                continue
            # Accepter : accent-only (le plus fiable)
            if _est_variante_accent(low, c):
                best_forme, best_freq, best_source = c, freq, "accent"
            elif abs(len(c) - len(low)) <= 1 and freq >= 50.0:
                # Correction edit d=1 : exiger freq tres elevee
                best_forme, best_freq, best_source = c, freq, "edit_d1"

    # c-e. Pas de AZERTY/phonetique/d2 en mode conservatif —
    #      trop de faux positifs sur les mots etrangers/techniques.

    # Appliquer la meilleure correction trouvee
    if best_forme is not None and best_freq >= 10.0:
        mot.forme = best_forme
        mot.dans_lexique = True
        mot.source_ortho = best_source
        mot.corrections.append(
            (1, f"ortho.{best_source}", f"{mot.original} -> {best_forme}")
        )
