"""Passe de correction orthographique via roundtrip P2G.

Exploite p2g_result['ortho'] pour corriger les mots dont la forme P2G
differe de l'original, avec des gardes conservatrices pour limiter les
faux positifs.
"""

from __future__ import annotations

import logging

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import PUNCT_RE, est_changement_genre, transferer_casse
from lectura_correcteur.orthographe._suggestions import (
    _edit_distance_rapide,
    _est_variante_accent,
)

logger = logging.getLogger(__name__)


def corriger_p2g_ortho(
    mots: list[str],
    p2g_ortho: list[str],
    p2g_confiance: list[float],
    p2g_alternatives: list,
    pos_tags: list[str],
    lexique,
    originaux: list[str],
    *,
    seuil_accent: float = 0.70,
    seuil_ortho: float = 0.85,
) -> tuple[list[str], list[Correction]]:
    """Corrige les mots via la sortie ortho du P2G.

    Args:
        mots: Formes courantes (lowercased, post-orthographe).
        p2g_ortho: Formes predites par le P2G (roundtrip phonetique).
        p2g_confiance: Confiance P2G par token.
        p2g_alternatives: Alternatives P2G par token (non utilise pour l'instant).
        pos_tags: POS tags (P2G ou lexique).
        lexique: Lexique (existe, info, frequence).
        originaux: Formes originales (pour transfert de casse).
        seuil_accent: Seuil de confiance pour variantes accent seul.
        seuil_ortho: Seuil de confiance pour corrections ortho generales.

    Returns:
        (mots_corriges, corrections)
    """
    n = len(mots)
    result = list(mots)
    corrections: list[Correction] = []

    for i in range(n):
        # Verifier que les donnees P2G existent pour cet index
        if i >= len(p2g_ortho) or i >= len(p2g_confiance):
            continue

        forme = mots[i]
        p2g = p2g_ortho[i]
        confiance = p2g_confiance[i]

        # --- GARDE 1 : Exclusions structurelles ---
        # Ponctuation
        if PUNCT_RE.match(forme):
            continue
        # Mots de 1-2 caracteres (trop de FP : s->c, a->à)
        if len(forme) <= 2:
            continue
        # Tokens numeriques
        if forme.isdigit():
            continue

        # --- GARDE 2 : Identite ---
        if p2g.lower() == forme.lower():
            continue

        # --- GARDE 3 : Seuil de confiance ---
        est_accent = _est_variante_accent(forme, p2g)
        seuil = seuil_accent if est_accent else seuil_ortho
        if confiance < seuil:
            continue

        # --- GARDE 4 : Validation lexique ---
        if not lexique.existe(p2g):
            continue

        # --- GARDE 5 : Plausibilite phonetique (distance d'edition <= 3) ---
        dist = _edit_distance_rapide(forme.lower(), p2g.lower())
        if dist > 3:
            continue

        # --- GARDE 6 : Pas de changement genre/nombre seul ---
        if est_changement_genre(forme.lower(), p2g.lower()):
            continue
        # Changement de nombre seul (s/x final)
        if _est_changement_nombre_seul(forme.lower(), p2g.lower()):
            continue

        # --- GARDE 7 : Original deja dans le lexique ---
        # Si le mot original est valide ET que ce n'est pas une variante
        # accent, ne pas corriger (le P2G ne doit pas remplacer un mot
        # valide par un autre mot valide).
        if lexique.existe(forme) and not est_accent:
            continue

        # --- Appliquer la correction ---
        # Determiner l'original pour le transfert de casse
        orig_casse = originaux[i] if i < len(originaux) else forme
        forme_corrigee = transferer_casse(orig_casse, p2g)
        result[i] = forme_corrigee

        corrections.append(Correction(
            index=i,
            original=orig_casse,
            corrige=forme_corrigee,
            type_correction=TypeCorrection.HORS_LEXIQUE,
            regle="p2g_ortho",
            explication=(
                f"P2G ortho: '{orig_casse}' -> '{forme_corrigee}' "
                f"(confiance={confiance:.2f})"
            ),
        ))

    return result, corrections


def _est_changement_nombre_seul(forme: str, p2g: str) -> bool:
    """True si la seule difference est un s ou x final (singulier/pluriel)."""
    # forme + s = p2g (chat -> chats)
    if p2g == forme + "s" or p2g == forme + "x":
        return True
    # forme = p2g + s (chats -> chat)
    if forme == p2g + "s" or forme == p2g + "x":
        return True
    # -aux <-> -al (animaux <-> animal)
    if forme.endswith("aux") and p2g == forme[:-3] + "al":
        return True
    if p2g.endswith("aux") and forme == p2g[:-3] + "al":
        return True
    return False
