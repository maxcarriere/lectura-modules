"""Regles de participe passe : auxiliaire + infinitif -> participe passe.

Inclut aussi l'accord du PP avec le sujet quand l'auxiliaire est etre.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    AUXILIAIRES,
    ETRE_FORMES,
    generer_candidats_participe,
    generer_candidats_pp_accorde,
    trouver_sujet_genre_nombre,
)


def verifier_participes_passes(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Corrige les infinitifs utilises a la place d'un participe passe.

    Pattern : auxiliaire (avoir/etre) + infinitif -> participe passe.
    Ex: "j'ai manger" -> "j'ai mange"
    """
    if not mots:
        return mots, []

    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    _TRANSPARENTS = frozenset({
        "ne", "n'", "pas", "plus", "jamais", "rien",
        "point", "y", "en",
    })

    for i in range(1, n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]
        curr_low = curr.lower()

        if pos != "VER":
            continue

        # Chercher un auxiliaire avant le verbe (en sautant ne/pas/y/en)
        aux_found = False
        for j in range(i - 1, max(-1, i - 4), -1):
            w = result[j].lower()
            if w in AUXILIAIRES:
                aux_found = True
                break
            if w not in _TRANSPARENTS:
                break
        if not aux_found:
            continue

        # Verifier que le mot ressemble a un infinitif (-er, -ir, -re)
        if not curr_low.endswith(("er", "ir", "re")):
            continue

        # Generer des candidats participe passe
        candidats = generer_candidats_participe(curr)
        for candidate in candidats:
            if lexique is None or lexique.existe(candidate):
                ancien = result[i]
                result[i] = transferer_casse(curr, candidate)
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    explication="Infinitif apres auxiliaire -> participe passe",
                ))
                break

    return result, corrections


def verifier_pp_accord_etre(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Accorde le participe passe avec le sujet quand l'auxiliaire est etre.

    Pattern : sujet + etre_conjugue + PP (VER qui n'est pas un infinitif)
    Ex: "elle est allé" -> "elle est allée"
    """
    if not mots:
        return mots, []

    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    _TRANSPARENTS = frozenset({
        "ne", "n'", "pas", "plus", "jamais", "rien",
        "point", "y", "en",
    })

    for i in range(1, n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]
        curr_low = curr.lower()

        if pos != "VER":
            continue

        # Exclure les infinitifs (-er, -ir, -re) : ce n'est pas un PP
        if curr_low.endswith(("er", "ir", "re")):
            continue

        # Chercher une forme d'etre avant le mot (en sautant ne/pas/y/en)
        etre_found = False
        etre_idx = -1
        for j in range(i - 1, max(-1, i - 4), -1):
            w = result[j].lower()
            if w in ETRE_FORMES:
                etre_found = True
                etre_idx = j
                break
            if w not in _TRANSPARENTS:
                break
        if not etre_found:
            continue

        # Trouver le genre/nombre du sujet
        gn = trouver_sujet_genre_nombre(
            result, pos_tags, morpho, etre_idx, lexique,
        )
        if gn is None:
            continue
        genre, nombre = gn

        # Generer la forme accordee
        candidats = generer_candidats_pp_accorde(curr_low, genre, nombre)
        for candidate in candidats:
            if lexique is None or lexique.existe(candidate):
                if candidate != curr_low:
                    ancien = result[i]
                    result[i] = transferer_casse(curr, candidate)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication=f"Accord PP avec sujet ({genre} {nombre}) apres etre",
                    ))
                break

    return result, corrections
