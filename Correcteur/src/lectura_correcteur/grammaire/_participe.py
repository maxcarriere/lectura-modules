"""Regles de participe passe : auxiliaire + infinitif -> participe passe.

Inclut aussi l'accord du PP avec le sujet quand l'auxiliaire est etre.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    ALLER_FORMES,
    AUXILIAIRES,
    ETRE_FORMES,
    MODAUX_FORMES,
    generer_candidats_participe,
    generer_candidats_pp_accorde,
    trouver_sujet_genre_nombre,
)

# Terminaisons de PP (pour heuristique morpho sans POS)
_PP_SUFFIXES = ("é", "és", "ée", "ées", "i", "is", "ie", "ies",
                "u", "us", "ue", "ues", "it", "ite", "ites",
                "ert", "erte", "ertes", "erts", "oint", "ointe",
                "eint", "einte", "aint", "ainte", "ort", "orte")


def verifier_participes_passes(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Corrige les confusions infinitif/PP et present/PP.

    Patterns :
    - auxiliaire + infinitif -> participe passe ("j'ai manger" -> "j'ai mange")
    - auxiliaire + present 1er groupe -> PP ("a sonne" -> "a sonne")
    - modal + PP -> infinitif ("faut ecoute" -> "faut ecouter")
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

        # Chercher un auxiliaire ou modal avant le mot (en sautant ne/pas/y/en)
        aux_found = False
        modal_found = False
        aller_found = False
        for j in range(i - 1, max(-1, i - 4), -1):
            w = result[j].lower()
            if w in AUXILIAIRES:
                aux_found = True
                break
            if w in MODAUX_FORMES:
                modal_found = True
                break
            if w in ALLER_FORMES:
                aller_found = True
                break
            if w not in _TRANSPARENTS:
                break

        # --- Regle 1 : AUX + infinitif -> PP ---
        if aux_found and pos == "VER" and curr_low.endswith(("er", "ir", "re")):
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
            if result[i] != curr:
                continue

        # --- Regle 2 : AUX + present 1er groupe (-e) -> PP (-e) ---
        # "a sonne" -> "a sonne" (present → PP pour 1er groupe)
        if aux_found and (pos == "VER" or curr_low.endswith(_PP_SUFFIXES)):
            if (
                curr_low.endswith("e")
                and not curr_low.endswith(("ee", "er", "re", "le", "ne", "se", "te", "ée"))
                and len(curr_low) > 2
            ):
                candidate = curr_low[:-1] + "é"
                if lexique is None or lexique.existe(candidate):
                    ancien = result[i]
                    result[i] = transferer_casse(curr, candidate)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="Present apres auxiliaire -> participe passe",
                    ))
                    continue

        # --- Regle 3 : Modal + PP (-e/-i/-u) -> infinitif ---
        # "faut ecoute" -> "faut ecouter", "doit fini" -> "doit finir"
        if modal_found and (pos == "VER" or curr_low.endswith(_PP_SUFFIXES)):
            candidate = None
            if curr_low.endswith("é") and not curr_low.endswith("er"):
                candidate = curr_low[:-1] + "er"
            elif curr_low.endswith("i") and not curr_low.endswith("ir"):
                candidate = curr_low + "r"
            elif curr_low.endswith("u") and not curr_low.endswith(("re", "ur")):
                candidate = curr_low + "re"  # rendu -> rendre (not exact)
            if candidate and (lexique is None or lexique.existe(candidate)):
                ancien = result[i]
                result[i] = transferer_casse(curr, candidate)
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    explication="PP apres modal -> infinitif",
                ))
                continue

        # --- Regle 4 : Aller + PP (-e) -> infinitif ---
        # (extension de la regle existante dans _homophones.py, ici plus large)
        if aller_found and pos == "VER" and curr_low.endswith("é") and not curr_low.endswith("er"):
            candidate = curr_low[:-1] + "er"
            if lexique is None or lexique.existe(candidate):
                ancien = result[i]
                result[i] = transferer_casse(curr, candidate)
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    explication="PP -> infinitif apres aller",
                ))
                continue

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

        # Accepter VER ou mots ressemblant a un PP (heuristique morpho)
        _is_pp_like = pos == "VER" or curr_low.endswith(_PP_SUFFIXES)
        if not _is_pp_like:
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
