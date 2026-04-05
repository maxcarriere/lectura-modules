"""Fonctions de phonetique : rimes, contient_son, mots_par_syllabes."""

from __future__ import annotations

from typing import Any

from lectura_lexique._utils import _tokenize_ipa


def rimes(
    phone_index: dict[str, list[dict[str, Any]]],
    mot_phone: str,
    nb_phonemes: int = 2,
    limite: int = 50,
) -> list[dict[str, Any]]:
    """Mots partageant les N derniers phonemes.

    Args:
        phone_index: Index phone -> [entrees]
        mot_phone: Transcription IPA du mot de reference
        nb_phonemes: Nombre de phonemes de fin a matcher
        limite: Nombre max de resultats

    Returns:
        Liste d'entrees triees par frequence decroissante
    """
    if not mot_phone:
        return []

    segments = _tokenize_ipa(mot_phone)
    if len(segments) < nb_phonemes:
        return []

    suffixe = segments[-nb_phonemes:]
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for phone, entries in phone_index.items():
        phone_segments = _tokenize_ipa(phone)
        if len(phone_segments) < nb_phonemes:
            continue
        if phone_segments[-nb_phonemes:] != suffixe:
            continue
        # Exclure le phone identique au mot de reference
        for e in entries:
            ortho = str(e.get("ortho", "") or "").lower()
            if ortho and ortho not in seen:
                seen.add(ortho)
                results.append(e)

    # Trier par frequence decroissante
    results.sort(key=lambda x: -float(x.get("freq", 0) or 0))
    return results[:limite]


def contient_son(
    phone_index: dict[str, list[dict[str, Any]]],
    son: str,
    limite: int = 50,
) -> list[dict[str, Any]]:
    """Mots contenant une sequence phonetique.

    Args:
        phone_index: Index phone -> [entrees]
        son: Sequence IPA a rechercher
        limite: Nombre max de resultats

    Returns:
        Liste d'entrees triees par frequence decroissante
    """
    if not son:
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for phone, entries in phone_index.items():
        if son in phone:
            for e in entries:
                ortho = str(e.get("ortho", "") or "").lower()
                if ortho and ortho not in seen:
                    seen.add(ortho)
                    results.append(e)

    results.sort(key=lambda x: -float(x.get("freq", 0) or 0))
    return results[:limite]


def mots_par_syllabes(
    ortho_index: dict[str, list[dict[str, Any]]],
    n: int,
    cgram: str | None = None,
    limite: int = 50,
) -> list[dict[str, Any]]:
    """Mots avec exactement N syllabes.

    Utilise nb_syllabes ou syllabes.count('.')+1.

    Args:
        ortho_index: Index ortho -> [entrees]
        n: Nombre de syllabes souhaite
        cgram: Filtre POS optionnel
        limite: Nombre max de resultats

    Returns:
        Liste d'entrees triees par frequence decroissante
    """
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for _ortho, entries in ortho_index.items():
        for e in entries:
            # Filtre POS
            if cgram is not None:
                e_cgram = str(e.get("cgram", "") or "")
                if not e_cgram.startswith(cgram):
                    continue

            # Determiner le nombre de syllabes
            nb = _get_nb_syllabes(e)
            if nb != n:
                continue

            ortho = str(e.get("ortho", "") or "").lower()
            if ortho and ortho not in seen:
                seen.add(ortho)
                results.append(e)

    results.sort(key=lambda x: -float(x.get("freq", 0) or 0))
    return results[:limite]


def _get_nb_syllabes(entry: dict[str, Any]) -> int | None:
    """Extrait le nombre de syllabes d'une entree."""
    # Essayer nb_syllabes d'abord
    nb_raw = entry.get("nb_syllabes") or entry.get("nbsyll")
    if nb_raw is not None:
        try:
            return int(nb_raw)
        except (ValueError, TypeError):
            pass

    # Fallback : compter les points dans syllabes + 1
    syllabes = str(entry.get("syllabes", "") or entry.get("syll", "") or "")
    if syllabes:
        return syllabes.count(".") + 1

    return None
