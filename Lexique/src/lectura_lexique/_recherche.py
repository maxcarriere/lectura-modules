"""Fonctions de recherche : rechercher, filtrer, anagrammes."""

from __future__ import annotations

import re
from typing import Any, Iterable


def rechercher(
    index: dict[str, list[dict[str, Any]]],
    pattern: str,
    champ: str = "ortho",
    limite: int = 50,
) -> list[dict[str, Any]]:
    """Regex sur ortho ou phone.

    Args:
        index: Index (ortho ou phone) -> [entrees]
        pattern: Expression reguliere a chercher
        champ: Champ a tester ("ortho" ou "phone")
        limite: Nombre max de resultats

    Returns:
        Liste d'entrees correspondantes, triees par frequence
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for _key, entries in index.items():
        for e in entries:
            valeur = str(e.get(champ, "") or "")
            if valeur and regex.search(valeur):
                ortho = str(e.get("ortho", "") or "").lower()
                if ortho not in seen:
                    seen.add(ortho)
                    results.append(e)

    results.sort(key=lambda x: -float(x.get("freq", 0) or 0))
    return results[:limite]


def filtrer(
    entries_iter: Iterable[dict[str, Any]],
    *,
    cgram: str | None = None,
    genre: str | None = None,
    nombre: str | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    nb_syllabes: int | None = None,
    age_min: float | None = None,
    age_max: float | None = None,
    illustrable_min: float | None = None,
    categorie: str | None = None,
    has_age: bool | None = None,
    has_phone: bool | None = None,
    limite: int = 100,
) -> list[dict[str, Any]]:
    """Filtre multi-critere sur un iterateur d'entrees.

    Args:
        entries_iter: Iterateur d'entrees lexicales
        cgram: Filtre POS (prefix match)
        genre: Filtre genre (m/f)
        nombre: Filtre nombre (s/p)
        freq_min: Frequence minimale
        freq_max: Frequence maximale
        nb_syllabes: Nombre exact de syllabes
        age_min: Age d'acquisition minimal (Manulex)
        age_max: Age d'acquisition maximal (Manulex)
        illustrable_min: Score illustrable minimal (0.0-1.0)
        categorie: Categorie semantique exacte
        has_age: True = age NOT NULL, False = age IS NULL
        has_phone: True = phone NOT NULL/vide, False = phone NULL/vide
        limite: Nombre max de resultats

    Returns:
        Liste d'entrees correspondantes
    """
    results: list[dict[str, Any]] = []

    for e in entries_iter:
        if cgram is not None:
            e_cgram = str(e.get("cgram", "") or "")
            if not e_cgram.startswith(cgram):
                continue

        if genre is not None:
            if str(e.get("genre", "") or "") != genre:
                continue

        if nombre is not None:
            if str(e.get("nombre", "") or "") != nombre:
                continue

        freq = 0.0
        try:
            freq = float(e.get("freq", 0) or 0)
        except (ValueError, TypeError):
            pass

        if freq_min is not None and freq < freq_min:
            continue
        if freq_max is not None and freq > freq_max:
            continue

        if nb_syllabes is not None:
            nb = _get_nb_syllabes(e)
            if nb != nb_syllabes:
                continue

        # Filtres educatifs
        if has_age is not None:
            raw_age = e.get("age")
            age_present = raw_age is not None and raw_age != ""
            if has_age and not age_present:
                continue
            if not has_age and age_present:
                continue

        if age_min is not None or age_max is not None:
            raw_age = e.get("age")
            if raw_age is None or raw_age == "":
                continue
            try:
                age_val = float(raw_age)
            except (ValueError, TypeError):
                continue
            if age_min is not None and age_val < age_min:
                continue
            if age_max is not None and age_val > age_max:
                continue

        if illustrable_min is not None:
            raw_ill = e.get("illustrable")
            if raw_ill is None or raw_ill == "":
                continue
            try:
                ill_val = float(raw_ill)
            except (ValueError, TypeError):
                continue
            if ill_val < illustrable_min:
                continue

        if categorie is not None:
            if str(e.get("categorie", "") or "") != categorie:
                continue

        if has_phone is not None:
            phone = str(e.get("phone", "") or "")
            phone_present = bool(phone.strip())
            if has_phone and not phone_present:
                continue
            if not has_phone and phone_present:
                continue

        results.append(e)
        if limite and len(results) >= limite:
            break

    return results


def anagrammes(
    ortho_index: dict[str, list[dict[str, Any]]],
    mot: str,
    limite: int = 50,
) -> list[dict[str, Any]]:
    """Mots avec les memes lettres rearrangees.

    Exclut le mot lui-meme. Trie par frequence.

    Args:
        ortho_index: Index ortho -> [entrees]
        mot: Mot de reference
        limite: Nombre max de resultats

    Returns:
        Liste d'entrees anagrammes
    """
    mot_lower = mot.strip().lower()
    mot_sorted = sorted(mot_lower)
    results: list[dict[str, Any]] = []

    for ortho, entries in ortho_index.items():
        if ortho == mot_lower:
            continue
        if sorted(ortho) == mot_sorted:
            # Prendre la premiere entree du groupe
            if entries:
                results.append(entries[0])

    results.sort(key=lambda x: -float(x.get("freq", 0) or 0))
    return results[:limite]


def _get_nb_syllabes(entry: dict[str, Any]) -> int | None:
    """Extrait le nombre de syllabes d'une entree."""
    nb_raw = entry.get("nb_syllabes") or entry.get("nbsyll")
    if nb_raw is not None:
        try:
            return int(nb_raw)
        except (ValueError, TypeError):
            pass

    syllabes = str(entry.get("syllabes", "") or entry.get("syll", "") or "")
    if syllabes:
        return syllabes.count(".") + 1

    return None
