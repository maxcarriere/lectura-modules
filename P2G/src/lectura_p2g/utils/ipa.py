"""Utilitaires IPA pour le français.

Gère les combining marks Unicode (diacritiques IPA) et fournit
des fonctions de classification phonétique.
"""

from __future__ import annotations

import unicodedata

from lectura_p2g._chargeur import (
    voyelles as _load_voyelles,
    consonnes as _load_consonnes,
    semi_voyelles as _load_semi_voyelles,
    nasales as _load_nasales,
    liquides as _load_liquides,
    occlusives as _load_occlusives,
    fricatives as _load_fricatives,
)

_VOYELLES = _load_voyelles()
_CONSONNES = _load_consonnes()
_SEMI_VOYELLES = _load_semi_voyelles()
_NASALES = _load_nasales()
_LIQUIDES = _load_liquides()
_OCCLUSIVES = _load_occlusives()
_FRICATIVES = _load_fricatives()


def iter_phonemes(ipa: str) -> list[str]:
    """Itère sur les phonèmes d'une chaîne IPA, en regroupant les combining marks."""
    if not ipa:
        return []

    phonemes: list[str] = []
    current = ""

    for ch in ipa:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            current += ch
        else:
            if current:
                phonemes.append(current)
            current = ch

    if current:
        phonemes.append(current)

    return phonemes


def _base_character(phoneme: str) -> str | None:
    if not phoneme:
        return None
    return phoneme[0]


def est_voyelle(phoneme: str) -> bool:
    if not phoneme:
        return False
    if phoneme in _VOYELLES:
        return True
    base = _base_character(phoneme)
    return bool(base and base in _VOYELLES)


def est_consonne(phoneme: str) -> bool:
    if not phoneme:
        return False
    return phoneme in _CONSONNES


def est_semi_voyelle(phoneme: str) -> bool:
    if not phoneme:
        return False
    return phoneme in _SEMI_VOYELLES
