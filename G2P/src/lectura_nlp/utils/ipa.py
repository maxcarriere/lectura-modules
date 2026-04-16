"""Utilitaires IPA pour le français.

Gère les combining marks Unicode (diacritiques IPA) et fournit
des fonctions de classification phonétique.
"""

from __future__ import annotations

import unicodedata

# Inventaire phonologique standard du francais (IPA)
_VOYELLES = {"a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə"}
_CONSONNES = {"p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "ŋ", "l", "ʁ"}
_SEMI_VOYELLES = {"j", "w", "ɥ"}
_NASALES = {"m", "n", "ɲ", "ŋ"}
_LIQUIDES = {"l", "ʁ"}
_OCCLUSIVES = {"p", "b", "t", "d", "k", "ɡ"}
_FRICATIVES = {"f", "v", "s", "z", "ʃ", "ʒ"}


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
