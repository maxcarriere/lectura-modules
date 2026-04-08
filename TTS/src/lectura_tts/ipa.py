"""Utilitaires IPA pour lectura-tts.

Gère les combining marks Unicode (diacritiques IPA) et fournit
des fonctions de classification phonétique du français.
"""

from __future__ import annotations

import unicodedata


# Inventaire IPA du français (restreint au projet)
_VOYELLES: set[str] = {
    "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə",
}

_CONSONNES: set[str] = {
    "p", "b", "t", "d", "k", "ɡ",
    "f", "v", "s", "z", "ʃ", "ʒ",
    "m", "n", "ɲ", "ŋ",
    "l", "ʁ",
}

_SEMI_VOYELLES: set[str] = {"j", "w", "ɥ"}

# Sous-types consonantiques
_NASALES: set[str] = {"m", "n", "ɲ", "ŋ"}
_LIQUIDES: set[str] = {"l", "ʁ"}
_OCCLUSIVES: set[str] = {"p", "b", "t", "d", "k", "ɡ"}
_FRICATIVES: set[str] = {"f", "v", "s", "z", "ʃ", "ʒ"}


def iter_phonemes(ipa: str) -> list[str]:
    """Itère sur les phonèmes d'une chaîne IPA, en regroupant les combining marks.

    Exemples :
        >>> iter_phonemes("ʃa")
        ['ʃ', 'a']
        >>> iter_phonemes("ɑ̃")  # a + combining tilde
        ['ɑ̃']
        >>> iter_phonemes("")
        []
    """
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


def est_voyelle(phoneme: str) -> bool:
    """Vrai si le phonème est une voyelle IPA (orale ou nasale)."""
    if not phoneme:
        return False
    if phoneme in _VOYELLES:
        return True
    base = phoneme[0] if phoneme else ""
    return bool(base and base in _VOYELLES)


def est_consonne(phoneme: str) -> bool:
    """Vrai si le phonème est une consonne IPA."""
    return bool(phoneme) and phoneme in _CONSONNES


def est_semi_voyelle(phoneme: str) -> bool:
    """Vrai si le phonème est une semi-voyelle IPA."""
    return bool(phoneme) and phoneme in _SEMI_VOYELLES


def est_nasale(phoneme: str) -> bool:
    """Vrai si le phonème est une consonne nasale."""
    return bool(phoneme) and phoneme in _NASALES


def est_liquide(phoneme: str) -> bool:
    """Vrai si le phonème est une consonne liquide."""
    return bool(phoneme) and phoneme in _LIQUIDES


def est_occlusive(phoneme: str) -> bool:
    """Vrai si le phonème est une consonne occlusive."""
    return bool(phoneme) and phoneme in _OCCLUSIVES


def est_fricative(phoneme: str) -> bool:
    """Vrai si le phonème est une consonne fricative."""
    return bool(phoneme) and phoneme in _FRICATIVES
