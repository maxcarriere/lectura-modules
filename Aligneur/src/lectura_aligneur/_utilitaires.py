"""Utilitaires phonologiques IPA.

Fonctions de classification des phonemes (voyelle, consonne, semi-voyelle)
et decoupage de chaines IPA en phonemes.
"""

from __future__ import annotations

import unicodedata

# Sets de phonemes IPA du francais (statiques, pas besoin du JSON serveur)
_VOYELLES: set[str] = {
    "a", "e", "i", "o", "u", "y",
    "\u0251",         # ɑ
    "\u0254",         # ɔ
    "\u0259",         # ə
    "\u025b",         # ɛ
    "\u025c",         # ɜ
    "\u0153",         # œ
    "\u00f8",         # ø
    "\u0251\u0303",   # ɑ̃
    "\u025b\u0303",   # ɛ̃
    "\u0254\u0303",   # ɔ̃
    "\u0153\u0303",   # œ̃
}

_CONSONNES: set[str] = {
    "b", "d", "f", "g", "k", "l", "m", "n", "p", "s", "t", "v", "z",
    "\u0261",   # ɡ (IPA g)
    "\u0281",   # ʁ
    "\u0283",   # ʃ
    "\u0292",   # ʒ
    "\u014b",   # ŋ
    "\u0272",   # ɲ
    "\u0263",   # ɣ
    "x",
    "h",
}

_SEMI_VOYELLES: set[str] = {
    "j",
    "w",
    "\u0265",   # ɥ
}


def iter_phonemes(ipa: str) -> list[str]:
    """Decoupe une chaine IPA en phonemes, regroupant les combining marks Unicode.

    >>> iter_phonemes("\u0283a")
    ['\u0283', 'a']
    >>> iter_phonemes("\u0251\u0303")
    ['\u0251\u0303']
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
    """Vrai si le phoneme est une voyelle IPA (orale ou nasale)."""
    if not phoneme:
        return False
    if phoneme in _VOYELLES:
        return True
    return bool(phoneme[0] in _VOYELLES)


def est_consonne(phoneme: str) -> bool:
    """Vrai si le phoneme est une consonne IPA."""
    return bool(phoneme and phoneme in _CONSONNES)


def est_semi_voyelle(phoneme: str) -> bool:
    """Vrai si le phoneme est une semi-voyelle IPA."""
    return bool(phoneme and phoneme in _SEMI_VOYELLES)
