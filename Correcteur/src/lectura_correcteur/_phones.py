"""Generateur de variantes phonetiques proches.

Deplace depuis lectura-lexique (specifique au correcteur).
"""

from __future__ import annotations

import unicodedata

# Confusions vocaliques courantes (SMS/informel)
_PHONE_VARIANTES: dict[str, list[str]] = {
    "e": ["e", "\u025b", "\u0259"],       # e <-> ɛ <-> ə
    "\u025b": ["\u025b", "e", "\u0259"],   # ɛ <-> e <-> ə
    "\u0259": ["\u0259", "e", "\u025b"],   # ə <-> e <-> ɛ
    "o": ["o", "\u0254"],        # o <-> ɔ
    "\u0254": ["\u0254", "o"],   # ɔ <-> o
    "\u00f8": ["\u00f8", "\u0153"],  # ø <-> œ
    "\u0153": ["\u0153", "\u00f8"],  # œ <-> ø
}


def _segmenter_ipa(phone: str) -> list[str]:
    """Decoupe une chaine IPA en segments (char + diacritiques combinants)."""
    segments: list[str] = []
    i = 0
    while i < len(phone):
        seg = phone[i]
        i += 1
        while i < len(phone) and unicodedata.category(phone[i]).startswith("M"):
            seg += phone[i]
            i += 1
        segments.append(seg)
    return segments


def generer_phones_proches(phone: str) -> list[str]:
    """Genere les variantes proches d'un phone (e<->ɛ, o<->ɔ, etc.).

    Remplace chaque voyelle ambigue par ses variantes et retourne
    la liste de tous les phones possibles (incluant l'original).
    Limite a 1 substitution a la fois.
    """
    segments = _segmenter_ipa(phone)
    results = {phone}

    for idx, seg in enumerate(segments):
        base = seg[0] if seg else seg
        if base in _PHONE_VARIANTES:
            for v in _PHONE_VARIANTES[base]:
                new_seg = v + seg[1:]  # conserver diacritiques
                new_segs = segments[:idx] + [new_seg] + segments[idx + 1:]
                results.add("".join(new_segs))

    return list(results)


def generer_phones_d1(phone: str) -> list[str]:
    """Genere les variantes a distance phonetique <= 1 (deletions + substitutions vocaliques).

    Combine :
    - Suppression d'un segment IPA (ex: [zanimo] -> [animo])
    - Variantes vocaliques (ex: [dinozɔʁ] -> [dinozoʁ])
    Retourne la liste sans l'original.
    """
    segments = _segmenter_ipa(phone)
    results: set[str] = set()

    # Deletions : supprimer un segment a la fois
    for idx in range(len(segments)):
        variant = "".join(segments[:idx] + segments[idx + 1:])
        if variant and variant != phone:
            results.add(variant)

    # Substitutions vocaliques (1 a la fois)
    for idx, seg in enumerate(segments):
        base = seg[0] if seg else seg
        if base in _PHONE_VARIANTES:
            for v in _PHONE_VARIANTES[base]:
                new_seg = v + seg[1:]
                new_segs = segments[:idx] + [new_seg] + segments[idx + 1:]
                variant = "".join(new_segs)
                if variant != phone:
                    results.add(variant)

    return list(results)
