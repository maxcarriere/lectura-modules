"""Generateur de variantes phonetiques proches.

Deplace depuis lectura-lexique (specifique au correcteur).
"""

from __future__ import annotations

import unicodedata

# Confusions vocaliques courantes (SMS/informel)
_PHONE_VARIANTES: dict[str, list[str]] = {
    "e": ["e", "\u025b"],       # e <-> ɛ
    "\u025b": ["\u025b", "e"],   # ɛ <-> e
    "o": ["o", "\u0254"],        # o <-> ɔ
    "\u0254": ["\u0254", "o"],   # ɔ <-> o
    "\u00f8": ["\u00f8", "\u0153"],  # ø <-> œ
    "\u0153": ["\u0153", "\u00f8"],  # œ <-> ø
}


def generer_phones_proches(phone: str) -> list[str]:
    """Genere les variantes proches d'un phone (e<->ɛ, o<->ɔ, etc.).

    Remplace chaque voyelle ambigue par ses variantes et retourne
    la liste de tous les phones possibles (incluant l'original).
    Limite a 1 substitution a la fois.
    """
    positions: list[tuple[int, list[str]]] = []

    i = 0
    while i < len(phone):
        ch = phone[i]
        if (
            i + 1 < len(phone)
            and unicodedata.category(phone[i + 1]).startswith("M")
        ):
            i += 2
            continue
        if ch in _PHONE_VARIANTES:
            positions.append((i, _PHONE_VARIANTES[ch]))
        i += 1

    if not positions:
        return [phone]

    results = {phone}
    for pos, variantes in positions:
        for v in variantes:
            variant = phone[:pos] + v + phone[pos + 1:]
            results.add(variant)

    return list(results)
