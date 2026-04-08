"""Regles d'elision : contracter les particules elidables devant voyelle/h.

"parce que il" -> "parce qu'il", "je arrive" -> "j'arrive", etc.
"""

from __future__ import annotations

_ELIDABLES: dict[str, str] = {
    "que": "qu'", "je": "j'", "me": "m'", "te": "t'",
    "se": "s'", "le": "l'", "la": "l'", "de": "d'",
    "ne": "n'",
}

_VOYELLES = frozenset("aeiouyàâäéèêëïîôùûüæœh")


def appliquer_elision(tokens: list[str]) -> list[str]:
    """Contracte les particules elidables devant voyelle/h muet."""
    result: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            low = tokens[i].lower()
            next_tok = tokens[i + 1]
            next_low = next_tok.lower()
            if low in _ELIDABLES and next_low and next_low[0] in _VOYELLES:
                elide = _ELIDABLES[low]
                result.append(elide)
                result.append(next_tok)
                i += 2
                continue
        result.append(tokens[i])
        i += 1
    return result
