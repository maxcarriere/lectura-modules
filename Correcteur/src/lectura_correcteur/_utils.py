"""Utilitaires partages pour le correcteur."""

from __future__ import annotations

import re

PUNCT_RE = re.compile(r'^[,;:!?.\u2026\u00ab\u00bb"()\[\]{}\u2013\u2014/]+$')

_NO_SPACE_BEFORE = frozenset(",.;:!?)\u00bb\u2026")
_NO_SPACE_AFTER = frozenset("(\u00ab")


def transferer_casse(original: str, nouveau: str) -> str:
    """Transfere le pattern de casse de l'original vers le nouveau mot."""
    if not nouveau:
        return nouveau
    if original.isupper():
        return nouveau.upper()
    if len(original) > 1 and original[0].isupper() and original[1:].islower():
        return nouveau[0].upper() + nouveau[1:]
    if len(original) == 1 and original[0].isupper():
        return nouveau[0].upper() + nouveau[1:]
    return nouveau


def reconstruire_phrase(tokens: list[str]) -> str:
    """Reconstruit une phrase a partir des tokens en gerant les espaces."""
    if not tokens:
        return ""

    parts = [tokens[0]]
    for i in range(1, len(tokens)):
        tok = tokens[i]
        prev = tokens[i - 1]

        if tok and tok[0] in _NO_SPACE_BEFORE:
            parts.append(tok)
        elif prev and prev[-1] in _NO_SPACE_AFTER:
            parts.append(tok)
        elif prev and prev.endswith(("'", "\u2019")):
            parts.append(tok)
        elif tok.startswith("'") or tok.startswith("\u2019"):
            parts.append(tok)
        else:
            parts.append(" ")
            parts.append(tok)

    return "".join(parts)
