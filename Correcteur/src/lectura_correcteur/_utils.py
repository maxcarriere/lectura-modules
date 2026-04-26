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


# Normalisation morpho : Lexique v4 utilise des valeurs longues,
# le correcteur attend des codes courts.
_MORPHO_NORM: dict[str, str] = {
    # genre
    "masculin": "m", "feminin": "f", "féminin": "f",
    # nombre
    "singulier": "s", "pluriel": "p",
    # mode
    "indicatif": "ind", "subjonctif": "sub", "imperatif": "imp",
    "impératif": "imp", "conditionnel": "con", "infinitif": "inf",
    "participe": "par",
    # temps
    "present": "pre", "présent": "pre", "imparfait": "imp",
    "passe": "pas", "passé": "pas", "futur": "fut",
    "passe simple": "pas", "passé simple": "pas",
    "passe compose": "pac", "passé composé": "pac",
}


def normaliser_morpho(val: str) -> str:
    """Normalise une valeur morpho (longue → code court)."""
    return _MORPHO_NORM.get(val, val)


def normaliser_info(entry: dict) -> dict:
    """Normalise une entree lexique.info() pour compat v4."""
    result = dict(entry)
    for key in ("genre", "nombre", "temps", "mode"):
        v = result.get(key)
        if v and isinstance(v, str):
            result[key] = normaliser_morpho(v)
    return result


class LexiqueNormalise:
    """Wrapper qui normalise les valeurs morpho retournees par lexique.info().

    Compatible avec Lexique v3 (codes courts) et v4 (valeurs longues).
    Delegue tout au lexique sous-jacent.
    """

    def __init__(self, lexique):
        self._lexique = lexique

    def __getattr__(self, name):
        return getattr(self._lexique, name)

    def info(self, mot):
        raw = self._lexique.info(mot)
        return [normaliser_info(e) for e in raw]


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
