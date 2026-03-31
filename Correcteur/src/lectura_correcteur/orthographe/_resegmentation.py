"""Resegmentation des tokens avec apostrophes/espaces manquants.

Pre-traitement pour le langage SMS/informel ou les apostrophes sont
souvent omises : "narrive" -> "n'arrive", "cest" -> "c'est", "jai" -> "j'ai".
"""

from __future__ import annotations

from typing import Any

_CLITIQUES = [
    "qu'", "quelqu'", "lorsqu'", "puisqu'", "jusqu'",
    "c'", "d'", "j'", "l'", "m'", "n'", "s'", "t'",
]

_CLITIQUES_SANS_APOS = [(c.replace("'", ""), c) for c in _CLITIQUES]
_CLITIQUES_SANS_APOS.sort(key=lambda x: -len(x[0]))

_VOYELLES = set("aeiouy\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb"
                "\u00ef\u00ee\u00f4\u00f9\u00fb\u00fc\u00e6\u0153")


def resegmenter(tokens: list[str], lexique: Any) -> list[str]:
    """Resegmente les tokens en separant les apostrophes manquantes."""
    result: list[str] = []

    for token in tokens:
        if lexique.existe(token):
            result.append(token)
            continue

        if len(token) < 3:
            result.append(token)
            continue

        split = _tenter_split_clitique(token, lexique)
        if split:
            result.extend(split)
        else:
            result.append(token)

    return result


def _tenter_split_clitique(token: str, lexique: Any) -> list[str] | None:
    """Tente de separer un clitique du debut du token."""
    token_low = token.lower()

    for prefix_sans, prefix_avec in _CLITIQUES_SANS_APOS:
        if not token_low.startswith(prefix_sans):
            continue

        reste = token[len(prefix_sans):]

        if len(reste) < 3:
            continue

        if len(prefix_sans) == 1:
            first_char = reste[0].lower()
            if first_char not in _VOYELLES and first_char != "h":
                continue

        if lexique.existe(reste):
            clitique = token[:len(prefix_sans)] + "'"
            return [clitique, reste]

    return None
