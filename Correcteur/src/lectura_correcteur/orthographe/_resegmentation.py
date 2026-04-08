"""Resegmentation des tokens avec apostrophes/espaces manquants.

Pre-traitement pour le langage SMS/informel ou les apostrophes sont
souvent omises : "narrive" -> "n'arrive", "cest" -> "c'est", "jai" -> "j'ai".
"""

from __future__ import annotations

from typing import Any

_CLITIQUES = [
    "qu'", "quelqu'", "lorsqu'", "puisqu'", "jusqu'",
    "presqu'", "aujourd'",
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
            continue

        split = _tenter_split_elargi(token, lexique)
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

        # Prefixes longs (>=2 chars) : reste doit faire >= 3
        # Prefixes courts (1 char) : reste doit faire >= 2
        min_reste = 2 if len(prefix_sans) == 1 else 3
        if len(reste) < min_reste:
            continue

        if len(prefix_sans) == 1:
            first_char = reste[0].lower()
            if first_char not in _VOYELLES and first_char != "h":
                continue

        if lexique.existe(reste):
            clitique = token[:len(prefix_sans)] + "'"
            return [clitique, reste]

    return None


def _tenter_split_elargi(token: str, lexique: Any) -> list[str] | None:
    """Split elargi pour prefixes multi-char avec reste court.

    Gere "quil" -> "qu'il" ou le reste (2 chars) est trop court pour
    la regle standard (min 3 pour prefixes multi-char).
    Garde la contrainte voyelle/h pour les prefixes d'1 char.
    """
    token_low = token.lower()

    for prefix_sans, prefix_avec in _CLITIQUES_SANS_APOS:
        if not token_low.startswith(prefix_sans):
            continue

        reste = token[len(prefix_sans):]

        if len(reste) < 2:
            continue

        # Pour les prefixes d'1 char, garder la contrainte voyelle/h
        # (ces cas sont deja geres par _tenter_split_clitique)
        if len(prefix_sans) == 1:
            first_char = reste[0].lower()
            if first_char not in _VOYELLES and first_char != "h":
                continue

        if lexique.existe(reste):
            clitique = token[:len(prefix_sans)] + "'"
            return [clitique, reste]

    return None
