"""Utilitaires partages pour le correcteur."""

from __future__ import annotations

import re

# Tokens de ponctuation OU numeriques (ne doivent pas etre corriges)
PUNCT_RE = re.compile(
    r'^[,;:!?.\u2026\u00ab\u00bb"()\[\]{}\u2013\u2014/*_~#`<>\u00a9\u00ae\u2122|\\^@&=+\u00b1\u00b0]+$'
    r'|^\d[\d,.°hm%/:+\-]*$'       # Nombres purs : 95, 20h, 3.14
    r'|^[+\-]?\d[\d,.°hm%/:+\-]*$'  # Nombres signes : +38°, -5
    r'|^:\d[\d.]*$'                   # Heures tronquees : :57, :02, :57.
)

_NO_SPACE_BEFORE = frozenset(",.)]\u2026")
# French typography: space before ? ! : ; » (espace insecable en francais)
_FRENCH_SPACE_BEFORE = frozenset("!?;:\u00bb")
_NO_SPACE_AFTER = frozenset("([")
# French guillemet ouvrant : espace apres (« mot)
_FRENCH_SPACE_AFTER = frozenset("\u00ab")


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
    # Sanitize genre: seules valeurs valides sont "m" et "f"
    g = result.get("genre")
    if g and g not in ("m", "f"):
        result["genre"] = ""
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

    def formes_de(self, *args, **kwargs):
        raw = self._lexique.formes_de(*args, **kwargs)
        return [normaliser_info(e) for e in raw]


def est_changement_genre(forme: str, p2g: str) -> bool:
    """True si la difference entre forme et p2g est un pattern de genre.

    Detecte les transformations fem<->masc courantes :
    -ees <-> -es (participiales), -ee <-> -e, -ues <-> -us, -ive <-> -if, etc.
    Les formes sont attendues en minuscules.
    """
    f, p = forme, p2g
    # -ees <-> -es (participiales : transformees <-> transformes)
    if (f.endswith("ées") and p == f[:-3] + "és") or \
       (p.endswith("ées") and f == p[:-3] + "és"):
        return True
    # -ee <-> -e (singulier : transformee <-> transforme)
    if (f.endswith("ée") and p == f[:-2] + "é") or \
       (p.endswith("ée") and f == p[:-2] + "é"):
        return True
    # -ues <-> -us (connues <-> connus)
    if (f.endswith("ues") and p == f[:-3] + "us") or \
       (p.endswith("ues") and f == p[:-3] + "us"):
        return True
    # -ue <-> -u (mais pas -que)
    if (f.endswith("ue") and not f.endswith("que") and p == f[:-2] + "u") or \
       (p.endswith("ue") and not p.endswith("que") and f == p[:-2] + "u"):
        return True
    # -ive <-> -if (active <-> actif)
    if (f.endswith("ive") and p == f[:-3] + "if") or \
       (p.endswith("ive") and f == p[:-3] + "if"):
        return True
    # -ives <-> -ifs (actives <-> actifs)
    if (f.endswith("ives") and p == f[:-4] + "ifs") or \
       (p.endswith("ives") and f == p[:-4] + "ifs"):
        return True
    # -ique <-> -ic (publique <-> public)
    if (f.endswith("ique") and p == f[:-4] + "ic") or \
       (p.endswith("ique") and f == p[:-4] + "ic"):
        return True
    return False


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
        elif tok and tok[0] in _FRENCH_SPACE_BEFORE:
            # French typography: space before ? ! : ; »
            parts.append(" ")
            parts.append(tok)
        elif prev and prev[-1] in _NO_SPACE_AFTER:
            parts.append(tok)
        elif prev and prev[-1] in _FRENCH_SPACE_AFTER:
            # French guillemet ouvrant : « mot (espace apres)
            parts.append(" ")
            parts.append(tok)
        elif prev and prev.endswith(("'", "\u2019")):
            parts.append(tok)
        elif tok.startswith("'") or tok.startswith("\u2019"):
            parts.append(tok)
        else:
            parts.append(" ")
            parts.append(tok)

    return "".join(parts)
