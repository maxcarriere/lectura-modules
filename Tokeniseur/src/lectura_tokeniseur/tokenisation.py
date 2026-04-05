"""Tokenisation — Passe 1 (scan, fusion mots composés, locutions)."""

from __future__ import annotations

import re

from lectura_tokeniseur.models import (
    Token, TokenType, Mot, Ponctuation, Separateur, Formule, FormuleType,
)

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

# Élision : formes courtes avant apostrophe qui restent séparées
_ELISION_RE = re.compile(
    r"^(?:[CcDdJjLlMmNnSsTt]|[Qq]u|[Jj]usqu|[Ll]orsqu|[Pp]uisqu|[Qq]uelqu)$"
)

# Locutions figées : élision + apostrophe + composé traités comme un seul mot
_LOCUTIONS: frozenset[str] = frozenset({
    "c'est-à-dire",
})

# Caractères autorisés dans un token FORMULE brut (chiffres, lettres, opérateurs, etc.)
_FORMULE_CHARS = set("0123456789.,'+-=×*÷/^√<>%°±²³⁰¹⁴⁵⁶⁷⁸⁹ⁿ(){}[]|~!@#_")
# Lettres grecques courantes en maths
_GREEK_LETTERS = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")


def _is_letter(ch: str) -> bool:
    return bool(_LETTER_RE.match(ch))


def _is_digit(ch: str) -> bool:
    return ch.isdigit()


def _is_formule_start(ch: str) -> bool:
    """Vrai si le caractère peut commencer un token FORMULE brut."""
    return ch.isdigit() or ch in _GREEK_LETTERS or ch in "+-=√~"


def _is_formule_char(ch: str, prev: str = "") -> bool:
    """Vrai si le caractère peut faire partie d'un token FORMULE brut."""
    if ch.isdigit():
        return True
    if ch in _FORMULE_CHARS or ch in _GREEK_LETTERS:
        return True
    # Lettres adjacentes à des chiffres (ex: 2x, sin, km)
    if ch.isalpha() and ch not in {"'", "-"}:
        return True
    return False


def _merge_compounds(tokens: list[Token]) -> list[Token]:
    """Fusionne les séquences Mot+Sep+Mot en mots composés.

    Règles :
    - Tiret entre lettres → toujours fusionner
    - Apostrophe entre lettres → fusionner sauf si le premier mot est une élision
    - Une fois la fusion démarrée par un tiret, les apostrophes internes
      ne déclenchent plus la coupure élision (ex: chef-d'œuvre)
    """
    if len(tokens) < 3:
        return tokens

    result: list[Token] = []
    i = 0
    n = len(tokens)

    while i < n:
        # Chercher un pattern Mot + Sep(apostrophe|hyphen) + Mot
        if (
            i + 2 < n
            and isinstance(tokens[i], Mot)
            and isinstance(tokens[i + 1], Separateur)
            and tokens[i + 1].sep_type in ("apostrophe", "hyphen")
            and isinstance(tokens[i + 2], Mot)
        ):
            sep = tokens[i + 1]
            mot_avant = tokens[i]

            # Apostrophe : ne pas fusionner si c'est une élision
            if sep.sep_type == "apostrophe" and _ELISION_RE.match(mot_avant.text):
                result.append(tokens[i])
                i += 1
                continue

            # Démarrer la fusion : collecter tous les sous-tokens
            children: list[Token] = [tokens[i], tokens[i + 1], tokens[i + 2]]
            # started_with_hyphen : vrai si au moins un tiret dans le composé
            has_hyphen = sep.sep_type == "hyphen"
            j = i + 3

            # Continuer à fusionner tant qu'on trouve Sep + Mot
            while (
                j + 1 < n
                and isinstance(tokens[j], Separateur)
                and tokens[j].sep_type in ("apostrophe", "hyphen")
                and isinstance(tokens[j + 1], Mot)
            ):
                inner_sep = tokens[j]
                inner_mot = tokens[j + 1]

                # Apostrophe interne : couper si élision ET pas de tiret dans le composé
                if (
                    inner_sep.sep_type == "apostrophe"
                    and not has_hyphen
                    and _ELISION_RE.match(inner_mot.text)
                ):
                    break

                if inner_sep.sep_type == "hyphen":
                    has_hyphen = True

                children.append(inner_sep)
                children.append(inner_mot)
                j += 2

            # Construire le Mot composé
            full_text = "".join(c.text for c in children)
            compound = Mot(
                type=TokenType.MOT,
                text=full_text,
                span=(children[0].span[0], children[-1].span[1]),
                ortho=full_text.lower(),
                children=children,
            )
            result.append(compound)
            i = j
        else:
            result.append(tokens[i])
            i += 1

    return result


def _merge_locutions(tokens: list[Token]) -> list[Token]:
    """Fusionne élision + apostrophe + composé quand ils forment une locution connue.

    Passe post-merge qui rattrape les locutions figées comme « c'est-à-dire »
    où l'élision fait partie intégrante du mot composé.
    """
    if len(tokens) < 3:
        return tokens

    result: list[Token] = []
    i = 0
    n = len(tokens)

    while i < n:
        if (
            i + 2 < n
            and isinstance(tokens[i], Mot)
            and isinstance(tokens[i + 1], Separateur)
            and tokens[i + 1].sep_type == "apostrophe"
            and isinstance(tokens[i + 2], Mot)
        ):
            candidate = (
                tokens[i].text + tokens[i + 1].text + tokens[i + 2].text
            ).lower()
            if candidate in _LOCUTIONS:
                children = [tokens[i], tokens[i + 1], tokens[i + 2]]
                full_text = tokens[i].text + tokens[i + 1].text + tokens[i + 2].text
                compound = Mot(
                    type=TokenType.MOT,
                    text=full_text,
                    span=(tokens[i].span[0], tokens[i + 2].span[1]),
                    ortho=full_text.lower(),
                    children=children,
                )
                result.append(compound)
                i += 3
                continue

        result.append(tokens[i])
        i += 1

    return result


def _inside_brackets(text: str, pos: int) -> bool:
    """Vérifie si *pos* est à l'intérieur de crochets ou accolades ouverts."""
    depth = 0
    for k in range(pos - 1, -1, -1):
        if text[k] in "]})":
            depth += 1
        elif text[k] in "[{(":
            if depth == 0:
                return True
            depth -= 1
    return False


def _scan_tokens(text: str) -> list[Token]:
    """Passe 1 : découpe le texte en tokens bruts (MOT, PONCTUATION, SEPARATEUR, FORMULE brut).

    Tout ce qui contient des chiffres ou des caractères de formule → FORMULE brut.
    Les lettres pures → MOT.
    """
    if not text:
        return []

    tokens: list[Token] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # ── Chiffres / début de formule → FORMULE brut ──
        if ch.isdigit():
            start = i
            while i < n and (text[i].isdigit() or text[i] in (".", "'", ",")):
                # Ne pas absorber la virgule si on est dans un contexte ensemble/intervalle
                if text[i] == "," and _inside_brackets(text, start):
                    break
                i += 1
            # Reculer si on finit par . ou ' ou , (ponctuation)
            while i > start and text[i - 1] in (".", "'", ","):
                i -= 1
            # 3'25" → ne pas absorber l'apostrophe (minutes/secondes d'arc)
            scanned = text[start:i]
            if "'" in scanned and i < n and text[i] == '"':
                i = start + scanned.index("'")
            tok = Formule(
                type=TokenType.FORMULE,
                text=text[start:i],
                span=(start, i),
                formule_type=FormuleType.NOMBRE,
                valeur=text[start:i],
            )
            tokens.append(tok)
            continue

        # ── Lettres consécutives → MOT ──
        if _is_letter(ch):
            start = i
            while i < n and _is_letter(text[i]):
                i += 1
            word = text[start:i]
            tok = Mot(
                type=TokenType.MOT,
                text=word,
                span=(start, i),
                ortho=word.lower(),
            )
            tokens.append(tok)
            continue

        # ── Apostrophe entre lettres → Separateur ──
        if ch == "'":
            prev_letter = i > 0 and _is_letter(text[i - 1])
            next_letter = i + 1 < n and _is_letter(text[i + 1])
            if prev_letter and next_letter:
                tok = Separateur(
                    type=TokenType.SEPARATEUR,
                    text="'",
                    span=(i, i + 1),
                    sep_type="apostrophe",
                )
                tokens.append(tok)
                i += 1
                continue

        # ── Tiret entre lettres → Separateur ──
        if ch == "-":
            prev_letter = i > 0 and _is_letter(text[i - 1])
            next_letter = i + 1 < n and _is_letter(text[i + 1])
            if prev_letter and next_letter:
                tok = Separateur(
                    type=TokenType.SEPARATEUR,
                    text="-",
                    span=(i, i + 1),
                    sep_type="hyphen",
                )
                tokens.append(tok)
                i += 1
                continue

        # ── Espace → Separateur(sep_type="space") ──
        if ch == " ":
            tok = Separateur(
                type=TokenType.SEPARATEUR,
                text=" ",
                span=(i, i + 1),
                sep_type="space",
            )
            tokens.append(tok)
            i += 1
            continue

        # ── √ → Formule MATHS (pour permettre le merge avec le nombre adjacent) ──
        if ch == "√":
            tok = Formule(
                type=TokenType.FORMULE,
                text="√",
                span=(i, i + 1),
                formule_type=FormuleType.MATHS,
            )
            tokens.append(tok)
            i += 1
            continue

        # ── Tout le reste → Ponctuation ──
        tok = Ponctuation(
            type=TokenType.PONCTUATION,
            text=ch,
            span=(i, i + 1),
        )
        tokens.append(tok)
        i += 1

    return tokens
