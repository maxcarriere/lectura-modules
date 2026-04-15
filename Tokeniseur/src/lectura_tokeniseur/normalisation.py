"""Normalisation typographique pour le français."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_RE_SPACES = re.compile(r"[ \t\u00A0]+")
_RE_SPACE_AROUND_APOSTROPHE = re.compile(r"\s*'\s*")

# Ponctuation faible : pas d'espace avant, un espace après
_RE_NO_SPACE_BEFORE = re.compile(r"\s+([,.;])")
# Un espace après ponctuation faible, sauf point entre chiffres (3.14)
_RE_ONE_SPACE_AFTER = re.compile(r"([,.;])(?=[^\s\d])")

# Ponctuation forte : espace avant et après (groupes)
_RE_STRONG_PUNCT_GROUP = re.compile(r"([!?;:]+)")

# Nombres : séquences de chiffres avec espaces
_RE_DECIMAL_COMMA = re.compile(r"(\d),(\d)")
_RE_NUMBER_WITH_SPACES = re.compile(r"\d+(?:[\s']\d+)+")

# Guillemets droits
_RE_QUOTES = re.compile(r'"([^"]*)"')

# Parenthèses / crochets (espaces internes)
_RE_PAREN_INNER = re.compile(r"\(\s*(.*?)\s*\)")
_RE_BRACKET_INNER = re.compile(r"\[\s*(.*?)\s*\]")

_OPENING_PUNCT = {"(", "[", "«", "\n"}


def _normalize_spaces(text: str) -> str:
    text = _RE_SPACES.sub(" ", text)
    return text.strip()


def _normalize_apostrophes(text: str) -> str:
    return _RE_SPACE_AROUND_APOSTROPHE.sub("'", text)


def _normalize_basic_punctuation(text: str) -> str:
    # Ponctuation faible
    text = _RE_NO_SPACE_BEFORE.sub(r"\1", text)
    text = _RE_ONE_SPACE_AFTER.sub(r"\1 ", text)

    # Ponctuation forte : traiter par groupes
    def _repl_strong(match: re.Match) -> str:
        grp = match.group(1)
        # Factorielle : ne pas ajouter d'espace autour de ! si précédé d'alphanum
        if grp == "!" and match.start() > 0 and text[match.start() - 1].isalnum():
            return "!"
        return f" {grp} "

    text = _RE_STRONG_PUNCT_GROUP.sub(_repl_strong, text)

    return _normalize_spaces(text)


def _normalize_numbers(text: str) -> str:
    """Normalise les nombres avec séparateurs.

    Protège les téléphones (0X XX XX XX XX), numéros et autres formules.
    Seuls les groupes clairement numériques (groupes de 3 chiffres) sont normalisés.
    """
    # Virgule décimale → point (seulement si chiffre,chiffre isolé)
    text = _RE_DECIMAL_COMMA.sub(r"\1.\2", text)

    def _repl_number(match: re.Match) -> str:
        raw = match.group(0)
        # Ne pas toucher si la séquence contient des lettres
        if re.search(r"[a-zA-Z]", raw):
            return raw
        digits = re.sub(r"[ '\s]", "", raw)
        # Protéger les téléphones : 10 chiffres commençant par 0X
        if len(digits) == 10 and digits[0] == "0" and digits[1] != "0":
            return raw
        # Protéger les groupes de 2 chiffres séparés par espaces (pattern téléphone)
        groups = raw.split()
        if all(len(g) == 2 and g.isdigit() for g in groups) and len(groups) >= 3:
            return raw
        # Protéger les groupes irréguliers (pas de pattern standard de milliers)
        # Un nombre standard a des groupes de 3 chiffres (sauf le premier)
        if len(groups) >= 2:
            # Vérifier si tous les groupes sauf le premier font 3 chiffres
            standard_number = all(
                len(g.replace("'", "")) == 3 for g in groups[1:]
            )
            if not standard_number:
                return raw  # Garder tel quel (probable numéro/code)
        # Même vérification pour les groupes séparés par apostrophe (3'25 ≠ milliers)
        if "'" in raw and " " not in raw:
            apos_groups = raw.split("'")
            if not all(len(g) == 3 for g in apos_groups[1:]):
                return raw  # Pas un format de milliers standard
        if len(digits) <= 5:
            return digits
        parts: list[str] = []
        while digits:
            parts.append(digits[-3:])
            digits = digits[:-3]
        return "'".join(reversed(parts))

    return _RE_NUMBER_WITH_SPACES.sub(_repl_number, text)


def _normalize_quotes(text: str) -> str:
    return _RE_QUOTES.sub(r"« \1 »", text)


def _normalize_parentheses(text: str) -> str:
    text = _RE_PAREN_INNER.sub(r"(\1)", text)
    text = _RE_BRACKET_INNER.sub(r"[\1]", text)
    return text


def _normalize_ellipsis(text: str) -> str:
    """Remplace ... par … avec espacement contextuel."""
    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i: i + 3] == "...":
            prev = text[i - 1] if i > 0 else None

            if i == 0 or prev in _OPENING_PUNCT:
                result.append("…")
            else:
                if prev != " ":
                    result.append(" ")
                result.append("…")

            if i + 3 < n and text[i + 3] != " ":
                result.append(" ")

            i += 3
            continue

        result.append(text[i])
        i += 1

    return "".join(result)


def _normalize_dashes(text: str) -> str:
    """Normalisation graphique des tirets.

    - mot-mot → inchangé (séparateur de mots composés)
    - chiffre/eE-chiffre → inchangé (notation scientifique, formule)
    - tiret isolé ou en début/fin → espace autour
    """
    chars = list(text)
    n = len(chars)

    for i, ch in enumerate(chars):
        if ch != "-":
            continue

        prev = chars[i - 1] if i > 0 else ""
        next_ = chars[i + 1] if i + 1 < n else ""

        # Séparateur de mot : lettre-lettre → inchangé
        if prev.isalpha() and next_.isalpha():
            continue

        # Notation scientifique / formule : chiffre-chiffre ou eE-chiffre → inchangé
        if next_.isdigit() and (prev.isdigit() or prev.lower() in ("e",)):
            continue
        # Signe négatif devant un chiffre en début ou après opérateur : -12, (-3
        if next_.isdigit() and (prev == "" or prev in " ([:=<>≤≥≠,;"):
            continue
        # Signe négatif devant un symbole maths (√, ∑, ∏, ∫) : -√3
        if next_ in "√∑∏∫∂∇" and (prev == "" or prev in " ([:=<>≤≥≠,;"):
            continue
        # Signe négatif devant une variable (1 lettre seule) : -x, -a
        # Mais pas devant un mot : -bonjour doit être normalisé
        if next_.isalpha() and (prev == "" or prev in " ([:=<>≤≥≠,;"):
            after = chars[i + 2] if i + 2 < n else ""
            if not after.isalpha():
                continue
        # Opérateur dans formule : chiffre/lettre + op + chiffre/lettre (ex: 3+5, x-2)
        _MATH_SYMS = "√∑∏∫∂∇∞"
        if ((prev.isdigit() or prev.isalpha()) and
                (next_.isdigit() or next_.isalpha() or next_ in _MATH_SYMS)):
            continue

        # Dialogue ou ponctuation : forcer espace
        if prev != " ":
            chars[i] = " -"
        if next_ != " ":
            chars[i] = chars[i] + " "

    return _normalize_spaces("".join(chars))


# Protège les ensembles {…} et intervalles avec crochets contenant ; ou ,
# pour que la normalisation ne les altère pas.
_RE_BRACE_CONTENT = re.compile(r"\{[^}]*\}")
_RE_BRACKET_INTERVAL = re.compile(r"([\[\]])([^[\]]*[;,][^[\]]*)([\[\]])")


def _protect_sets_intervals(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Remplace les ensembles/intervalles par des placeholders avant normalisation."""
    placeholders: list[tuple[str, str]] = []
    counter = 0

    def _replace(m: re.Match) -> str:
        nonlocal counter
        original = m.group(0)
        ph = f"\x00SET{counter}\x00"
        placeholders.append((ph, original))
        counter += 1
        return ph

    text = _RE_BRACE_CONTENT.sub(_replace, text)
    text = _RE_BRACKET_INTERVAL.sub(_replace, text)
    return text, placeholders


def _restore_sets_intervals(text: str, placeholders: list[tuple[str, str]]) -> str:
    """Restaure les placeholders par le texte original."""
    for ph, original in placeholders:
        text = text.replace(ph, original)
    return text


def normalise(text: str) -> str:
    """Normalise le texte brut avant tokenisation.

    Applique dans l'ordre : espaces, apostrophes, ellipses,
    ponctuation, nombres, guillemets, parenthèses, tirets.

    Args:
        text: Texte brut à normaliser.

    Returns:
        Texte normalisé.
    """
    logger.debug("normalise() called, input length=%s", len(text) if text else 0)
    if not text:
        return text

    # Protéger les ensembles {…} et intervalles [… ; …] / ]… ; …[
    text, placeholders = _protect_sets_intervals(text)

    text = _normalize_spaces(text)
    text = _normalize_apostrophes(text)
    text = _normalize_ellipsis(text)
    text = _normalize_numbers(text)
    text = _normalize_basic_punctuation(text)
    text = _normalize_quotes(text)
    text = _normalize_parentheses(text)
    text = _normalize_dashes(text)

    # Restaurer les ensembles/intervalles protégés
    text = _restore_sets_intervals(text, placeholders)

    return text
