"""Lectura Tokeniseur — Normalisateur et tokeniseur pour le français.

Fichier unique, autonome, zéro dépendance externe.

Usage rapide :
    from lectura_tokeniseur import LecturaTokeniseur

    tok = LecturaTokeniseur()
    tokens = tok.tokenize("L'enfant mange du chocolat.")
    for t in tokens:
        print(f"{t.text:12s}  {t.type.value:12s}  span={t.span}")
    # L            mot           span=(0, 1)
    # '            separateur    span=(1, 2)
    # enfant       mot           span=(2, 8)
    # ...

    # Normalisation seule
    text = tok.normalize("L'enfant  mange...du  chocolat")
    # → "L'enfant mange… du chocolat"

Pré-requis :
    - Python 3.10+
    - Aucune bibliothèque externe requise

Copyright (c) 2025 Lectura — Licence CC BY-SA 4.0.
Voir LICENCE.txt et ATTRIBUTION.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

__version__ = "1.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Modèles de données
# ══════════════════════════════════════════════════════════════════════════════

Span = tuple[int, int]


class TokenType(Enum):
    """Types de tokens reconnus."""
    MOT = "mot"
    PONCTUATION = "ponctuation"
    SEPARATEUR = "separateur"
    NOMBRE = "nombre"
    SIGLE = "sigle"


@dataclass
class Token:
    """Token de base avec type, texte et position dans le texte source."""
    type: TokenType
    text: str
    span: Span


@dataclass
class Mot(Token):
    """Token de type MOT (séquence de lettres).

    Attributs :
        ortho : forme orthographique en minuscules
        children : sous-tokens pour les mots composés (tiret/apostrophe)
    """
    ortho: str = ""
    children: list[Token] = field(default_factory=list)


@dataclass
class Ponctuation(Token):
    """Token de type PONCTUATION (virgule, point, etc.)."""
    pass


@dataclass
class Separateur(Token):
    """Token de type SEPARATEUR (apostrophe, trait d'union, espace).

    Attributs :
        sep_type : "apostrophe" | "hyphen" | "space"
    """
    sep_type: str | None = None


@dataclass
class Nombre(Token):
    """Token de type NOMBRE (séquence de chiffres, éventuellement avec . ou ')."""
    pass


@dataclass
class Sigle(Token):
    """Token de type SIGLE (suite de lettres majuscules épelées individuellement).

    Attributs :
        children : liste de Token (un Mot par lettre)
    """
    children: list[Token] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation
# ══════════════════════════════════════════════════════════════════════════════

_RE_SPACES = re.compile(r"[ \t\u00A0]+")
_RE_SPACE_AROUND_APOSTROPHE = re.compile(r"\s*'\s*")

# Ponctuation faible : pas d'espace avant, un espace après
_RE_NO_SPACE_BEFORE = re.compile(r"\s+([,.;])")
# Un espace après ponctuation faible, sauf point entre chiffres (3.14)
_RE_ONE_SPACE_AFTER = re.compile(r"([,.;])(?=[^\s\d])")

# Ponctuation forte : espace avant et après (groupes)
_RE_STRONG_PUNCT_GROUP = re.compile(r"([!?;:]+)")

# Nombres
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
        return f" {grp} "

    text = _RE_STRONG_PUNCT_GROUP.sub(_repl_strong, text)

    return _normalize_spaces(text)


def _normalize_numbers(text: str) -> str:
    # Virgule décimale → point
    text = _RE_DECIMAL_COMMA.sub(r"\1.\2", text)

    def _repl_number(match: re.Match) -> str:
        raw = match.group(0)
        digits = re.sub(r"[ '\s]", "", raw)
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

        # Dialogue ou ponctuation : forcer espace
        if prev != " ":
            chars[i] = " -"
        if next_ != " ":
            chars[i] = chars[i] + " "

    return _normalize_spaces("".join(chars))


def normalise(text: str) -> str:
    """Normalise le texte brut avant tokenisation.

    Applique dans l'ordre : espaces, apostrophes, ellipses,
    ponctuation, nombres, guillemets, parenthèses, tirets.

    Args:
        text: Texte brut à normaliser.

    Returns:
        Texte normalisé.
    """
    if not text:
        return text

    text = _normalize_spaces(text)
    text = _normalize_apostrophes(text)
    text = _normalize_ellipsis(text)
    text = _normalize_numbers(text)
    text = _normalize_basic_punctuation(text)
    text = _normalize_quotes(text)
    text = _normalize_parentheses(text)
    text = _normalize_dashes(text)

    return text


# ══════════════════════════════════════════════════════════════════════════════
# Tokenisation
# ══════════════════════════════════════════════════════════════════════════════

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

# Détection sigle : 2+ lettres majuscules consécutives
_SIGLE_RE = re.compile(r"^[A-ZÀ-Ö]{2,}$")

# Élision : formes courtes avant apostrophe qui restent séparées
_ELISION_RE = re.compile(
    r"^(?:[CcDdJjLlMmNnSsTt]|[Qq]u|[Jj]usqu|[Ll]orsqu|[Pp]uisqu|[Qq]uelqu)$"
)

# Locutions figées : élision + apostrophe + composé traités comme un seul mot
_LOCUTIONS: frozenset[str] = frozenset({
    "c'est-à-dire",
})


def _is_letter(ch: str) -> bool:
    return bool(_LETTER_RE.match(ch))


def _is_digit(ch: str) -> bool:
    return ch.isdigit()


def _is_number_char(ch: str) -> bool:
    return ch.isdigit() or ch in (".", "'")


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


def tokenise(text: str) -> list[Token]:
    """Découpe le texte normalisé en tokens.

    Algorithme linéaire :
    1. Lettres consécutives → Mot (ou Sigle si 2+ majuscules)
    2. Chiffres consécutifs (+ . ') → Nombre
    3. ' ou - entre deux lettres → Separateur
    4. Espace → Separateur(sep_type="space")
    5. Tout le reste → Ponctuation

    Args:
        text: Texte normalisé à tokeniser.

    Returns:
        Liste de Token avec types et spans.
    """
    if not text:
        return []

    tokens: list[Token] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # ── Lettres consécutives → Mot ou Sigle ──
        if _is_letter(ch):
            start = i
            while i < n and _is_letter(text[i]):
                i += 1
            word = text[start:i]

            if _SIGLE_RE.match(word):
                children: list[Token] = []
                for j, letter in enumerate(word):
                    child = Mot(
                        type=TokenType.MOT,
                        text=letter,
                        span=(start + j, start + j + 1),
                        ortho=letter.lower(),
                    )
                    children.append(child)
                tok = Sigle(
                    type=TokenType.SIGLE,
                    text=word,
                    span=(start, i),
                    children=children,
                )
            else:
                tok = Mot(
                    type=TokenType.MOT,
                    text=word,
                    span=(start, i),
                    ortho=word.lower(),
                )
            tokens.append(tok)
            continue

        # ── Chiffres consécutifs → Nombre ──
        if _is_digit(ch):
            start = i
            while i < n and _is_number_char(text[i]):
                i += 1
            while i > start and text[i - 1] in (".", "'"):
                i -= 1
            tok = Nombre(
                type=TokenType.NOMBRE,
                text=text[start:i],
                span=(start, i),
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

        # ── Tout le reste → Ponctuation ──
        tok = Ponctuation(
            type=TokenType.PONCTUATION,
            text=ch,
            span=(i, i + 1),
        )
        tokens.append(tok)
        i += 1

    tokens = _merge_compounds(tokens)
    tokens = _merge_locutions(tokens)
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Résultat structuré
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ResultatTokenisation:
    """Résultat complet de la normalisation + tokenisation."""

    texte_original: str
    texte_normalise: str
    tokens: list[Token]

    @property
    def mots(self) -> list[Mot]:
        """Retourne uniquement les tokens de type Mot."""
        return [t for t in self.tokens if isinstance(t, Mot)]

    @property
    def nb_mots(self) -> int:
        return len(self.mots)

    @property
    def nb_tokens(self) -> int:
        return len(self.tokens)

    def words(self) -> list[str]:
        """Retourne la liste des formes orthographiques des mots."""
        return [t.ortho for t in self.mots]

    def format_table(self) -> str:
        """Retourne un affichage tabulaire des tokens."""
        lines = [f"{'Texte':15s}  {'Type':12s}  {'Span':10s}  Détail"]
        lines.append("-" * 55)
        for t in self.tokens:
            text_repr = repr(t.text)
            detail = ""
            if isinstance(t, Mot):
                detail = f"ortho={t.ortho!r}"
                if t.children:
                    detail += f" composé=[{'+'.join(c.text for c in t.children)}]"
            elif isinstance(t, Separateur):
                detail = f"sep={t.sep_type}"
            elif isinstance(t, Sigle):
                letters = "".join(c.text for c in t.children)
                detail = f"lettres={letters}"
            elif isinstance(t, Nombre):
                detail = f"val={t.text}"
            lines.append(
                f"{text_repr:15s}  {t.type.value:12s}  "
                f"{str(t.span):10s}  {detail}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════════════


class LecturaTokeniseur:
    """Normalisateur et tokeniseur pour le français.

    Combine normalisation typographique (espaces, apostrophes, guillemets,
    nombres, ellipses, tirets) et tokenisation en types structurés
    (Mot, Ponctuation, Separateur, Nombre, Sigle) avec spans.
    """

    def normalize(self, text: str) -> str:
        """Normalise un texte brut (étape 1 seule).

        Transformations : espaces, apostrophes, ellipses,
        ponctuation, nombres, guillemets, parenthèses, tirets.
        """
        return normalise(text)

    def tokenize(self, text: str, normalize: bool = True) -> list[Token]:
        """Tokenise un texte en tokens typés avec spans.

        Args:
            text: Texte à tokeniser.
            normalize: Si True (défaut), normalise le texte avant tokenisation.

        Returns:
            Liste de Token (Mot, Ponctuation, Separateur, Nombre, Sigle).
        """
        if normalize:
            text = normalise(text)
        return tokenise(text)

    def analyze(self, text: str) -> ResultatTokenisation:
        """Normalisation + tokenisation avec résultat structuré.

        Args:
            text: Texte brut à analyser.

        Returns:
            ResultatTokenisation avec texte original, normalisé, et tokens.
        """
        text_norm = normalise(text)
        tokens = tokenise(text_norm)
        return ResultatTokenisation(
            texte_original=text,
            texte_normalise=text_norm,
            tokens=tokens,
        )

    def extract_words(self, text: str) -> list[str]:
        """Raccourci : retourne la liste des mots (formes normalisées).

        Args:
            text: Texte brut.

        Returns:
            Liste de chaînes (formes orthographiques en minuscules).
        """
        return self.analyze(text).words()


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    tok = LecturaTokeniseur()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = tok.analyze(text)
        print(f"Original   : {result.texte_original}")
        print(f"Normalisé  : {result.texte_normalise}")
        print(f"Mots       : {result.nb_mots}")
        print(f"Tokens     : {result.nb_tokens}")
        print()
        print(result.format_table())
    else:
        print("Lectura Tokeniseur — Mode interactif (Ctrl+C pour quitter)")
        print()
        try:
            while True:
                text = input("Texte > ").strip()
                if not text:
                    continue
                result = tok.analyze(text)
                print(f"  Normalisé : {result.texte_normalise}")
                print(f"  Mots ({result.nb_mots}) : {result.words()}")
                print()
                print(result.format_table())
                print()
        except (KeyboardInterrupt, EOFError):
            print()
