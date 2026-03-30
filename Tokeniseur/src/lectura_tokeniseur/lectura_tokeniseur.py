"""Lectura Tokeniseur Complet — Normalisateur et tokeniseur pour le français.

Fichier unique, autonome, zéro dépendance externe.
Détecte les formules (nombres, sigles, dates, téléphones, numéros,
ordinaux, fractions, notations scientifiques, expressions mathématiques).

Usage rapide :
    from lectura_tokeniseur import LecturaTokeniseur

    tok = LecturaTokeniseur()
    tokens = tok.tokenize("L'enfant a 42 ans.")
    for t in tokens:
        print(f"{t.text:12s}  {t.type.value:12s}  span={t.span}")

    # Extraire les formules
    formules = tok.extract_formules("Appeler le 06 12 34 56 78 le 15/03/2024.")
    for f in formules:
        print(f"{f.text}  →  {f.formule_type.value}")

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

__version__ = "2.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# Modèles de données
# ══════════════════════════════════════════════════════════════════════════════

Span = tuple[int, int]


class TokenType(Enum):
    """Types de tokens reconnus."""
    MOT = "mot"
    PONCTUATION = "ponctuation"
    SEPARATEUR = "separateur"
    FORMULE = "formule"


class FormuleType(Enum):
    """Sous-types de formules détectées."""
    NOMBRE = "nombre"
    SIGLE = "sigle"
    DATE = "date"
    TELEPHONE = "telephone"
    NUMERO = "numero"
    ORDINAL = "ordinal"
    FRACTION = "fraction"
    SCIENTIFIQUE = "scientifique"
    MATHS = "maths"


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
class Formule(Token):
    """Token de type FORMULE (nombre, sigle, date, téléphone, etc.).

    Attributs :
        formule_type : sous-type de la formule
        children : sous-tokens pour les formules complexes
        valeur : valeur normalisée (ex: "42", "SNCF", "06 12 34 56 78")
    """
    formule_type: FormuleType = FormuleType.NOMBRE
    children: list[Token] = field(default_factory=list)
    valeur: str = ""
    display_fr: str = ""


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
        # Opérateur dans formule : chiffre/lettre + op + chiffre/lettre (ex: 3+5, x-2)
        if (prev.isdigit() or prev.isalpha()) and (next_.isdigit() or next_.isalpha()):
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
# Tokenisation — Passe 1 (scan)
# ══════════════════════════════════════════════════════════════════════════════

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
                i += 1
            # Reculer si on finit par . ou ' ou , (ponctuation)
            while i > start and text[i - 1] in (".", "'", ","):
                i -= 1
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

        # ── Tout le reste → Ponctuation ──
        tok = Ponctuation(
            type=TokenType.PONCTUATION,
            text=ch,
            span=(i, i + 1),
        )
        tokens.append(tok)
        i += 1

    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Détecteurs de formules (regex embarquées, zéro dep)
# ══════════════════════════════════════════════════════════════════════════════

# ── Nombres romains ──
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_ROMAN_VALID_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)


def _is_roman(s: str) -> bool:
    """Vérifie si la chaîne est un nombre romain valide."""
    if not s or not _ROMAN_RE.match(s):
        return False
    return bool(_ROMAN_VALID_RE.match(s.upper())) and len(s) > 0 and s.upper() != ""


# ── TELEPHONE ──
_TEL_CLEAN_RE = re.compile(r"[\s.\-]")
_TEL_RE = re.compile(r"^0[1-9]\d{8}$")


def _detect_telephone(text: str) -> bool:
    """Détecte un numéro de téléphone français (0X XX XX XX XX)."""
    cleaned = _TEL_CLEAN_RE.sub("", text)
    return bool(_TEL_RE.match(cleaned)) and len(cleaned) == 10


# ── DATE ──
_DATE_PATTERNS = [
    # DD/MM/YYYY ou DD-MM-YYYY ou DD.MM.YYYY
    re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$"),
    # YYYY-MM-DD ou YYYY/MM/DD
    re.compile(r"^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})$"),
]


def _detect_date(text: str) -> bool:
    """Détecte une date (DD/MM/YYYY, YYYY-MM-DD, etc.)."""
    for i, pat in enumerate(_DATE_PATTERNS):
        m = pat.match(text)
        if m:
            if i == 0:
                jour, mois = int(m.group(1)), int(m.group(2))
            else:
                mois, jour = int(m.group(2)), int(m.group(3))
            if 1 <= jour <= 31 and 1 <= mois <= 12:
                return True
    return False


# ── SCIENTIFIQUE ──
_SCIENTIFIQUE_RE = re.compile(r"^\d+\.?\d*[eE][+\-]?\d+$")


def _detect_scientifique(text: str) -> bool:
    """Détecte une notation scientifique (3.14e-5)."""
    return bool(_SCIENTIFIQUE_RE.match(text))


# ── FRACTION ──
_FRACTION_RE = re.compile(r"^(\d+)\s*/\s*(\d+)$")


def _detect_fraction(text: str) -> bool:
    """Détecte une fraction (3/4, 1/2)."""
    m = _FRACTION_RE.match(text)
    if m:
        denom = int(m.group(2))
        return denom != 0
    return False


# ── ORDINAL ──
_ORDINAL_SUFFIXES = (
    "ième", "ième", "ème", "ère", "er", "re", "nd", "nde",
    "ièmes", "èmes", "ères", "ers", "nds", "ndes",
)
# Version avec juste "e" en fin (1e, 2e, etc.) — prudent, seulement après chiffre
_ORDINAL_SUFFIX_E = "e"


def _detect_ordinal(text: str) -> bool:
    """Détecte un ordinal (42e, 1er, 21ème, IIe, etc.)."""
    text_lower = text.lower()
    for suffix in _ORDINAL_SUFFIXES:
        if text_lower.endswith(suffix):
            prefix = text[:len(text) - len(suffix)]
            if prefix.isdigit() and len(prefix) > 0:
                return True
            if _is_roman(prefix):
                return True
    # Cas spécial : "e" seul (1e, 2e, etc.)
    if text_lower.endswith("e") and len(text) > 1:
        prefix = text[:-1]
        if prefix.isdigit():
            return True
        if _is_roman(prefix):
            return True
    return False


# ── MATHS ──
_MATHS_OPERATORS = set("+-=×*÷^√<>≤≥≠≈∞∑∏∫∂∇±")
_MATHS_SUPERSCRIPTS = set("⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ")
_MATHS_FUNCTIONS = {
    "sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs",
    "lim", "max", "min", "sup", "inf",
}


def _detect_maths(text: str) -> bool:
    """Détecte une expression mathématique (2x+3=5, sin(x), x², etc.)."""
    has_digit = any(c.isdigit() for c in text)
    has_letter = any(c.isalpha() for c in text)
    has_operator = any(c in _MATHS_OPERATORS for c in text)
    has_superscript = any(c in _MATHS_SUPERSCRIPTS for c in text)
    has_greek = any(c in _GREEK_LETTERS for c in text)
    has_parens = "(" in text and ")" in text

    # Fonctions mathématiques
    text_lower = text.lower()
    has_func = any(f in text_lower for f in _MATHS_FUNCTIONS)

    # Lettre+chiffre adjacents avec opérateur (2x+3)
    if has_digit and has_letter and has_operator:
        return True
    # Exposant unicode (x²)
    if has_superscript and (has_letter or has_digit):
        return True
    # Lettre grecque avec chiffre ou opérateur
    if has_greek and (has_digit or has_operator):
        return True
    # Fonction mathématique avec parenthèses
    if has_func and has_parens:
        return True
    # Chiffre directement adjacent à lettre avec opérateur
    if has_operator and (has_digit or has_letter):
        # Vérifier qu'il y a au moins 2 éléments distincts (pas juste un signe)
        non_op = sum(1 for c in text if c not in _MATHS_OPERATORS and not c.isspace())
        if non_op >= 2:
            return True

    return False


# ── NUMERO ──
_NUMERO_SPLIT_RE = re.compile(r"[\s.]+")


def _detect_numero(text: str) -> bool:
    """Détecte un numéro composé (654 001 45, AB.123.CD).

    2+ groupes alphanumériques séparés par espaces ou points.
    Pas un nombre décimal (X.Y purement numérique avec 1 seul point).
    """
    parts = _NUMERO_SPLIT_RE.split(text.strip())
    if len(parts) < 2:
        return False
    # Vérifier que chaque partie est alphanumérique
    for p in parts:
        if not p or not re.match(r"^[A-Za-z0-9]+$", p):
            return False
    # Pas un nombre décimal simple (ex: 3.14 = 2 parties numériques avec 1 point)
    if len(parts) == 2 and all(p.isdigit() for p in parts) and "." in text and " " not in text:
        return False
    return True


# ── SIGLE ──
_SIGLE_RE = re.compile(r"^[A-Za-z0-9]+$")


def _detect_sigle(text: str) -> bool:
    """Détecte un sigle (SNCF, FR25, K2R).

    Conditions : ≥2 chars, ≥1 lettre, pas un nombre romain pur,
    au moins 2 majuscules consécutives ou mélange majuscules+chiffres.
    """
    if len(text) < 2:
        return False
    if not _SIGLE_RE.match(text):
        return False
    has_letter = any(c.isalpha() for c in text)
    if not has_letter:
        return False
    # Pas un nombre romain pur
    if _is_roman(text) and _ROMAN_VALID_RE.match(text.upper()):
        return True  # Romains valides ne sont PAS des sigles — retourner False
        # Correction : les romains valides sont traités comme des mots, pas des sigles
    if all(c in "IVXLCDM" for c in text.upper()) and _ROMAN_VALID_RE.match(text.upper()):
        return False
    # Doit avoir au moins 2 majuscules consécutives OU mélange lettre+chiffre
    has_digit = any(c.isdigit() for c in text)
    upper_count = sum(1 for c in text if c.isupper())
    if upper_count >= 2:
        return True
    if has_digit and has_letter:
        return True
    return False


# ── NOMBRE ──
_NOMBRE_RE = re.compile(r"^\d+([.']\d+)*$")


def _detect_nombre(text: str) -> bool:
    """Détecte un nombre (42, 3.14, 8'152'368)."""
    return bool(_NOMBRE_RE.match(text))


# ══════════════════════════════════════════════════════════════════════════════
# Passe 2 : Classification des formules
# ══════════════════════════════════════════════════════════════════════════════

def _classify_formule_single(text: str) -> FormuleType | None:
    """Classifie un texte en sous-type de formule (ordre de priorité).

    Retourne None si ce n'est pas une formule reconnue.
    """
    if not text or not text.strip():
        return None

    # 1. TELEPHONE
    if _detect_telephone(text):
        return FormuleType.TELEPHONE

    # 2. DATE
    if _detect_date(text):
        return FormuleType.DATE

    # 3. SCIENTIFIQUE
    if _detect_scientifique(text):
        return FormuleType.SCIENTIFIQUE

    # 4. FRACTION
    if _detect_fraction(text):
        return FormuleType.FRACTION

    # 5. ORDINAL
    if _detect_ordinal(text):
        return FormuleType.ORDINAL

    # 6. MATHS
    if _detect_maths(text):
        return FormuleType.MATHS

    # 7. SIGLE
    if _detect_sigle(text):
        return FormuleType.SIGLE

    # 8. NOMBRE (fallback)
    if _detect_nombre(text):
        return FormuleType.NOMBRE

    return None


def _try_merge_formule_group(tokens: list[Token], start: int) -> tuple[str, int, FormuleType | None] | None:
    """Essaie de fusionner des tokens consécutifs en une formule multi-tokens.

    Gère : téléphones (06 12 34 56 78), dates (15/03/2024), numéros (654 001 45),
    ordinaux (42 e), fractions (3 / 4), scientifiques (3.14e-5),
    maths (2x+3=5).

    Retourne (texte_fusionné, index_fin, type_forcé) ou None.
    """
    n = len(tokens)
    if start >= n:
        return None

    first = tokens[start]

    # ── Téléphone : FORMULE(NOMBRE) + espaces + FORMULE(NOMBRE) × 4 ──
    if isinstance(first, Formule) and first.text.startswith("0") and len(first.text) == 2:
        parts = [first.text]
        j = start + 1
        while j < n and len(parts) < 5:
            if isinstance(tokens[j], Separateur) and tokens[j].sep_type == "space":
                if j + 1 < n and isinstance(tokens[j + 1], Formule) and len(tokens[j + 1].text) == 2 and tokens[j + 1].text.isdigit():
                    parts.append(tokens[j + 1].text)
                    j += 2
                    continue
            break
        if len(parts) == 5:
            full = " ".join(parts)
            if _detect_telephone(full):
                return full, j, FormuleType.TELEPHONE

    # ── Scientifique : NOMBRE + MOT("e"/"E") + [PONCT("-"/"+")]  + NOMBRE ──
    if isinstance(first, Formule) and re.match(r"^\d+\.?\d*$", first.text):
        j = start + 1
        if j < n and isinstance(tokens[j], Mot) and tokens[j].text.lower() == "e":
            k = j + 1
            sign = ""
            if k < n and isinstance(tokens[k], Ponctuation) and tokens[k].text in ("-", "+"):
                sign = tokens[k].text
                k += 1
            if k < n and isinstance(tokens[k], Formule) and tokens[k].text.isdigit():
                full = first.text + "e" + sign + tokens[k].text
                if _detect_scientifique(full):
                    return full, k + 1, FormuleType.SCIENTIFIQUE

    # ── Date : NOMBRE / NOMBRE / NOMBRE ou NOMBRE - NOMBRE - NOMBRE ──
    if isinstance(first, Formule) and first.text.isdigit():
        j = start + 1
        if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text in ("/", "-", "."):
            sep_char = tokens[j].text
            j += 1
            if j < n and isinstance(tokens[j], Formule) and tokens[j].text.isdigit():
                j += 1
                if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text == sep_char:
                    j += 1
                    if j < n and isinstance(tokens[j], Formule) and tokens[j].text.isdigit():
                        j += 1
                        full = "".join(t.text for t in tokens[start:j])
                        if _detect_date(full):
                            return full, j, FormuleType.DATE

    # ── Fraction : NOMBRE / NOMBRE ──
    if isinstance(first, Formule) and first.text.isdigit():
        j = start + 1
        if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text == "/":
            j += 1
            if j < n and isinstance(tokens[j], Formule) and tokens[j].text.isdigit():
                j += 1
                full = "".join(t.text for t in tokens[start:j])
                if _detect_fraction(full):
                    return full, j, FormuleType.FRACTION

    # ── Ordinal : NOMBRE + MOT(suffixe) ──
    if isinstance(first, Formule) and first.text.isdigit():
        j = start + 1
        if j < n and isinstance(tokens[j], Mot):
            full = first.text + tokens[j].text
            if _detect_ordinal(full):
                return full, j + 1, FormuleType.ORDINAL

    # ── Sigle/Maths : séquence de tokens adjacents (lettres+chiffres+opérateurs) ──
    # Fusionne tant qu'on a des tokens sans espace entre eux
    # Seuls les opérateurs mathématiques et parenthèses sont inclus comme ponctuation
    _MERGE_PUNCT = _MATHS_OPERATORS | set("()[]{}^/")
    if isinstance(first, (Formule, Mot)):
        j = start
        parts: list[str] = []
        has_digit = False
        has_op = False
        has_letter = False
        prev_end = first.span[0]
        while j < n:
            t = tokens[j]
            # Stop si espace ou séparateur
            if isinstance(t, Separateur):
                break
            # Vérifier contiguïté (pas d'espace entre les tokens)
            if t.span[0] != prev_end and j > start:
                break
            if isinstance(t, (Formule, Mot)):
                txt = t.text
                if any(c.isdigit() for c in txt):
                    has_digit = True
                if any(c.isalpha() for c in txt):
                    has_letter = True
                parts.append(txt)
                prev_end = t.span[1]
                j += 1
            elif isinstance(t, Ponctuation) and t.text in _MERGE_PUNCT:
                has_op = True
                parts.append(t.text)
                prev_end = t.span[1]
                j += 1
            else:
                break

        if j - start >= 2 and has_digit and has_letter:
            full = "".join(parts)
            # Sigle en priorité (ex: FR25, K2R) — pas d'opérateur
            if not has_op and _detect_sigle(full):
                return full, j, FormuleType.SIGLE
            # Maths (ex: 2x+3=0)
            if _detect_maths(full):
                return full, j, FormuleType.MATHS

    # ── Numéro : groupes numériques séparés par espaces ──
    if isinstance(first, Formule) and first.text.isdigit():
        parts_text = [first.text]
        j = start + 1
        while j < n:
            if isinstance(tokens[j], Separateur) and tokens[j].sep_type == "space":
                if j + 1 < n and isinstance(tokens[j + 1], Formule) and tokens[j + 1].text.isdigit():
                    parts_text.append(tokens[j + 1].text)
                    j += 2
                    continue
            break
        if len(parts_text) >= 2:
            full = " ".join(parts_text)
            if _detect_numero(full):
                return full, j, FormuleType.NUMERO

    return None


def _classify_and_merge(tokens: list[Token]) -> list[Token]:
    """Passe 2 : Classifie les tokens FORMULE bruts et fusionne les multi-tokens.

    Parcourt les tokens, essaie d'abord les fusions multi-tokens (téléphone, date, etc.),
    puis classifie les FORMULE isolées.
    """
    result: list[Token] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        # Essayer de fusionner en formule multi-tokens
        if isinstance(tok, (Formule, Mot)):
            merged = _try_merge_formule_group(tokens, i)
            if merged is not None:
                full_text, end_idx, forced_type = merged
                ftype = forced_type or _classify_formule_single(full_text)
                if ftype is not None:
                    span_start = tokens[i].span[0]
                    span_end = tokens[end_idx - 1].span[1]
                    formule = Formule(
                        type=TokenType.FORMULE,
                        text=full_text,
                        span=(span_start, span_end),
                        formule_type=ftype,
                        children=_build_formule_children(full_text, ftype, span_start),
                        valeur=_extract_valeur(full_text, ftype),
                    )
                    result.append(formule)
                    i = end_idx
                    continue

        # Classifie les FORMULE isolées
        if isinstance(tok, Formule):
            ftype = _classify_formule_single(tok.text)
            if ftype is not None and ftype != FormuleType.NOMBRE:
                tok = Formule(
                    type=TokenType.FORMULE,
                    text=tok.text,
                    span=tok.span,
                    formule_type=ftype,
                    children=_build_formule_children(tok.text, ftype, tok.span[0]),
                    valeur=_extract_valeur(tok.text, ftype),
                )
            else:
                # Nombre par défaut
                tok = Formule(
                    type=TokenType.FORMULE,
                    text=tok.text,
                    span=tok.span,
                    formule_type=FormuleType.NOMBRE,
                    children=[],
                    valeur=tok.text,
                )
            result.append(tok)
            i += 1
            continue

        # Classifie les MOT qui sont en fait des sigles (2+ majuscules)
        if isinstance(tok, Mot) and _detect_sigle(tok.text):
            formule = Formule(
                type=TokenType.FORMULE,
                text=tok.text,
                span=tok.span,
                formule_type=FormuleType.SIGLE,
                children=_build_formule_children(tok.text, FormuleType.SIGLE, tok.span[0]),
                valeur=tok.text.upper(),
            )
            result.append(formule)
            i += 1
            continue

        result.append(tok)
        i += 1

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Sous-tokenisation (children) et extraction de valeur
# ══════════════════════════════════════════════════════════════════════════════

def _build_formule_children(text: str, ftype: FormuleType, offset: int) -> list[Token]:
    """Construit les sous-tokens d'une formule selon son type."""

    if ftype == FormuleType.NOMBRE:
        # Atomique, pas d'enfants
        return []

    if ftype == FormuleType.SIGLE:
        # 1 enfant Mot par lettre, 1 enfant Formule(NOMBRE) par groupe de chiffres
        children: list[Token] = []
        i = 0
        while i < len(text):
            if text[i].isalpha():
                children.append(Mot(
                    type=TokenType.MOT,
                    text=text[i],
                    span=(offset + i, offset + i + 1),
                    ortho=text[i].lower(),
                ))
                i += 1
            elif text[i].isdigit():
                start = i
                while i < len(text) and text[i].isdigit():
                    i += 1
                children.append(Formule(
                    type=TokenType.FORMULE,
                    text=text[start:i],
                    span=(offset + start, offset + i),
                    formule_type=FormuleType.NOMBRE,
                    valeur=text[start:i],
                ))
            else:
                i += 1
        return children

    if ftype == FormuleType.TELEPHONE:
        # Enfants = paires de chiffres
        cleaned = _TEL_CLEAN_RE.sub("", text)
        children = []
        pos = 0
        for ci in range(0, len(cleaned), 2):
            pair = cleaned[ci:ci + 2]
            # Trouver la position dans le texte original
            while pos < len(text) and not text[pos].isdigit():
                pos += 1
            start = pos
            count = 0
            while pos < len(text) and count < 2:
                if text[pos].isdigit():
                    count += 1
                pos += 1
            children.append(Formule(
                type=TokenType.FORMULE,
                text=pair,
                span=(offset + start, offset + pos),
                formule_type=FormuleType.NOMBRE,
                valeur=pair,
            ))
        return children

    if ftype == FormuleType.DATE:
        # Enfants = jour, séparateur, mois, séparateur, année
        children = []
        parts: list[str] = []
        seps: list[str] = []
        current = ""
        for ch in text:
            if ch.isdigit():
                current += ch
            else:
                if current:
                    parts.append(current)
                    current = ""
                seps.append(ch)
        if current:
            parts.append(current)

        pos = 0
        for pi, part in enumerate(parts):
            idx = text.index(part, pos)
            children.append(Formule(
                type=TokenType.FORMULE,
                text=part,
                span=(offset + idx, offset + idx + len(part)),
                formule_type=FormuleType.NOMBRE,
                valeur=part,
            ))
            pos = idx + len(part)
            if pi < len(seps):
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION,
                    text=seps[pi],
                    span=(offset + pos, offset + pos + 1),
                ))
                pos += 1
        return children

    if ftype == FormuleType.FRACTION:
        # Enfants = numérateur, barre, dénominateur
        m = _FRACTION_RE.match(text)
        if m:
            num, den = m.group(1), m.group(2)
            slash_pos = text.index("/")
            return [
                Formule(
                    type=TokenType.FORMULE, text=num,
                    span=(offset, offset + len(num)),
                    formule_type=FormuleType.NOMBRE, valeur=num,
                ),
                Ponctuation(
                    type=TokenType.PONCTUATION, text="/",
                    span=(offset + slash_pos, offset + slash_pos + 1),
                ),
                Formule(
                    type=TokenType.FORMULE, text=den,
                    span=(offset + slash_pos + 1, offset + slash_pos + 1 + len(den)),
                    formule_type=FormuleType.NOMBRE, valeur=den,
                ),
            ]
        return []

    if ftype == FormuleType.SCIENTIFIQUE:
        # Enfants = mantisse, e, exposant
        m = re.match(r"^(\d+\.?\d*)([eE])([+\-]?\d+)$", text)
        if m:
            mantisse, e_char, exposant = m.group(1), m.group(2), m.group(3)
            p1 = len(mantisse)
            p2 = p1 + 1
            return [
                Formule(
                    type=TokenType.FORMULE, text=mantisse,
                    span=(offset, offset + p1),
                    formule_type=FormuleType.NOMBRE, valeur=mantisse,
                ),
                Mot(
                    type=TokenType.MOT, text=e_char,
                    span=(offset + p1, offset + p2),
                    ortho=e_char,
                ),
                Formule(
                    type=TokenType.FORMULE, text=exposant,
                    span=(offset + p2, offset + p2 + len(exposant)),
                    formule_type=FormuleType.NOMBRE, valeur=exposant,
                ),
            ]
        return []

    if ftype == FormuleType.ORDINAL:
        # Enfants = nombre, suffixe
        # Trouver où finit le nombre et où commence le suffixe
        i = 0
        while i < len(text) and (text[i].isdigit() or text[i] in "IVXLCDM"):
            i += 1
        if i == 0:
            # Essayer les romains en minuscule — pas fiable, garder vide
            return []
        nombre_part = text[:i]
        suffixe_part = text[i:]
        children = [
            Formule(
                type=TokenType.FORMULE, text=nombre_part,
                span=(offset, offset + i),
                formule_type=FormuleType.NOMBRE, valeur=nombre_part,
            ),
        ]
        if suffixe_part:
            children.append(Mot(
                type=TokenType.MOT, text=suffixe_part,
                span=(offset + i, offset + i + len(suffixe_part)),
                ortho=suffixe_part.lower(),
            ))
        return children

    if ftype == FormuleType.MATHS:
        # Enfants typés : nombre, opérateur, variable, fonction, parenthèse, grec
        children = []
        i = 0
        while i < len(text):
            ch = text[i]

            # Nombre
            if ch.isdigit():
                start = i
                while i < len(text) and (text[i].isdigit() or text[i] == "."):
                    i += 1
                children.append(Formule(
                    type=TokenType.FORMULE, text=text[start:i],
                    span=(offset + start, offset + i),
                    formule_type=FormuleType.NOMBRE, valeur=text[start:i],
                ))
                continue

            # Opérateur
            if ch in _MATHS_OPERATORS:
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=ch,
                    span=(offset + i, offset + i + 1),
                ))
                i += 1
                continue

            # Exposant unicode
            if ch in _MATHS_SUPERSCRIPTS:
                children.append(Formule(
                    type=TokenType.FORMULE, text=ch,
                    span=(offset + i, offset + i + 1),
                    formule_type=FormuleType.NOMBRE, valeur=ch,
                ))
                i += 1
                continue

            # Lettre grecque
            if ch in _GREEK_LETTERS:
                children.append(Mot(
                    type=TokenType.MOT, text=ch,
                    span=(offset + i, offset + i + 1),
                    ortho=ch,
                ))
                i += 1
                continue

            # Parenthèses
            if ch in "()[]{}":
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=ch,
                    span=(offset + i, offset + i + 1),
                ))
                i += 1
                continue

            # Variable ou fonction (lettre)
            if ch.isalpha():
                start = i
                while i < len(text) and text[i].isalpha():
                    i += 1
                word = text[start:i]
                children.append(Mot(
                    type=TokenType.MOT, text=word,
                    span=(offset + start, offset + i),
                    ortho=word.lower(),
                ))
                continue

            # Autre caractère
            i += 1

        return children

    if ftype == FormuleType.NUMERO:
        # Enfants = groupes alphanumériques
        children = []
        parts = _NUMERO_SPLIT_RE.split(text.strip())
        pos = 0
        for part in parts:
            if not part:
                continue
            idx = text.index(part, pos)
            if part.isdigit():
                children.append(Formule(
                    type=TokenType.FORMULE, text=part,
                    span=(offset + idx, offset + idx + len(part)),
                    formule_type=FormuleType.NOMBRE, valeur=part,
                ))
            else:
                children.append(Mot(
                    type=TokenType.MOT, text=part,
                    span=(offset + idx, offset + idx + len(part)),
                    ortho=part.lower(),
                ))
            pos = idx + len(part)
        return children

    return []


def _extract_valeur(text: str, ftype: FormuleType) -> str:
    """Extrait la valeur normalisée d'une formule."""
    if ftype == FormuleType.TELEPHONE:
        return _TEL_CLEAN_RE.sub("", text)
    if ftype == FormuleType.DATE:
        return text
    if ftype == FormuleType.SIGLE:
        return "".join(c for c in text if c.isalnum()).upper()
    if ftype == FormuleType.ORDINAL:
        return text
    if ftype == FormuleType.FRACTION:
        m = _FRACTION_RE.match(text)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        return text
    if ftype == FormuleType.SCIENTIFIQUE:
        return text
    if ftype == FormuleType.MATHS:
        return text
    if ftype == FormuleType.NUMERO:
        return text
    if ftype == FormuleType.NOMBRE:
        return re.sub(r"['\s]", "", text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline complet de tokenisation
# ══════════════════════════════════════════════════════════════════════════════

def tokenise(text: str) -> list[Token]:
    """Découpe le texte normalisé en tokens avec détection de formules.

    Pipeline en 2 passes :
    1. Scan : MOT, PONCTUATION, SEPARATEUR, FORMULE brut
    2. Classification + fusion : détecte les sous-types de formules

    Args:
        text: Texte normalisé à tokeniser.

    Returns:
        Liste de Token (Mot, Ponctuation, Separateur, Formule).
    """
    if not text:
        return []

    # Passe 1 : scan brut
    tokens = _scan_tokens(text)

    # Fusions de mots composés et locutions (sur les Mot)
    tokens = _merge_compounds(tokens)
    tokens = _merge_locutions(tokens)

    # Passe 2 : classification et fusion des formules
    tokens = _classify_and_merge(tokens)

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
    def formules(self) -> list[Formule]:
        """Retourne uniquement les tokens de type Formule."""
        return [t for t in self.tokens if isinstance(t, Formule)]

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
        lines.append("-" * 65)
        for t in self.tokens:
            text_repr = repr(t.text)
            detail = ""
            if isinstance(t, Mot):
                detail = f"ortho={t.ortho!r}"
                if t.children:
                    detail += f" composé=[{'+'.join(c.text for c in t.children)}]"
            elif isinstance(t, Separateur):
                detail = f"sep={t.sep_type}"
            elif isinstance(t, Formule):
                detail = f"sous-type={t.formule_type.value}"
                if t.valeur:
                    detail += f" val={t.valeur!r}"
                if t.children:
                    detail += f" ({len(t.children)} enfants)"
            lines.append(
                f"{text_repr:15s}  {t.type.value:12s}  "
                f"{str(t.span):10s}  {detail}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════════════


class LecturaTokeniseur:
    """Normalisateur et tokeniseur complet pour le français.

    Combine normalisation typographique (espaces, apostrophes, guillemets,
    nombres, ellipses, tirets) et tokenisation en types structurés
    (Mot, Ponctuation, Separateur, Formule) avec spans et détection
    de formules (nombres, sigles, dates, téléphones, etc.).
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
            Liste de Token (Mot, Ponctuation, Separateur, Formule).
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

    def extract_formules(self, text: str) -> list[Formule]:
        """Raccourci : retourne la liste des formules détectées.

        Args:
            text: Texte brut.

        Returns:
            Liste de Formule avec sous-type, valeur, enfants.
        """
        return self.analyze(text).formules


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
        print(f"Formules   : {len(result.formules)}")
        print(f"Tokens     : {result.nb_tokens}")
        print()
        print(result.format_table())
    else:
        print("Lectura Tokeniseur Complet — Mode interactif (Ctrl+C pour quitter)")
        print()
        try:
            while True:
                text = input("Texte > ").strip()
                if not text:
                    continue
                result = tok.analyze(text)
                print(f"  Normalisé : {result.texte_normalise}")
                print(f"  Mots ({result.nb_mots}) : {result.words()}")
                if result.formules:
                    print(f"  Formules ({len(result.formules)}) :")
                    for f in result.formules:
                        print(f"    {f.text} → {f.formule_type.value} (val={f.valeur!r})")
                print()
                print(result.format_table())
                print()
        except (KeyboardInterrupt, EOFError):
            print()
