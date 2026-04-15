"""Détecteurs de formules (regex embarquées, zéro dépendance externe)."""

from __future__ import annotations

import re

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
    "ième", "ème", "eme", "ère", "er", "re", "nd", "nde",
    "ièmes", "èmes", "emes", "ères", "ers", "nds", "ndes",
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
            if _is_roman(prefix) and prefix == prefix.upper():
                return True
    # Cas spécial : "e" seul (1e, 2e, IIe, XXe, etc.)
    if text_lower.endswith("e") and len(text) > 1:
        prefix = text[:-1]
        if prefix.isdigit():
            return True
        # Pour les romains + "e", exiger préfixe majuscule et au moins 2 chars
        # pour éviter les faux positifs (Le, Ce, De, Me, livre, lire...)
        if len(prefix) >= 2 and _is_roman(prefix) and prefix == prefix.upper():
            return True
    return False


# ── MATHS ──
_MATHS_OPERATORS = set("+-−=×*÷/^√<>≤≥≠≈≃≡∞∑∏∫∂∇±∈∉⊂∪∩→←↔⇒⇔°%‰")
# Opérateurs qui sont *toujours* mathématiques (pas ambigus comme +, -, <, >)
_MATHS_SPECIAL = set("∈∉⊂∪∩→←↔⇒⇔≈≃≡∑∏∫∂∇")
_MATHS_SUPERSCRIPTS = set("⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ")
_MATHS_FUNCTIONS = {
    "sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs",
    "lim", "max", "min", "sup", "inf",
}
# Lettres grecques courantes en maths
_GREEK_LETTERS = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")


def _detect_maths(text: str) -> bool:
    """Détecte une expression mathématique (2x+3=5, sin(x), x², √9, etc.)."""
    # √ suivi de quelque chose → toujours maths
    if "√" in text and len(text) > 1:
        return True

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

    # Opérateur spécial (∈, ∪, →…) + lettres — toujours maths, même sans chiffre
    # "x∈A", "A∪B", "P→Q"
    has_special_op = any(c in _MATHS_SPECIAL for c in text)
    if has_special_op and has_letter:
        return True

    # ± devant un nombre → toujours maths
    if "±" in text and has_digit:
        return True

    # Factorielle : nombre ou lettre + !
    if "!" in text and (has_digit or has_letter):
        return True

    # Notation dérivée : f'(x), g''(x), sin'(x), etc.
    if has_parens and ("'" in text or "\u2032" in text):
        if re.search(r"[a-zA-Z]['\u2032]+\(", text):
            return True

    # lettre + parenthèses → maths : f(x), g(x+1)
    if has_parens and has_letter:
        if re.search(r'[a-zA-Z]\(', text):
            return True

    return False


# ── NUMERO ──
_NUMERO_SPLIT_RE = re.compile(r"[\s.\-]+")


def _detect_numero(text: str) -> bool:
    """Détecte un numéro composé (654 001 45, AB.123.CD, CL-067-TS).

    2+ groupes alphanumériques séparés par espaces, points ou tirets.
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
    # Avec tirets : règles strictes pour éviter confusion avec MATHS
    if "-" in text:
        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)
        if not has_digits:
            # x-y, a-b-c → pas un numéro
            return False
        if has_letters:
            # Lettres minuscules → MATHS (a-3-5b), majuscules → NUMERO (A-3-5B)
            letters = [c for c in text if c.isalpha()]
            if any(c.islower() for c in letters):
                return False
        else:
            # Tout-chiffres avec tirets : NUMERO si un groupe a un zéro initial
            # (0800-350-58 → numéro), sinon MATHS (800-350-58 → soustraction)
            if not any(len(p) > 1 and p[0] == "0" for p in parts):
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
        return False  # Romains valides ne sont pas des sigles
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
# 6 nouveaux détecteurs
# ══════════════════════════════════════════════════════════════════════════════

# ── HEURE ──
_HEURE_RE = re.compile(
    r"^\d{1,2}\s*[hH:]\s*\d{0,2}(?:\s*min\s*\d{0,2})?(?:\s*s)?$"
    r"|^\d{1,2}\s*min$"
    r"|^\d{1,2}\s*s$"
)


def _detect_heure(text: str) -> bool:
    """Détecte une heure (14h30, 8:30, 5min, 30s)."""
    return bool(_HEURE_RE.match(text.strip()))


# ── MONNAIE ──
_MONNAIE_RE = re.compile(
    r"^[€$£¥]\s*[-+±]?[0-9][0-9 ']*[.,]?\d*$"
    r"|^[-+±]?[0-9][0-9 ']*[.,]?\d*\s*[€$£¥]$"
    r"|^[-+±]?[0-9][0-9 ']*[0-9]*[€$£¥]\d{1,2}$"
    r"|^[-+±]?[0-9][0-9 ']*[.,]?\d*\s*(?:EUR|USD|GBP|CHF|JPY)$"
    r"|^(?:EUR|USD|GBP|CHF|JPY)\s*[-+±]?[0-9][0-9 ']*[.,]?\d*$",
    re.IGNORECASE,
)


def _detect_monnaie(text: str) -> bool:
    """Détecte une monnaie (42€, $100, 50 CHF)."""
    return bool(_MONNAIE_RE.match(text.strip()))


# ── POURCENTAGE ──
_POURCENTAGE_RE = re.compile(r"^[-+±]?[0-9][0-9 ']*\.?[0-9]*\s*[%‰]$")


def _detect_pourcentage(text: str) -> bool:
    """Détecte un pourcentage (45%, 3.5‰)."""
    return bool(_POURCENTAGE_RE.match(text.strip()))


# ── INTERVALLE ──
_INTERVALLE_RE = re.compile(r"^[\[\]]\s*[^;,\[\]]+[;,][^;,\[\]]+\s*[\[\]]$")


def _detect_intervalle(text: str) -> bool:
    """Détecte un intervalle ([1;5], ]0,1[)."""
    return bool(_INTERVALLE_RE.match(text.strip()))


# ── GPS ──
_GPS_RE = re.compile(
    r"^\d{1,3}°\d{1,2}'\d{0,2}\"?\s*[NSEOW]$"
    r"|^\d{1,3}°\d{1,2}\s*[NSEOW]$"
    r"|^\d{1,3}(?:\.\d+)?°\s*[NSEOW]$",
    re.IGNORECASE,
)


def _detect_gps(text: str) -> bool:
    """Détecte une coordonnée GPS (48°51'N, 2.35°E)."""
    return bool(_GPS_RE.match(text.strip()))


# ── PAGE / CHAPITRE ──
_PAGE_CHAP_RE = re.compile(
    r"^(?:p|P|page|chap|ch|Ch)\.?\s*\d+$",
)


def _detect_page_chapitre(text: str) -> bool:
    """Détecte une référence page/chapitre (p.42, chap3, page 12)."""
    return bool(_PAGE_CHAP_RE.match(text.strip()))
