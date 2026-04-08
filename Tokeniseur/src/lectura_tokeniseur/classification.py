"""Passe 2 : Classification des formules, fusion multi-tokens, sous-tokenisation."""

from __future__ import annotations

import logging
import re

from lectura_tokeniseur.models import (
    Token, TokenType, FormuleType,
    Mot, Ponctuation, Separateur, Formule,
)
from lectura_tokeniseur.detection import (
    _detect_telephone, _detect_date, _detect_scientifique,
    _detect_fraction, _detect_ordinal, _detect_maths, _detect_sigle,
    _detect_numero, _detect_nombre, _is_roman,
    _detect_heure, _detect_monnaie, _detect_pourcentage,
    _detect_intervalle, _detect_gps, _detect_page_chapitre,
    _TEL_CLEAN_RE, _FRACTION_RE, _NUMERO_SPLIT_RE,
    _ROMAN_VALID_RE, _MATHS_OPERATORS, _MATHS_SUPERSCRIPTS, _GREEK_LETTERS,
    _MATHS_FUNCTIONS, _MATHS_SPECIAL,
)
from lectura_tokeniseur.maths import UNIT_NAMES_LOWER as _UNIT_NAMES_LOWER

logger = logging.getLogger(__name__)


def _is_unit_name(text: str) -> bool:
    """Vérifie si un texte est un nom d'unité (casse originale ou minuscule)."""
    return text in _UNIT_NAMES_LOWER or text.lower() in _UNIT_NAMES_LOWER


# ══════════════════════════════════════════════════════════════════════════════
# Classification single-token
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

    # 3. HEURE
    if _detect_heure(text):
        return FormuleType.HEURE

    # 4. SCIENTIFIQUE
    if _detect_scientifique(text):
        return FormuleType.SCIENTIFIQUE

    # 5. FRACTION
    if _detect_fraction(text):
        return FormuleType.FRACTION

    # 6. ORDINAL
    if _detect_ordinal(text):
        return FormuleType.ORDINAL

    # 7. POURCENTAGE
    if _detect_pourcentage(text):
        return FormuleType.POURCENTAGE

    # 8. MONNAIE
    if _detect_monnaie(text):
        return FormuleType.MONNAIE

    # 9. GPS
    if _detect_gps(text):
        return FormuleType.GPS

    # 10. INTERVALLE
    if _detect_intervalle(text):
        return FormuleType.INTERVALLE

    # 11. PAGE_CHAPITRE
    if _detect_page_chapitre(text):
        return FormuleType.PAGE_CHAPITRE

    # 12. MATHS
    if _detect_maths(text):
        return FormuleType.MATHS

    # 13. SIGLE
    if _detect_sigle(text):
        return FormuleType.SIGLE

    # 14. NUMERO
    if _detect_numero(text):
        return FormuleType.NUMERO

    # 15. NOMBRE (fallback)
    if _detect_nombre(text):
        return FormuleType.NOMBRE

    logger.warning("Unrecognized formule pattern: %r", text)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Fusion multi-tokens
# ══════════════════════════════════════════════════════════════════════════════

def _try_merge_formule_group(tokens: list[Token], start: int) -> tuple[str, int, FormuleType | None] | None:
    """Essaie de fusionner des tokens consécutifs en une formule multi-tokens.

    Gère : téléphones, dates, numéros, ordinaux, fractions, scientifiques,
    maths, heures, monnaies, pourcentages, GPS, intervalles, pages/chapitres.

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

    # ── Heure : FORMULE("14") + MOT("h") + [FORMULE("30")] ──
    if isinstance(first, Formule) and first.text.isdigit():
        j = start + 1
        if j < n and isinstance(tokens[j], Mot) and tokens[j].text.lower() in ("h", "min", "s"):
            unit = tokens[j].text.lower()
            k = j + 1
            # h peut être suivi de minutes
            if unit == "h":
                if k < n and isinstance(tokens[k], Formule) and tokens[k].text.isdigit():
                    full = first.text + tokens[j].text + tokens[k].text
                    k += 1
                    # Optionnel : min après
                    if k < n and isinstance(tokens[k], Mot) and tokens[k].text.lower() == "min":
                        full += tokens[k].text
                        k += 1
                        if k < n and isinstance(tokens[k], Formule) and tokens[k].text.isdigit():
                            full += tokens[k].text
                            k += 1
                    return full, k, FormuleType.HEURE
                else:
                    # ex: "14h" sans minutes
                    full = first.text + tokens[j].text
                    return full, k, FormuleType.HEURE
            elif unit == "min":
                full = first.text + tokens[j].text
                return full, k, FormuleType.HEURE
            elif unit == "s":
                full = first.text + tokens[j].text
                return full, k, FormuleType.HEURE
        # Heure avec PONCT(":") → FORMULE + [SPACE] + PONCT(":") + [SPACE] + FORMULE
        j2 = j
        if j2 < n and isinstance(tokens[j2], Separateur) and tokens[j2].sep_type == "space":
            j2 += 1
        if j2 < n and isinstance(tokens[j2], Ponctuation) and tokens[j2].text == ":":
            k = j2 + 1
            if k < n and isinstance(tokens[k], Separateur) and tokens[k].sep_type == "space":
                k += 1
            if k < n and isinstance(tokens[k], Formule) and tokens[k].text.isdigit():
                full = first.text + ":" + tokens[k].text
                if _detect_heure(full):
                    return full, k + 1, FormuleType.HEURE

    # ── Pourcentage : [PONCT(sign)] + FORMULE + PONCT("%"/"%") ──
    _SIGN_CHARS = {"-", "+", "±", "−"}
    # Variante avec signe
    if isinstance(first, Ponctuation) and first.text in _SIGN_CHARS:
        j = start + 1
        if (j < n and isinstance(tokens[j], Formule)
                and tokens[j].text.replace(".", "").replace("'", "").isdigit()
                and tokens[j].span[0] == first.span[1]):
            k = j + 1
            if k < n and isinstance(tokens[k], Ponctuation) and tokens[k].text in ("%", "‰"):
                full = first.text + tokens[j].text + tokens[k].text
                return full, k + 1, FormuleType.POURCENTAGE
            # PONCT(sign) + FORMULE + espace + PONCT(%/‰)
            if (k < n and isinstance(tokens[k], Separateur) and tokens[k].sep_type == "space"
                    and k + 1 < n and isinstance(tokens[k + 1], Ponctuation)
                    and tokens[k + 1].text in ("%", "‰")):
                full = first.text + tokens[j].text + tokens[k + 1].text
                return full, k + 2, FormuleType.POURCENTAGE
    # Variante sans signe
    if isinstance(first, Formule) and first.text.replace(".", "").replace("'", "").isdigit():
        j = start + 1
        if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text in ("%", "‰"):
            full = first.text + tokens[j].text
            return full, j + 1, FormuleType.POURCENTAGE
        # FORMULE + espace + PONCT(%/‰)
        if (j < n and isinstance(tokens[j], Separateur) and tokens[j].sep_type == "space"
                and j + 1 < n and isinstance(tokens[j + 1], Ponctuation)
                and tokens[j + 1].text in ("%", "‰")):
            full = first.text + tokens[j + 1].text
            return full, j + 2, FormuleType.POURCENTAGE

    # ── Monnaie : [PONCT(sign)] + FORMULE + PONCT("€"/"$"/"£"/"¥") ou inverse ──
    _CURRENCY_SYMBOLS = {"€", "$", "£", "¥"}
    _CURRENCY_CODES = {"EUR", "USD", "GBP", "CHF", "JPY"}
    # Variante avec signe : PONCT(sign) + FORMULE + PONCT(€) [+ FORMULE(centimes)]
    if isinstance(first, Ponctuation) and first.text in _SIGN_CHARS:
        j = start + 1
        if (j < n and isinstance(tokens[j], Formule)
                and tokens[j].text.replace(".", "").replace("'", "").replace(",", "").isdigit()
                and tokens[j].span[0] == first.span[1]):
            k = j + 1
            if k < n and isinstance(tokens[k], Ponctuation) and tokens[k].text in _CURRENCY_SYMBOLS:
                m2 = k + 1
                # Centimes collés
                if (m2 < n and isinstance(tokens[m2], Formule)
                        and tokens[m2].text.isdigit() and len(tokens[m2].text) <= 2
                        and tokens[m2].span[0] == tokens[k].span[1]):
                    full = first.text + tokens[j].text + tokens[k].text + tokens[m2].text
                    return full, m2 + 1, FormuleType.MONNAIE
                full = first.text + tokens[j].text + tokens[k].text
                return full, k + 1, FormuleType.MONNAIE
            # PONCT(sign) + FORMULE + espace + PONCT(€/$) ou MOT(code devise)
            if k < n and isinstance(tokens[k], Separateur) and tokens[k].sep_type == "space":
                if k + 1 < n and isinstance(tokens[k + 1], Ponctuation) and tokens[k + 1].text in _CURRENCY_SYMBOLS:
                    full = first.text + tokens[j].text + tokens[k + 1].text
                    return full, k + 2, FormuleType.MONNAIE
                if k + 1 < n and isinstance(tokens[k + 1], Mot) and tokens[k + 1].text.upper() in _CURRENCY_CODES:
                    full = first.text + tokens[j].text + " " + tokens[k + 1].text
                    return full, k + 2, FormuleType.MONNAIE
    # Variante sans signe
    if isinstance(first, Formule) and first.text.replace(".", "").replace("'", "").replace(",", "").isdigit():
        j = start + 1
        if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text in _CURRENCY_SYMBOLS:
            k = j + 1
            # 42€50 : FORMULE + PONCT(€) + FORMULE(centimes, contigu)
            if (k < n and isinstance(tokens[k], Formule)
                    and tokens[k].text.isdigit() and len(tokens[k].text) <= 2
                    and tokens[k].span[0] == tokens[j].span[1]):
                full = first.text + tokens[j].text + tokens[k].text
                return full, k + 1, FormuleType.MONNAIE
            full = first.text + tokens[j].text
            return full, j + 1, FormuleType.MONNAIE
        # FORMULE + espace + PONCT(€/$) ou MOT(code devise)
        if j < n and isinstance(tokens[j], Separateur) and tokens[j].sep_type == "space":
            if j + 1 < n and isinstance(tokens[j + 1], Ponctuation) and tokens[j + 1].text in _CURRENCY_SYMBOLS:
                full = first.text + tokens[j + 1].text
                return full, j + 2, FormuleType.MONNAIE
            if j + 1 < n and isinstance(tokens[j + 1], Mot) and tokens[j + 1].text.upper() in _CURRENCY_CODES:
                full = first.text + " " + tokens[j + 1].text
                return full, j + 2, FormuleType.MONNAIE
    # PONCT("€") + FORMULE
    if isinstance(first, Ponctuation) and first.text in _CURRENCY_SYMBOLS:
        j = start + 1
        if j < n and isinstance(tokens[j], Formule) and tokens[j].text.replace(".", "").replace("'", "").replace(",", "").isdigit():
            full = first.text + tokens[j].text
            return full, j + 1, FormuleType.MONNAIE
    # MOT(code devise) + espace + FORMULE ou MOT(code devise) + FORMULE
    if isinstance(first, Mot) and first.text.upper() in _CURRENCY_CODES:
        j = start + 1
        if j < n and isinstance(tokens[j], Separateur) and tokens[j].sep_type == "space":
            if j + 1 < n and isinstance(tokens[j + 1], Formule):
                full = first.text + " " + tokens[j + 1].text
                return full, j + 2, FormuleType.MONNAIE
        if j < n and isinstance(tokens[j], Formule):
            full = first.text + tokens[j].text
            return full, j + 1, FormuleType.MONNAIE

    # ── Minutes/secondes d'arc : FORMULE + PONCT("'") + FORMULE + PONCT('"') ──
    # Ex: 3'25" → MATHS (lu par lire_maths comme "3 prime 25 seconde")
    if isinstance(first, Formule) and first.text.isdigit():
        j = start + 1
        if (j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text == "'"
                and tokens[j].span[0] == first.span[1]):
            k = j + 1
            if (k < n and isinstance(tokens[k], Formule) and tokens[k].text.isdigit()
                    and tokens[k].span[0] == tokens[j].span[1]):
                m = k + 1
                if m < n and isinstance(tokens[m], Ponctuation) and tokens[m].text == '"':
                    full = first.text + "'" + tokens[k].text + '"'
                    return full, m + 1, FormuleType.MATHS

    # ── GPS : FORMULE + PONCT("°") + [FORMULE + PONCT("'")] + MOT(N/S/E/O/W) ──
    if isinstance(first, Formule) and first.text.replace(".", "").isdigit():
        j = start + 1
        if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text == "°":
            k = j + 1
            parts_text = first.text + "°"
            # Optionnel : minutes d'arc
            if k < n and isinstance(tokens[k], Formule) and tokens[k].text.isdigit():
                parts_text += tokens[k].text
                k += 1
                # Optionnel : PONCT("'") pour les minutes
                if k < n and isinstance(tokens[k], Ponctuation) and tokens[k].text == "'":
                    parts_text += "'"
                    k += 1
                    # Optionnel : secondes d'arc
                    if k < n and isinstance(tokens[k], Formule) and tokens[k].text.isdigit():
                        parts_text += tokens[k].text
                        k += 1
                        # Optionnel : PONCT('"')
                        if k < n and isinstance(tokens[k], Ponctuation) and tokens[k].text == '"':
                            parts_text += '"'
                            k += 1
            # Direction N/S/E/O/W
            if k < n and isinstance(tokens[k], Mot) and tokens[k].text.upper() in ("N", "S", "E", "O", "W"):
                parts_text += tokens[k].text
                if _detect_gps(parts_text):
                    return parts_text, k + 1, FormuleType.GPS

    # ── Intervalle : PONCT("["/"]") + FORMULE + PONCT(";"/",") + FORMULE + PONCT("["/"]") ──
    if isinstance(first, Ponctuation) and first.text in ("[", "]"):
        j = start + 1
        inner_parts = [first.text]
        # Collecter tokens internes
        while j < n and not (isinstance(tokens[j], Ponctuation) and tokens[j].text in ("[", "]") and len(inner_parts) > 1):
            inner_parts.append(tokens[j].text)
            j += 1
        # Fermant
        if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text in ("[", "]"):
            inner_parts.append(tokens[j].text)
            full = "".join(inner_parts)
            if _detect_intervalle(full):
                return full, j + 1, FormuleType.INTERVALLE

    # ── Ensemble : PONCT("{") + tokens internes + PONCT("}") → MATHS ──
    if isinstance(first, Ponctuation) and first.text == "{":
        j = start + 1
        inner_parts = [first.text]
        while j < n and not (isinstance(tokens[j], Ponctuation) and tokens[j].text == "}"):
            inner_parts.append(tokens[j].text)
            j += 1
        if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text == "}":
            inner_parts.append(tokens[j].text)
            full = "".join(inner_parts)
            return full, j + 1, FormuleType.MATHS

    # ── Page/Chapitre : MOT("page"/"chap"/"p"/"ch") + [PONCT(".")] + FORMULE ──
    _PAGE_WORDS = {"p", "page", "chap", "ch"}
    if isinstance(first, Mot) and first.text.lower() in _PAGE_WORDS:
        j = start + 1
        # Optionnel : point
        prefix = first.text
        if j < n and isinstance(tokens[j], Ponctuation) and tokens[j].text == ".":
            prefix += "."
            j += 1
        if j < n and isinstance(tokens[j], Formule) and tokens[j].text.isdigit():
            full = prefix + tokens[j].text
            if _detect_page_chapitre(full):
                return full, j + 1, FormuleType.PAGE_CHAPITRE
        # Avec espace : "page 42"
        if j < n and isinstance(tokens[j], Separateur) and tokens[j].sep_type == "space":
            if j + 1 < n and isinstance(tokens[j + 1], Formule) and tokens[j + 1].text.isdigit():
                full = prefix + tokens[j + 1].text
                if _detect_page_chapitre(full):
                    return full, j + 2, FormuleType.PAGE_CHAPITRE

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

    # ── Nombre + espace + unité → MATHS (5 km, 36.5 °C, -5 km) ──
    # Variante avec signe : PONCT(sign) + FORMULE + espace + unité
    if isinstance(first, Ponctuation) and first.text in _SIGN_CHARS:
        j = start + 1
        if (j < n and isinstance(tokens[j], Formule)
                and tokens[j].text.replace(".", "").replace(",", "").isdigit()
                and tokens[j].span[0] == first.span[1]):
            k = j + 1
            if k < n and isinstance(tokens[k], Separateur) and tokens[k].sep_type == "space":
                m2 = k + 1
                if m2 < n and isinstance(tokens[m2], Mot) and _is_unit_name(tokens[m2].text):
                    full = first.text + tokens[j].text + " " + tokens[m2].text
                    return full, m2 + 1, FormuleType.MATHS
                if (m2 + 1 < n and isinstance(tokens[m2], Ponctuation) and tokens[m2].text == "°"
                        and isinstance(tokens[m2 + 1], Mot) and tokens[m2 + 1].text in ("C", "F")):
                    full = first.text + tokens[j].text + " °" + tokens[m2 + 1].text
                    return full, m2 + 2, FormuleType.MATHS
    # Variante sans signe
    if isinstance(first, Formule) and first.text.replace(".", "").replace(",", "").isdigit():
        j = start + 1
        if j < n and isinstance(tokens[j], Separateur) and tokens[j].sep_type == "space":
            k = j + 1
            # Unité simple : MOT("km"), MOT("m"), ...
            if k < n and isinstance(tokens[k], Mot) and _is_unit_name(tokens[k].text):
                full = first.text + " " + tokens[k].text
                return full, k + 1, FormuleType.MATHS
            # Unité composée : °C, °F
            if (k + 1 < n and isinstance(tokens[k], Ponctuation) and tokens[k].text == "°"
                    and isinstance(tokens[k + 1], Mot) and tokens[k + 1].text in ("C", "F")):
                full = first.text + " °" + tokens[k + 1].text
                return full, k + 2, FormuleType.MATHS

    # ── Sigle/Maths : séquence de tokens adjacents (lettres+chiffres+opérateurs) ──
    # Accepte aussi un signe initial (-/+) collé à un chiffre : -12.5/54, -3x+1
    _MERGE_PUNCT = _MATHS_OPERATORS | set("()[]{}^/!")
    can_start = isinstance(first, (Formule, Mot))
    starts_with_sign = False
    starts_with_pm = False  # ± (toujours maths, jamais NOMBRE)
    starts_with_paren = False  # ( ouvrante
    if not can_start and isinstance(first, Ponctuation) and first.text in ("-", "+", "±"):
        nxt = start + 1
        if nxt < n and isinstance(tokens[nxt], (Formule, Mot)) and tokens[nxt].span[0] == first.span[1]:
            can_start = True
            starts_with_sign = True
            if first.text == "±":
                starts_with_pm = True
    # Parenthèse/crochet/accolade ouvrante : (3+5), [2x+3], {a+b}
    if not can_start and isinstance(first, Ponctuation) and first.text in ("(", "[", "{"):
        nxt = start + 1
        if nxt < n and isinstance(tokens[nxt], (Formule, Mot, Ponctuation)) and tokens[nxt].span[0] == first.span[1]:
            can_start = True
            starts_with_paren = True

    if can_start:
        j = start
        parts: list[str] = []
        has_digit = False
        has_op = False
        has_letter = False
        has_math_op = False
        has_func = False
        has_prime = False
        has_parens = False
        bracket_depth = 0
        prev_end = first.span[0]
        # Opérateurs maths qui peuvent être espacés (x < y, a + b = c)
        _SPACED_OPS = set("<>≤≥≠=≈≃≡∈∉⊂∪∩→←↔⇒⇔") | {"<=", ">=", "!=", "=="}
        last_was_spaced_op = False
        while j < n:
            t = tokens[j]
            # Espace : autoriser si suivi d'un opérateur maths espacé,
            # ou si on vient de consommer un opérateur espacé (espace post-op)
            if isinstance(t, Separateur):
                # Peek : SPACE + PONCT(op) + [SPACE] + (FORMULE|MOT)
                if (j + 2 < n
                        and isinstance(tokens[j + 1], Ponctuation)
                        and tokens[j + 1].text in _SPACED_OPS):
                    k2 = j + 2
                    if k2 < n and isinstance(tokens[k2], Separateur):
                        k2 += 1
                    if k2 < n and isinstance(tokens[k2], (Formule, Mot)):
                        # Accepter : sauter l'espace pré-opérateur
                        prev_end = t.span[1]
                        j += 1
                        continue
                # Espace post-opérateur : SPACE + (FORMULE|MOT)
                if last_was_spaced_op and j + 1 < n and isinstance(tokens[j + 1], (Formule, Mot)):
                    prev_end = t.span[1]
                    j += 1
                    last_was_spaced_op = False
                    continue
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
                if isinstance(t, Mot) and t.text.lower() in _MATHS_FUNCTIONS:
                    has_func = True
                # Formule contenant un opérateur maths (ex: √ scanné comme Formule)
                if isinstance(t, Formule) and any(c in _MATHS_OPERATORS for c in txt):
                    if not (j == start and starts_with_sign):
                        has_op = True
                        has_math_op = True
                parts.append(txt)
                prev_end = t.span[1]
                last_was_spaced_op = False
                j += 1
            elif isinstance(t, Ponctuation) and t.text in _MERGE_PUNCT:
                # Le signe initial ne compte pas comme opérateur interne
                if not (j == start and starts_with_sign):
                    has_op = True
                    # Vrai opérateur maths (pas juste brackets)
                    if t.text not in "()[]{}":
                        has_math_op = True
                if t.text in ("(", ")"):
                    has_parens = True
                if t.text in "([{":
                    bracket_depth += 1
                elif t.text in ")]}":
                    bracket_depth -= 1
                parts.append(t.text)
                last_was_spaced_op = t.text in _SPACED_OPS
                prev_end = t.span[1]
                j += 1
            elif isinstance(t, Ponctuation) and t.text in ",;" and bracket_depth != 0:
                # Virgule/point-virgule à l'intérieur de brackets (ensembles, intervalles)
                parts.append(t.text)
                prev_end = t.span[1]
                j += 1
            elif isinstance(t, Ponctuation) and t.text in ("'", "\u2032"):
                # Prime / apostrophe math — accepter dans la fusion
                has_prime = True
                parts.append(t.text)
                prev_end = t.span[1]
                j += 1
            else:
                break

        if j - start >= 2 and (has_digit or has_math_op or has_func
                               or (has_prime and has_parens)
                               or (has_letter and has_parens)):
            full = "".join(parts)
            # Signe + nombre seul → NOMBRE (-3.14, +42) — sauf ± qui est MATHS
            if starts_with_sign and not has_op and not has_letter:
                return full, j, FormuleType.NOMBRE
            if starts_with_pm or has_letter or has_op:
                # Sigle en priorité (ex: FR25, K2R) — pas d'opérateur, pas de signe
                if not has_op and not starts_with_sign and _detect_sigle(full):
                    return full, j, FormuleType.SIGLE
                # Numéro alphanumérique avec tirets (ex: CL-067-TS)
                if _detect_numero(full):
                    return full, j, FormuleType.NUMERO
                # Maths (ex: 2x+3=0, -12.5/54, sin(x), x∈A)
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


# ══════════════════════════════════════════════════════════════════════════════
# Classification + Merge (Passe 2)
# ══════════════════════════════════════════════════════════════════════════════

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
        if isinstance(tok, (Formule, Mot, Ponctuation)):
            merged = _try_merge_formule_group(tokens, i)
            if merged is not None:
                full_text, end_idx, forced_type = merged
                ftype = forced_type or _classify_formule_single(full_text)
                if ftype is not None:
                    # Re-fusion : absorber PONCT(/) + Formule/Mot contigus
                    # pour les unités composées (ex: "60 km" + "/" + "h")
                    while (end_idx < n
                           and isinstance(tokens[end_idx], Ponctuation)
                           and tokens[end_idx].text == "/"
                           and tokens[end_idx].span[0] == tokens[end_idx - 1].span[1]
                           and end_idx + 1 < n
                           and isinstance(tokens[end_idx + 1], (Formule, Mot))
                           and tokens[end_idx + 1].span[0] == tokens[end_idx].span[1]):
                        full_text += "/" + tokens[end_idx + 1].text
                        end_idx += 2
                        ftype = FormuleType.MATHS
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

        # Classifie les MOT qui sont en fait des ordinaux romains (XXIème, IIe, Ier)
        if isinstance(tok, Mot) and _detect_ordinal(tok.text):
            formule = Formule(
                type=TokenType.FORMULE,
                text=tok.text,
                span=tok.span,
                formule_type=FormuleType.ORDINAL,
                children=_build_formule_children(tok.text, FormuleType.ORDINAL, tok.span[0]),
                valeur=tok.text,
            )
            result.append(formule)
            i += 1
            continue

        # Classifie les MOT qui sont des unités seules (km, cm, etc.)
        # Sauf si suivi d'une apostrophe (élision : m'appelle, s'il)
        if isinstance(tok, Mot) and _is_unit_name(tok.text):
            _next_is_apos = (i + 1 < n and isinstance(tokens[i + 1], Separateur)
                             and tokens[i + 1].sep_type == "apostrophe")
            if not _next_is_apos:
                formule = Formule(
                    type=TokenType.FORMULE,
                    text=tok.text,
                    span=tok.span,
                    formule_type=FormuleType.MATHS,
                    children=_build_formule_children(tok.text, FormuleType.MATHS, tok.span[0]),
                    valeur=tok.text,
                )
                result.append(formule)
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
        i = 0
        while i < len(text) and (text[i].isdigit() or text[i] in "IVXLCDM"):
            i += 1
        if i == 0:
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
        from lectura_tokeniseur.maths import tokenize_maths
        math_toks = tokenize_maths(text)
        children = []
        scan = 0
        for mt in math_toks:
            idx = text.find(mt.text, scan)
            if idx < 0:
                idx = scan
            mt_span = (offset + idx, offset + idx + len(mt.text))
            scan = idx + len(mt.text)
            if mt.math_type == "number":
                children.append(Formule(
                    type=TokenType.FORMULE, text=mt.text,
                    span=mt_span,
                    formule_type=FormuleType.NOMBRE, valeur=mt.text,
                ))
            elif mt.math_type in ("variable", "function", "greek", "unit"):
                children.append(Mot(
                    type=TokenType.MOT, text=mt.text,
                    span=mt_span, ortho=mt.text.lower(),
                ))
            else:
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=mt.text,
                    span=mt_span,
                ))
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

    # ── Nouveaux types ──

    if ftype == FormuleType.HEURE:
        # Enfants : heures (nombre), séparateur (h/:), minutes (nombre), etc.
        children = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isdigit():
                start = i
                while i < len(text) and text[i].isdigit():
                    i += 1
                children.append(Formule(
                    type=TokenType.FORMULE, text=text[start:i],
                    span=(offset + start, offset + i),
                    formule_type=FormuleType.NOMBRE, valeur=text[start:i],
                ))
                continue
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
            if ch in ":":
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=ch,
                    span=(offset + i, offset + i + 1),
                ))
                i += 1
                continue
            # Espaces et autres
            i += 1
        return children

    if ftype == FormuleType.MONNAIE:
        # Enfants : montant (nombre) + symbole/code devise
        children = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isdigit():
                start = i
                while i < len(text) and (text[i].isdigit() or text[i] in ".,'" ):
                    i += 1
                # Reculer si finit par ponctuation
                while i > start and text[i - 1] in ".,'" :
                    i -= 1
                children.append(Formule(
                    type=TokenType.FORMULE, text=text[start:i],
                    span=(offset + start, offset + i),
                    formule_type=FormuleType.NOMBRE, valeur=text[start:i],
                ))
                continue
            if ch in "€$£¥":
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=ch,
                    span=(offset + i, offset + i + 1),
                ))
                i += 1
                continue
            if ch.isalpha():
                start = i
                while i < len(text) and text[i].isalpha():
                    i += 1
                word = text[start:i]
                children.append(Mot(
                    type=TokenType.MOT, text=word,
                    span=(offset + start, offset + i),
                    ortho=word.upper(),
                ))
                continue
            i += 1
        return children

    if ftype == FormuleType.POURCENTAGE:
        # Enfants : nombre + signe %/‰
        children = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isdigit():
                start = i
                while i < len(text) and (text[i].isdigit() or text[i] in ".'"):
                    i += 1
                children.append(Formule(
                    type=TokenType.FORMULE, text=text[start:i],
                    span=(offset + start, offset + i),
                    formule_type=FormuleType.NOMBRE, valeur=text[start:i],
                ))
                continue
            if ch in "%‰":
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=ch,
                    span=(offset + i, offset + i + 1),
                ))
                i += 1
                continue
            i += 1
        return children

    if ftype == FormuleType.GPS:
        # Enfants : degrés, °, minutes, ', secondes, ", direction
        children = []
        i = 0
        while i < len(text):
            ch = text[i]
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
            if ch in "°'\"":
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=ch,
                    span=(offset + i, offset + i + 1),
                ))
                i += 1
                continue
            if ch.isalpha():
                children.append(Mot(
                    type=TokenType.MOT, text=ch,
                    span=(offset + i, offset + i + 1),
                    ortho=ch.upper(),
                ))
                i += 1
                continue
            i += 1
        return children

    if ftype == FormuleType.INTERVALLE:
        # Enfants : crochets, nombres, séparateur
        children = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in "[]":
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=ch,
                    span=(offset + i, offset + i + 1),
                ))
                i += 1
                continue
            if ch.isdigit() or (ch in "-+" and i + 1 < len(text) and text[i + 1].isdigit()):
                start = i
                if ch in "-+":
                    i += 1
                while i < len(text) and (text[i].isdigit() or text[i] in ".,"):
                    i += 1
                children.append(Formule(
                    type=TokenType.FORMULE, text=text[start:i],
                    span=(offset + start, offset + i),
                    formule_type=FormuleType.NOMBRE, valeur=text[start:i],
                ))
                continue
            if ch in ";,":
                children.append(Ponctuation(
                    type=TokenType.PONCTUATION, text=ch,
                    span=(offset + i, offset + i + 1),
                ))
                i += 1
                continue
            i += 1
        return children

    if ftype == FormuleType.PAGE_CHAPITRE:
        # Enfants : mot (page/chap) + nombre
        children = []
        # Trouver où finit le préfixe texte et où commence le nombre
        m = re.match(r"^([a-zA-Z]+\.?\s*)(\d+)$", text)
        if m:
            prefix, num = m.group(1), m.group(2)
            # Enlever espaces de fin du préfixe pour le span
            prefix_stripped = prefix.rstrip()
            children.append(Mot(
                type=TokenType.MOT, text=prefix_stripped,
                span=(offset, offset + len(prefix_stripped)),
                ortho=prefix_stripped.lower(),
            ))
            num_start = len(prefix)
            children.append(Formule(
                type=TokenType.FORMULE, text=num,
                span=(offset + num_start, offset + num_start + len(num)),
                formule_type=FormuleType.NOMBRE, valeur=num,
            ))
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
    # Nouveaux types
    if ftype == FormuleType.HEURE:
        return text
    if ftype == FormuleType.MONNAIE:
        return text
    if ftype == FormuleType.POURCENTAGE:
        return text
    if ftype == FormuleType.INTERVALLE:
        return text
    if ftype == FormuleType.GPS:
        return text
    if ftype == FormuleType.PAGE_CHAPITRE:
        # Extraire le numéro
        m = re.search(r"\d+", text)
        return m.group(0) if m else text
    return text
