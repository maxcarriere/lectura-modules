"""Tokenisation fine des formules mathématiques.

Produit des MathToken typés (number, variable, function, operator, greek,
unit, superscript, subscript, bracket, prime, factorial, second).

Ce module est autonome (zéro dépendance externe) et peut être utilisé
indépendamment du reste du tokeniseur.
"""

from __future__ import annotations

from typing import NamedTuple


# ══════════════════════════════════════════════════════════════════════════════
# MathToken
# ══════════════════════════════════════════════════════════════════════════════

class MathToken(NamedTuple):
    """Token typé pour les formules mathématiques."""
    text: str
    math_type: str  # number, variable, function, operator, greek, unit,
                    # superscript, subscript, bracket, prime, factorial, second
    extra: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Constantes
# ══════════════════════════════════════════════════════════════════════════════

MATHS_OPS = set("+-−±=≠<>≤≥×*÷/^√∞∑∏∫∂∇∈∉⊂∪∩→←↔⇒⇔°%‰≈≃≡")
MATHS_BRACKETS = set("()[]{}⟨⟩|")
MULTI_OPS = {"<=", ">=", "!=", "=="}
_MULTI_OPS_FIRST = {s[0] for s in MULTI_OPS}

SUPERSCRIPTS: dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "ⁿ": "n",
}

SUBSCRIPTS_UNICODE: dict[str, str] = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}
_SUBSCRIPT_UNICODE_CHARS = set(SUBSCRIPTS_UNICODE.keys())

UNIT_MULTI = sorted(
    ["°C", "°F", "km", "kg", "mg", "cm", "mm", "ml"],
    key=len, reverse=True,
)
UNIT_SINGLE = {"g", "m", "s", "h", "L"}
UNIT_NAMES_LOWER = {mu.lower() for mu in UNIT_MULTI} | {u for u in UNIT_SINGLE}

FUNC_NAMES = {"sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs"}
FUNCTION_LIKE_VARS = set("fghkFGHK")

GREEK_CHARS = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩς")


# ══════════════════════════════════════════════════════════════════════════════
# tokenize_maths
# ══════════════════════════════════════════════════════════════════════════════

def tokenize_maths(text: str) -> list[MathToken]:
    """Tokenise une formule maths.

    Types retournés : number, operator, variable, function, bracket,
    superscript, subscript, greek, unit, prime, factorial, second.
    """
    tokens: list[MathToken] = []
    i = 0
    n = len(text)
    brace_depth = 0  # profondeur des accolades {}, pour inhiber virgule décimale

    def _ascii_digit(c: str) -> bool:
        return "0" <= c <= "9"

    while i < n:
        ch = text[i]

        # Espaces
        if ch in (" ", "\t", "\u202f"):
            i += 1
            continue

        # Opérateurs multi-caractères (<=, >=, !=, ==)
        if ch in _MULTI_OPS_FIRST and i + 1 < n:
            duo = text[i:i + 2]
            if duo in MULTI_OPS:
                tokens.append(MathToken(duo, "operator"))
                i += 2
                continue

        # Superscripts unicode (AVANT isdigit car Python traite ² comme digit)
        if ch in SUPERSCRIPTS:
            j = i
            while j < n and text[j] in SUPERSCRIPTS:
                j += 1
            raw = text[i:j]
            converted = "".join(SUPERSCRIPTS[c] for c in raw)
            tokens.append(MathToken(raw, "superscript", converted))
            i = j
            continue

        # Subscripts unicode (₀₁₂…)
        if ch in _SUBSCRIPT_UNICODE_CHARS:
            j = i
            while j < n and text[j] in _SUBSCRIPT_UNICODE_CHARS:
                j += 1
            raw = text[i:j]
            converted = "".join(SUBSCRIPTS_UNICODE[c] for c in raw)
            tokens.append(MathToken(raw, "subscript", converted))
            i = j
            continue

        # Subscript ASCII : _suivi de alphanumérique
        if ch == "_" and i + 1 < n and text[i + 1].isalnum():
            j = i + 1
            while j < n and text[j].isalnum():
                j += 1
            content = text[i + 1 : j]
            tokens.append(MathToken(text[i:j], "subscript", content))
            i = j
            continue

        # Chiffres (nombres entiers ou décimaux)
        if _ascii_digit(ch):
            j = i
            while j < n and _ascii_digit(text[j]):
                j += 1
            # Partie décimale (virgule inhibée dans les accolades {1,2,3})
            decimal_sep = "." if brace_depth > 0 else ".,"
            if j < n and text[j] in decimal_sep and j + 1 < n and _ascii_digit(text[j + 1]):
                j += 1
                while j < n and _ascii_digit(text[j]):
                    j += 1
            tokens.append(MathToken(text[i:j], "number"))

            # Unité collée au nombre (12kg, 15°C…)
            if j < n:
                for mu in UNIT_MULTI:
                    end = j + len(mu)
                    if end <= n and text[j:end] == mu:
                        tokens.append(MathToken(mu, "unit"))
                        j = end
                        break
                else:
                    c2 = text[j]
                    if c2 in UNIT_SINGLE and (j + 1 >= n or not text[j + 1].isalpha()):
                        tokens.append(MathToken(c2, "unit"))
                        j += 1
            i = j
            continue

        # Prime (′ ou ') après variable/greek/subscript/superscript
        if ch in ("'", "\u2032") and tokens and tokens[-1].math_type in (
            "variable", "function", "greek", "subscript", "superscript",
        ):
            tokens.append(MathToken(ch, "prime"))
            i += 1
            continue

        # Double-prime / secondes d'arc (") après nombre ou prime
        if ch == '"' and tokens and tokens[-1].math_type in (
            "number", "prime", "variable", "greek", "subscript", "superscript",
        ):
            tokens.append(MathToken('"', "second"))
            i += 1
            continue

        # Factorielle
        if ch == "!" and tokens and tokens[-1].math_type in (
            "number", "variable", "bracket", "subscript", "superscript", "greek",
        ):
            tokens.append(MathToken("!", "factorial"))
            i += 1
            continue

        # Parenthèses / crochets / pipes
        if ch in MATHS_BRACKETS:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            tokens.append(MathToken(ch, "bracket"))
            i += 1
            continue

        # Lettres grecques
        if ch in GREEK_CHARS:
            tokens.append(MathToken(ch, "greek"))
            i += 1
            continue

        # Opérateurs — vérifier d'abord les unités commençant par ° (°C, °F)
        if ch in MATHS_OPS:
            if ch == "°":
                matched_unit = False
                for mu in UNIT_MULTI:
                    if mu.startswith("°"):
                        end = i + len(mu)
                        if end <= n and text[i:end] == mu:
                            tokens.append(MathToken(mu, "unit"))
                            i = end
                            matched_unit = True
                            break
                if matched_unit:
                    continue
            tokens.append(MathToken(ch, "operator"))
            i += 1
            continue

        # Lettres latines
        if ch.isalpha() and ch not in SUPERSCRIPTS and ch not in _SUBSCRIPT_UNICODE_CHARS:
            # Essayer d'abord les unités multi-caractères (km, kg, cm…)
            matched_unit = False
            for mu in UNIT_MULTI:
                if mu[0].isalpha():
                    end = i + len(mu)
                    if end <= n and text[i:end] == mu:
                        tokens.append(MathToken(mu, "unit"))
                        i = end
                        matched_unit = True
                        break
            if matched_unit:
                continue

            j = i
            while j < n and text[j].isalpha() and text[j] not in SUPERSCRIPTS and text[j] not in _SUBSCRIPT_UNICODE_CHARS:
                j += 1
            word = text[i:j]
            wl = word.lower()
            if wl in FUNC_NAMES:
                tokens.append(MathToken(word, "function"))
            elif wl in ("min",):
                tokens.append(MathToken(word, "unit"))
            elif len(word) == 1:
                tokens.append(MathToken(word, "variable"))
            else:
                for c in word:
                    tokens.append(MathToken(c, "variable"))
            i = j
            continue

        # Virgule / point-virgule (séparateur dans ensembles/intervalles)
        if ch in ",;":
            tokens.append(MathToken(ch, "separator"))
            i += 1
            continue

        # Inconnu
        i += 1

    # Post-traitement : requalifier les variables comme unités dans un contexte unité
    _requalify_unit_vars(tokens)

    return tokens


def _requalify_unit_vars(tokens: list[MathToken]) -> None:
    """Requalifie les variables en unités dans un contexte d'unité.

    Ex: km/h → variable 'h' après / + unit → requalifié en unit.
    """
    n = len(tokens)
    for i in range(n):
        if tokens[i].math_type == "variable" and tokens[i].text.lower() in UNIT_NAMES_LOWER:
            # Variable after number → requalify as unit
            if i >= 1 and tokens[i - 1].math_type == "number":
                tokens[i] = MathToken(tokens[i].text, "unit")
            # Adjacent à une unité via /
            elif i >= 2 and tokens[i - 1].text == "/" and tokens[i - 1].math_type == "operator" and tokens[i - 2].math_type == "unit":
                tokens[i] = MathToken(tokens[i].text, "unit")
            elif i + 2 < n and tokens[i + 1].text == "/" and tokens[i + 1].math_type == "operator" and tokens[i + 2].math_type in ("unit", "variable"):
                tokens[i] = MathToken(tokens[i].text, "unit")
