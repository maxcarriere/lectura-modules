"""Chiffres romains — conversion bidirectionnelle int ↔ romain.

Supporte la plage 1–39 999 avec vinculum (V̅ = 5000, X̅ = 10 000).

Licence : CC-BY-SA-4.0
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
# Tables de conversion
# ══════════════════════════════════════════════════════════════════════════════

# Paires (valeur, symbole) triées par valeur décroissante
_INT_TO_ROMAN: list[tuple[int, str]] = [
    (10000, "X̅"),
    (9000, "MX̅"),
    (5000, "V̅"),
    (4000, "MV̅"),
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]

# Mapping inverse pour le parsing
_ROMAN_VALUES: list[tuple[str, int]] = [
    ("X̅", 10000),
    ("V̅", 5000),
    ("CM", 900),
    ("CD", 400),
    ("XC", 90),
    ("XL", 40),
    ("IX", 9),
    ("IV", 4),
    ("M", 1000),
    ("D", 500),
    ("C", 100),
    ("L", 50),
    ("X", 10),
    ("V", 5),
    ("I", 1),
]


# ══════════════════════════════════════════════════════════════════════════════
# int → romain
# ══════════════════════════════════════════════════════════════════════════════

def int_to_roman(n: int) -> str:
    """Convertit un entier en chiffres romains.

    Plage supportée : 1–39 999 (avec vinculum pour 5000+).
    Lève ValueError si n est hors limites.
    """
    if not isinstance(n, int) or n < 1 or n > 39999:
        raise ValueError(f"int_to_roman : n={n} hors limites (1–39999)")

    result: list[str] = []
    remaining = n
    for value, symbol in _INT_TO_ROMAN:
        while remaining >= value:
            result.append(symbol)
            remaining -= value
    return "".join(result)


# ══════════════════════════════════════════════════════════════════════════════
# romain → int
# ══════════════════════════════════════════════════════════════════════════════

def roman_to_int(s: str) -> int:
    """Convertit une chaîne en chiffres romains en entier.

    Supporte les symboles standard (I, V, X, L, C, D, M)
    et le vinculum (V̅, X̅).
    Lève ValueError si la chaîne est invalide.
    """
    if not s:
        raise ValueError("roman_to_int : chaîne vide")

    result = 0
    i = 0
    text = s.upper()

    while i < len(text):
        matched = False
        # Essayer les tokens multi-caractères d'abord
        for token, value in _ROMAN_VALUES:
            tok_upper = token.upper()
            if text[i:i + len(tok_upper)] == tok_upper:
                result += value
                i += len(tok_upper)
                matched = True
                break
        if not matched:
            # Caractère simple avec règle soustractive
            ch = text[i]
            val = _SINGLE_VALUES.get(ch)
            if val is None:
                raise ValueError(f"roman_to_int : caractère invalide '{s[i]}'")
            # Vérifier règle soustractive
            if i + 1 < len(text):
                next_val = _SINGLE_VALUES.get(text[i + 1])
                if next_val and next_val > val:
                    result += next_val - val
                    i += 2
                    continue
            result += val
            i += 1

    return result


# Valeurs des caractères simples pour le parsing soustractif
_SINGLE_VALUES: dict[str, int] = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}
