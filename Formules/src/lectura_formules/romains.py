"""Chiffres romains — conversion bidirectionnelle int ↔ romain.

Supporte la plage 1–39 999 avec vinculum (V̅ = 5000, X̅ = 10 000).

Licence : CC-BY-SA-4.0
"""

from __future__ import annotations

from lectura_formules._chargeur import (
    romains_int_to_roman as _load_int_to_roman,
    romains_values as _load_roman_values,
    romains_single as _load_single_values,
)


# ══════════════════════════════════════════════════════════════════════════════
# Tables de conversion (chargees depuis JSON)
# ══════════════════════════════════════════════════════════════════════════════

_MODE_API = False
try:
    _INT_TO_ROMAN = _load_int_to_roman()
    _ROMAN_VALUES = _load_roman_values()
except FileNotFoundError:
    _MODE_API = True
    _INT_TO_ROMAN = []
    _ROMAN_VALUES = []


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


try:
    _SINGLE_VALUES = _load_single_values()
except FileNotFoundError:
    _SINGLE_VALUES = {}
