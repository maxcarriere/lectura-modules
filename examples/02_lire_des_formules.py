"""Exemple 2 — Lire des formules en francais (nombres, dates, heures...).

pip install lectura-formules
"""

from lectura_formules import (
    lire_formule,
    lire_nombre,
    lire_date,
    lire_telephone,
    lire_heure,
    lire_ordinal,
    lire_fraction,
    int_to_roman,
    roman_to_int,
)

# --- Nombres ---
r = lire_nombre("42")
print(f"42        → {r.display_fr}")

r = lire_nombre("1000000")
print(f"1000000   → {r.display_fr}")

# --- Dates ---
r = lire_date("25/12/2025")
print(f"25/12/2025 → {r.display_fr}")

# --- Heures ---
r = lire_heure("14h30")
print(f"14h30     → {r.display_fr}")

# --- Telephone ---
r = lire_telephone("06 12 34 56 78")
print(f"Telephone → {r.display_fr}")

# --- Ordinaux ---
r = lire_ordinal("1er")
print(f"1er       → {r.display_fr}")

# --- Fractions ---
r = lire_fraction("3/4")
print(f"3/4       → {r.display_fr}")

# --- Chiffres romains ---
print(f"\n42 en romain : {int_to_roman(42)}")
print(f"XLII en arabe : {roman_to_int('XLII')}")

# --- Dispatch automatique via lire_formule() ---
print("\n--- Dispatch automatique ---")
for type_f, texte in [
    ("NOMBRE", "2025"),
    ("DATE", "14/07/1789"),
    ("TELEPHONE", "01 23 45 67 89"),
    ("ORDINAL", "3eme"),
    ("SIGLE", "SNCF"),
]:
    r = lire_formule(type_f, texte)
    print(f"  {type_f:12s} {texte!r:25s} → {r.display_fr}")
