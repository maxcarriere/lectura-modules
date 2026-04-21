"""Decodeur de tags Multext-GRACE pour le francais.

Le tag Multext encode la categorie et les traits flexionnels en une chaine
de caracteres positionnels. Exemple : ``Vmip3s`` = Verb main indicatif present
3e personne singulier.

Ce module fournit :
- ``decoder_multext(tag)`` : tag → dict de traits lisibles
- ``filtre_multext(**criteres)`` : criteres → pattern SQL LIKE

Reference : http://www.ilc.cnr.it/EAGLES96/annotate/node20.html
"""

from __future__ import annotations

# ── Tables de correspondance ─────────────────────────────────────────

# Position 0 : Categorie majeure
_POS = {
    "N": "NOM",
    "V": "VER",
    "A": "ADJ",
    "D": "DET",
    "P": "PRO",
    "R": "ADV",
    "S": "PRE",
    "C": "CON",
    "I": "ONO",
    "X": "AUT",
    "Y": "SIGLE",
}

# Position 1 : Sous-type (depend de la categorie)
_SUBTYPE = {
    "N": {"c": "commun", "p": "propre"},
    "V": {"m": "principal", "a": "auxiliaire"},
    "A": {"f": "qualificatif", "o": "ordinal", "i": "indefini",
           "p": "possessif", "d": "demonstratif"},
    "D": {"a": "article", "d": "demonstratif", "i": "indefini",
           "p": "possessif", "t": "interrogatif", "e": "exclamatif",
           "r": "relatif", "s": "partitif"},
    "P": {"p": "personnel", "d": "demonstratif", "i": "indefini",
           "s": "possessif", "t": "interrogatif", "r": "relatif",
           "x": "reflexif"},
    "Y": {"s": "sigle", "a": "acronyme"},
}

# Modes verbaux (position 2 des verbes)
_MODE = {
    "i": "indicatif",
    "s": "subjonctif",
    "m": "imperatif",
    "c": "conditionnel",
    "n": "infinitif",
    "p": "participe",
    "g": "gerondif",
}

# Temps (position 3 des verbes)
_TEMPS = {
    "p": "present",
    "i": "imparfait",
    "f": "futur",
    "s": "passe_simple",
}

# Personne (position 4 des verbes, position variable pour autres)
_PERSONNE = {"1": "1", "2": "2", "3": "3"}

# Nombre
_NOMBRE = {"s": "singulier", "p": "pluriel"}

# Genre
_GENRE = {"m": "masculin", "f": "feminin"}


def decoder_multext(tag: str) -> dict[str, str]:
    """Decode un tag Multext-GRACE en dictionnaire de traits lisibles.

    Args:
        tag: Tag multext (ex: "Vmip3s", "Ncms", "Afpfs")

    Returns:
        Dictionnaire avec les cles : pos, sous_type, mode, temps,
        personne, nombre, genre. Les valeurs absentes sont omises.
    """
    if not tag:
        return {}

    result: dict[str, str] = {}
    cat = tag[0] if len(tag) > 0 else ""

    result["pos"] = _POS.get(cat, cat)

    if cat == "V":
        # Verbe : V type mode temps personne nombre genre
        if len(tag) > 1 and tag[1] != "-":
            st = _SUBTYPE.get("V", {}).get(tag[1])
            if st:
                result["sous_type"] = st
        if len(tag) > 2 and tag[2] != "-":
            result["mode"] = _MODE.get(tag[2], tag[2])
        if len(tag) > 3 and tag[3] != "-":
            result["temps"] = _TEMPS.get(tag[3], tag[3])
        if len(tag) > 4 and tag[4] != "-":
            result["personne"] = _PERSONNE.get(tag[4], tag[4])
        if len(tag) > 5 and tag[5] != "-":
            result["nombre"] = _NOMBRE.get(tag[5], tag[5])
        if len(tag) > 6 and tag[6] != "-":
            result["genre"] = _GENRE.get(tag[6], tag[6])

    elif cat == "N":
        # Nom : N type genre nombre
        if len(tag) > 1 and tag[1] != "-":
            st = _SUBTYPE.get("N", {}).get(tag[1])
            if st:
                result["sous_type"] = st
        if len(tag) > 2 and tag[2] != "-":
            result["genre"] = _GENRE.get(tag[2], tag[2])
        if len(tag) > 3 and tag[3] != "-":
            result["nombre"] = _NOMBRE.get(tag[3], tag[3])

    elif cat == "A":
        # Adjectif : A type degre genre nombre
        if len(tag) > 1 and tag[1] != "-":
            st = _SUBTYPE.get("A", {}).get(tag[1])
            if st:
                result["sous_type"] = st
        # Position 2 = degre (p=positif, c=comparatif, s=superlatif)
        if len(tag) > 3 and tag[3] != "-":
            result["genre"] = _GENRE.get(tag[3], tag[3])
        if len(tag) > 4 and tag[4] != "-":
            result["nombre"] = _NOMBRE.get(tag[4], tag[4])

    elif cat in ("D", "P"):
        # Determinant / Pronom : type personne genre nombre
        if len(tag) > 1 and tag[1] != "-":
            st = _SUBTYPE.get(cat, {}).get(tag[1])
            if st:
                result["sous_type"] = st
        if len(tag) > 2 and tag[2] != "-":
            result["personne"] = _PERSONNE.get(tag[2], tag[2])
        if len(tag) > 3 and tag[3] != "-":
            result["genre"] = _GENRE.get(tag[3], tag[3])
        if len(tag) > 4 and tag[4] != "-":
            result["nombre"] = _NOMBRE.get(tag[4], tag[4])

    elif cat == "Y":
        # Sigle : Y sous_type (s=sigle epele, a=acronyme)
        if len(tag) > 1 and tag[1] != "-":
            st = _SUBTYPE.get("Y", {}).get(tag[1])
            if st:
                result["sous_type"] = st

    return result


# ── Positions par categorie pour les traits filtres ──────────────────

# Mapping (categorie, trait) -> position dans le tag
_TRAIT_POSITIONS: dict[str, dict[str, int]] = {
    "V": {"mode": 2, "temps": 3, "personne": 4, "nombre": 5, "genre": 6},
    "N": {"genre": 2, "nombre": 3},
    "A": {"genre": 3, "nombre": 4},
    "D": {"personne": 2, "genre": 3, "nombre": 4},
    "P": {"personne": 2, "genre": 3, "nombre": 4},
}

# Mapping inverse : valeur lisible -> code multext
_MODE_INV = {v: k for k, v in _MODE.items()}
_TEMPS_INV = {v: k for k, v in _TEMPS.items()}
_NOMBRE_INV = {v: k for k, v in _NOMBRE.items()}
_GENRE_INV = {v: k for k, v in _GENRE.items()}
_PERSONNE_INV = {v: k for k, v in _PERSONNE.items()}

_TRAIT_INV: dict[str, dict[str, str]] = {
    "mode": _MODE_INV,
    "temps": _TEMPS_INV,
    "nombre": _NOMBRE_INV,
    "genre": _GENRE_INV,
    "personne": _PERSONNE_INV,
}


def filtre_multext(
    *,
    pos: str | None = None,
    mode: str | None = None,
    temps: str | None = None,
    personne: str | None = None,
    nombre: str | None = None,
    genre: str | None = None,
) -> str:
    """Construit un pattern SQL LIKE pour filtrer par traits multext.

    Les criteres non specifies sont remplaces par ``_`` (un caractere quelconque)
    ou ``%`` (fin ouverte).

    Args:
        pos: Categorie ("NOM", "VER", "ADJ"...)
        mode: Mode verbal ("indicatif", "subjonctif"...)
        temps: Temps ("present", "imparfait"...)
        personne: Personne ("1", "2", "3")
        nombre: Nombre ("singulier", "pluriel")
        genre: Genre ("masculin", "feminin")

    Returns:
        Pattern SQL LIKE (ex: "Vm_p3s%")
    """
    # Determiner la categorie cible
    pos_code = ""
    if pos:
        pos_upper = pos.upper()
        for code, label in _POS.items():
            if label == pos_upper or code == pos_upper:
                pos_code = code
                break
        if not pos_code:
            # Tentative directe
            pos_code = pos[0].upper()

    criteres = {
        "mode": mode,
        "temps": temps,
        "personne": personne,
        "nombre": nombre,
        "genre": genre,
    }

    # Si pas de pos specifiee, essayer d'inferer depuis les criteres
    if not pos_code:
        if mode or temps:
            pos_code = "V"

    if not pos_code:
        # Pattern large : juste les traits qu'on connait en fin
        # Ne peut pas etre precis sans connaitre la categorie
        return "%"

    # Construire le pattern pour cette categorie
    positions = _TRAIT_POSITIONS.get(pos_code, {})
    if not positions:
        return f"{pos_code}%"

    max_pos = max(positions.values())
    pattern = [pos_code] + ["_"] * max_pos

    for trait_name, trait_value in criteres.items():
        if trait_value is None:
            continue
        idx = positions.get(trait_name)
        if idx is None:
            continue
        # Encoder la valeur
        inv_map = _TRAIT_INV.get(trait_name, {})
        code = inv_map.get(trait_value, trait_value)
        if len(code) == 1:
            pattern[idx] = code

    return "".join(pattern) + "%"
