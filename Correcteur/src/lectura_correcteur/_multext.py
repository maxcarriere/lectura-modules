"""Decodeur de tags Multext-GRACE pour le francais (version allegee).

Copie allegee de lectura_lexique._multext : seul ``decoder_multext()``
est conserve (pas besoin de ``filtre_multext`` qui genere du SQL LIKE).

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
            # Check for extended subtypes (Np-p, Np-f)
            if tag[1] == "p" and len(tag) > 3 and tag[2] == "-" and tag[3] in ("p", "f"):
                result["sous_type"] = "propre"
            else:
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

    return result
