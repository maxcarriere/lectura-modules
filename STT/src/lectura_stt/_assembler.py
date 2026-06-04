"""Reconstruction du texte a partir des mots ortho + ponctuation.

Gere la majuscule initiale, les elisions (l', d', j', ...) et la ponctuation.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

# Voyelles graphemiques (debut de mot) pour la detection d'elision
_VOYELLES = set("aeiouyàâäéèêëïîôùûüœæ")
_H_ASPIRE = {
    "hache", "haine", "hamac", "hameau", "hanche", "handicap", "hangar",
    "hanter", "harangue", "harceler", "hardi", "harem", "hareng", "haricot",
    "harnais", "harpe", "hasard", "hate", "hâte", "haut", "haute", "hauteur",
    "heros", "héros", "hetre", "hêtre", "hibou", "hideux", "hierarchie",
    "hiérarchie", "hisser", "hobby", "hocher", "hockey", "hollande", "homard",
    "hongrie", "honte", "hoquet", "horde", "hors", "houille", "houle",
    "hublot", "huche", "huer", "hurler", "hussard", "hutte",
}

# Clitiques pouvant s'elider devant voyelle
_ELISIONS: dict[str, str] = {
    "le": "l'",
    "la": "l'",
    "de": "d'",
    "je": "j'",
    "ne": "n'",
    "se": "s'",
    "que": "qu'",
    "me": "m'",
    "te": "t'",
    "ce": "c'",
}


def _commence_par_voyelle(mot: str) -> bool:
    """Verifie si un mot commence par une voyelle (elision possible).

    Prend en compte le h muet (elision) vs h aspire (pas d'elision).
    """
    if not mot:
        return False
    first = mot[0].lower()
    if first in _VOYELLES:
        return True
    if first == "h":
        return mot.lower() not in _H_ASPIRE
    return False


def assembler_texte(
    mots_ortho: list[str],
    ponctuation_finale: str = "",
) -> str:
    """Reconstruit le texte a partir des mots ortho + ponctuation.

    - Majuscule au premier mot
    - Gestion des elisions (l' + voyelle, d' + voyelle, etc.)
    - Ponctuation finale

    Parameters
    ----------
    mots_ortho : list[str]
        Liste de mots orthographiques (ex: ["bonjour", "le", "monde"]).
    ponctuation_finale : str
        Ponctuation a ajouter en fin de phrase (".", "?", "!", "").

    Returns
    -------
    str
        Texte reconstruit (ex: "Bonjour le monde.").
    """
    if not mots_ortho:
        return ""

    # Filtrer les mots vides
    mots = [m for m in mots_ortho if m]
    if not mots:
        return ""

    # Appliquer les elisions
    parts: list[str] = []
    i = 0
    while i < len(mots):
        mot = mots[i]
        mot_lower = mot.lower()

        # Verifier si c'est un clitique elidable
        if mot_lower in _ELISIONS and i + 1 < len(mots):
            next_mot = mots[i + 1]
            if _commence_par_voyelle(next_mot):
                elided = _ELISIONS[mot_lower]
                # Preserver la casse si le clitique etait capitalise
                if mot[0].isupper():
                    elided = elided[0].upper() + elided[1:]
                parts.append(elided + next_mot)
                i += 2
                continue

        parts.append(mot)
        i += 1

    if not parts:
        return ""

    # Majuscule au premier mot
    first = parts[0]
    if first and first[0].islower():
        parts[0] = first[0].upper() + first[1:]

    # Joindre avec espaces
    texte = " ".join(parts)

    # Ponctuation finale
    if ponctuation_finale:
        texte += ponctuation_finale

    return texte
