"""Reconstruction du texte a partir des mots ortho + ponctuation.

Gere la majuscule initiale, les elisions (l', d', j', ...) et la ponctuation.

Deux niveaux :
  - ``assembler_texte``    : reconstruction simple (ortho-only)
  - ``rejoin_elisions``    : reconstruction avancee avec contexte IPA
    (clitiques, elisions, mots composes)

Copyright (C) 2025-2026 Max Carriere
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


# ── Mapping IPA clitique → forme apostrophe / forme pleine ──────

_IPA_CLITIC_MAP: dict[str, str] = {
    "l": "l'", "d": "d'", "n": "n'", "s": "s'",
    "ʒ": "j'", "k": "qu'", "m": "m'", "t": "t'",
}

_IPA_CLITIC_FULL: dict[str, str] = {
    "l": "le", "d": "de", "n": "ne", "s": "se",
    "ʒ": "je", "k": "que", "m": "me", "t": "te",
}

# Voyelles IPA (pour detecter si le mot suivant commence par voyelle)
_IPA_VOWELS = frozenset("aeiouyøœəɛɔɑɥ")


def _starts_with_ipa_vowel(ipa_word: str) -> bool:
    """Verifie si un mot IPA commence par une voyelle."""
    if not ipa_word:
        return False
    return ipa_word[0] in _IPA_VOWELS


def rejoin_elisions(
    ortho_words: list[str],
    ipa_words: list[str],
    compound_joins: set[int] | None = None,
) -> str:
    """Recombine les mots graphemiques en gerant elisions et composes.

    Utilise le contexte IPA pour decider du traitement des clitiques :
      - Clitique + mot IPA commencant par voyelle → apostrophe (l'abattoir)
      - Clitique + mot IPA commencant par consonne → forme pleine (le chat)
      - Composes (compound_joins) → mots joints par tiret (grand-pere)

    Parameters
    ----------
    ortho_words : list[str]
        Mots orthographiques (sortie P2G).
    ipa_words : list[str]
        Mots IPA correspondants (pour contexte elision).
    compound_joins : set[int] | None
        Indices i tels que ortho_words[i] et ortho_words[i+1] doivent
        etre joints par un tiret.

    Returns
    -------
    str
        Texte reconstruit.
    """
    if compound_joins is None:
        compound_joins = set()

    parts: list[str] = []
    i = 0
    while i < len(ortho_words):
        ipa = ipa_words[i] if i < len(ipa_words) else ""
        ortho = ortho_words[i]

        if ipa in _IPA_CLITIC_MAP and i + 1 < len(ortho_words):
            next_ipa = ipa_words[i + 1] if i + 1 < len(ipa_words) else ""
            if _starts_with_ipa_vowel(next_ipa):
                # Elision : l'abattoir, d'abord, j'ai
                if ortho.endswith("'"):
                    parts.append(ortho + ortho_words[i + 1])
                else:
                    parts.append(_IPA_CLITIC_MAP[ipa] + ortho_words[i + 1])
                i += 2
                continue
            else:
                # Devant consonne : forme pleine (le, de, je...)
                if ortho.endswith("'"):
                    parts.append(_IPA_CLITIC_FULL.get(ipa, ortho))
                elif len(ortho) > 1:
                    parts.append(ortho)
                else:
                    parts.append(_IPA_CLITIC_FULL.get(ipa, ortho))
                if i in compound_joins:
                    parts[-1] = parts[-1] + "-"
                i += 1
                continue

        parts.append(ortho)
        if i in compound_joins:
            parts[-1] = parts[-1] + "-"
        i += 1

    # Fusionner les parties tiretees
    final: list[str] = []
    for p in parts:
        if final and final[-1].endswith("-"):
            final[-1] = final[-1] + p
        else:
            final.append(p)

    # Filtrer les mots vides
    final = [f for f in final if f]

    if not final:
        return ""

    # Majuscule au premier mot
    first = final[0]
    if first and first[0].islower():
        final[0] = first[0].upper() + first[1:]

    return " ".join(final)
