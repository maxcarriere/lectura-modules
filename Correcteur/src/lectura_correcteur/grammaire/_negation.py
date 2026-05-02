"""Regles de negation : verbe + pas/plus/jamais sans ne -> inserer ne."""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection

_NEGATIFS = frozenset({"pas", "plus", "jamais", "rien", "personne"})

_VOYELLES = frozenset("aeiouyàâäéèêëïîôùûüæœ")


def verifier_negation(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Insere 'ne' quand il manque dans une negation.

    Pattern : verbe + pas/plus/jamais/rien/personne sans ne avant le verbe.
    Ex: "je mange pas" -> "je ne mange pas"

    Returns the word list with insertions applied, and Corrections
    with original="" for each insertion (index is in the output list).
    Processes in reverse order to preserve indices.
    """
    if not mots:
        return mots, []

    result = list(mots)
    pos = list(pos_tags)
    corrections: list[Correction] = []
    n = len(result)

    # Collecter les positions d'insertion
    insertions: list[int] = []

    for i in range(n):
        mot_negatif = result[i].lower()
        if mot_negatif not in _NEGATIFS:
            continue

        # Guard: "plus" comparatif/superlatif — ne PAS inserer "ne"
        if mot_negatif == "plus":
            if i + 1 < n:
                next_pos = pos[i + 1] if i + 1 < len(pos) else ""
                next_word = result[i + 1].lower() if i + 1 < n else ""
                # plus + ADJ/ADV/que → comparatif/superlatif, pas negation
                if next_pos in ("ADJ", "ADV") or next_word == "que":
                    continue

        # Chercher le verbe juste avant (i-1 ou i-2 si pronom intercale)
        verbe_idx = None
        if i >= 1:
            p = pos[i - 1] if i - 1 < len(pos) else ""
            if p in ("VER", "AUX"):
                verbe_idx = i - 1
            elif i >= 2:
                p2 = pos[i - 2] if i - 2 < len(pos) else ""
                if p2 in ("VER", "AUX"):
                    verbe_idx = i - 2

        if verbe_idx is None:
            continue

        # Verifier qu'il n'y a pas deja "ne"/"n'" avant le verbe
        deja_ne = False
        if verbe_idx >= 1:
            avant = result[verbe_idx - 1].lower()
            if avant in ("ne", "n'", "n\u2019"):
                deja_ne = True

        if not deja_ne:
            insertions.append(verbe_idx)

    # Inserer en ordre inverse pour preserver les indices
    for verbe_idx in reversed(insertions):
        verbe = result[verbe_idx]
        if verbe and verbe[0].lower() in (_VOYELLES | {"h"}):
            # Elision : ne + aime → n'aime (modifier le verbe en place)
            result[verbe_idx] = "n'" + verbe
            corrections.append(Correction(
                index=verbe_idx,
                original=verbe,
                corrige="n'" + verbe,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="negation.elision",
                explication="Negation incomplete -> elision n'",
            ))
        else:
            result.insert(verbe_idx, "ne")
            pos.insert(verbe_idx, "ADV")
            corrections.append(Correction(
                index=verbe_idx,
                original="",
                corrige="ne",
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="negation.insertion",
                explication="Negation incomplete -> insertion de 'ne'",
            ))

    return result, corrections
