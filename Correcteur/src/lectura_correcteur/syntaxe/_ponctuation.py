"""Regles typographiques francaises pour la ponctuation."""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection


def verifier_ponctuation(tokens: list[str]) -> list[Correction]:
    """Verifie les regles typographiques sur les tokens.

    Capitalisation apres un point. Modifie les tokens en place.
    """
    corrections: list[Correction] = []
    n = len(tokens)

    for i in range(n):
        tok = tokens[i]

        if i == 0 or (i > 0 and tokens[i - 1] in (".", "!", "?", "\u2026")):
            if tok and tok[0].islower() and tok.isalpha():
                ancien = tok
                tokens[i] = tok[0].upper() + tok[1:]
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=tokens[i],
                    type_correction=TypeCorrection.SYNTAXE,
                    regle="syntaxe.majuscule",
                    explication="Majuscule en debut de phrase",
                ))

    # Point final automatique — desactive (FP sur corpus Wikipedia)
    # if tokens and tokens[-1] not in (".", "!", "?", "\u2026"):
    #     tokens.append(".")
    #     ...

    return corrections
