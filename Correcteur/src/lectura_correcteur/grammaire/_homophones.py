"""Desambiguation contextuelle des homophones grammaticaux.

Paires traitees : et/est, son/sont.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection


def verifier_homophones(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Desambigue les homophones grammaticaux par le contexte.

    Returns:
        Tuple (mots_corriges, corrections).
    """
    if not mots:
        return mots, []

    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    for i in range(n):
        curr_low = result[i].lower()
        pos = pos_tags[i] if i < len(pos_tags) else ""

        # --- et / est ---
        # "et" devant un ADJ/ADV -> probablement "est"
        if curr_low == "et" and pos == "CON":
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos in ("ADJ", "ADV"):
                    if i > 0:
                        prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                        if prev_pos in ("PRO:per", "NOM", "PRO:dem", "PRO:rel"):
                            ancien = result[i]
                            result[i] = "est"
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige="est",
                                type_correction=TypeCorrection.GRAMMAIRE,
                                explication="'et' -> 'est' (sujet + _ + adjectif)",
                            ))
                            continue

        # "est" comme CON -> probablement "et"
        if curr_low == "est" and pos == "CON":
            ancien = result[i]
            result[i] = "et"
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige="et",
                type_correction=TypeCorrection.GRAMMAIRE,
                explication="'est' -> 'et' (conjonction attendue)",
            ))
            continue

        # --- son / sont ---
        # "son" + VER -> probablement "sont"
        if curr_low == "son" and pos in ("ADJ:pos", "ADJ"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos in ("VER", "AUX"):
                    if lexique is not None and lexique.existe("sont"):
                        ancien = result[i]
                        result[i] = "sont"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="sont",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="'son' -> 'sont' (suivi d'un verbe)",
                        ))
                        continue

    return result, corrections
