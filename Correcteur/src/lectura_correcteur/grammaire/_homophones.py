"""Desambiguation contextuelle des homophones grammaticaux.

Paires traitees : et/est, son/sont, a/à, ou/où, on/ont, ce/se, la/là.
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

        # "est" (VER/AUX) suivi d'un DET ou PRO -> probablement "et" (coordination)
        if curr_low == "est" and pos in ("VER", "AUX"):
            if i > 0 and i + 1 < n:
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if (
                    prev_pos in ("NOM", "ADJ", "ADJ:pos")
                    and next_pos in (
                        "ART:def", "ART:ind", "DET", "PRO:per", "ADJ:pos",
                    )
                ):
                    ancien = result[i]
                    result[i] = "et"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="et",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="'est' -> 'et' (coordination NOM + DET/PRO)",
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

        # --- a / à ---
        # "a" (VER/AUX) sans sujet 3sg avant + suivi de VER -> "à"
        if curr_low == "a" and pos in ("VER", "AUX"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos == "VER" and i > 0:
                    prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                    if prev_pos not in ("PRO:per", "NOM", "PRO:dem"):
                        ancien = result[i]
                        result[i] = "à"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="à",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="'a' -> 'à' (preposition devant verbe)",
                        ))
                        continue

        # --- ou / où ---
        # "ou" etiquete PRO:rel -> "où" (pronom relatif lieu/temps)
        if curr_low == "ou" and pos == "PRO:rel":
            ancien = result[i]
            result[i] = "où"
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige="où",
                type_correction=TypeCorrection.GRAMMAIRE,
                explication="'ou' -> 'où' (pronom relatif)",
            ))
            continue

        # --- on / ont ---
        # "on" apres NOM/PRO + suivi de VER/ADV -> "ont" (3pl avoir)
        if curr_low == "on" and pos in ("PRO:ind", "PRO:per"):
            if i > 0 and i + 1 < n:
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if prev_pos in ("NOM", "PRO:per") and next_pos in ("VER", "ADV"):
                    ancien = result[i]
                    result[i] = "ont"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="ont",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="'on' -> 'ont' (auxiliaire 3pl)",
                    ))
                    continue

        # --- ce / se ---
        # "ce" (DET:dem) + VER/AUX -> "se" (pronom reflexif)
        if curr_low == "ce" and pos.startswith("DET"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos in ("VER", "AUX"):
                    ancien = result[i]
                    result[i] = "se"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="se",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="'ce' -> 'se' (pronom reflexif devant verbe)",
                    ))
                    continue

        # "se" (PRO:per) + NOM -> "ce" (determinant)
        if curr_low == "se" and pos == "PRO:per":
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos == "NOM":
                    ancien = result[i]
                    result[i] = "ce"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="ce",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="'se' -> 'ce' (determinant devant nom)",
                    ))
                    continue

        # --- la / là ---
        # "la" etiquete ADV -> "là" (adverbe de lieu)
        if curr_low == "la" and pos == "ADV":
            ancien = result[i]
            result[i] = "là"
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige="là",
                type_correction=TypeCorrection.GRAMMAIRE,
                explication="'la' -> 'là' (adverbe de lieu)",
            ))
            continue

    return result, corrections
