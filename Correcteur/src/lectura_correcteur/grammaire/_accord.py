"""Regles d'accord : determinant+nom, determinant+adjectif+nom, etc.

Correspond aux regles 0, 1, 2, 4 du POC.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    INVARIABLES,
    PLUR_DET,
    SUJETS_3PL,
)


def verifier_accords(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Applique les regles d'accord sur la phrase.

    Regles :
    0. Restaurer "ils/elles" si corrige en "il/elle"
    1. Det. pluriel + NOM/ADJ -> ajouter -s si absent
    2. Det. pluriel + ADJ + NOM -> idem
    4. Det. pluriel + NOM/ADJ(s) + VER en -e -> -ent
    """
    if not mots:
        return mots, []

    origs = originaux if originaux else mots
    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    for i in range(n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]
        curr_low = curr.lower()

        # Regle 0 : Restaurer ils/elles si corrige en il/elle par erreur
        orig_low = origs[i].lower() if i < len(origs) else ""
        if orig_low in SUJETS_3PL and curr_low in ("il", "elle"):
            ancien = curr
            result[i] = transferer_casse(curr, origs[i])
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige=result[i],
                type_correction=TypeCorrection.GRAMMAIRE,
                explication="Restauration du pronom pluriel",
            ))
            curr = result[i]
            curr_low = curr.lower()

        # Regle 1 : Det. pluriel -> NOM/ADJ doit avoir -s
        if i > 0 and pos in ("NOM", "ADJ"):
            prev_low = result[i - 1].lower()
            if (
                prev_low in PLUR_DET
                and not curr_low.endswith(("s", "x", "z"))
                and len(curr) > 1
                and curr_low not in INVARIABLES
            ):
                candidate = curr + "s"
                if lexique is None or lexique.existe(candidate):
                    ancien = result[i]
                    result[i] = candidate
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=candidate,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication=f"Accord pluriel apres '{prev_low}'",
                    ))

        # Regle 2 : Det. pluriel + ADJ + NOM
        if (
            i > 1
            and pos == "NOM"
            and (pos_tags[i - 1] if i - 1 < len(pos_tags) else "") == "ADJ"
        ):
            prev2_low = result[i - 2].lower()
            if (
                prev2_low in PLUR_DET
                and not result[i].lower().endswith(("s", "x", "z"))
                and len(result[i]) > 1
                and result[i].lower() not in INVARIABLES
            ):
                candidate = result[i] + "s"
                if lexique is None or lexique.existe(candidate):
                    ancien = result[i]
                    result[i] = candidate
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=candidate,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication=f"Accord pluriel (det+adj+nom) apres '{prev2_low}'",
                    ))

        # Regle 4 : Det. pluriel + NOM/ADJ(s) + VER en -e -> -ent
        if i > 1 and pos in ("VER", "AUX"):
            prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            if prev_pos in ("NOM", "ADJ"):
                det_idx = i - 2
                while det_idx >= 0:
                    det_pos = pos_tags[det_idx] if det_idx < len(pos_tags) else ""
                    if det_pos in ("NOM", "ADJ"):
                        det_idx -= 1
                        continue
                    break
                if (
                    det_idx >= 0
                    and result[det_idx].lower() in PLUR_DET
                    and curr.endswith("e")
                    and not curr.endswith(("ent", "nt"))
                ):
                    candidate = curr + "nt"
                    if lexique is None or lexique.existe(candidate):
                        ancien = result[i]
                        result[i] = candidate
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=candidate,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="Accord sujet pluriel -> verbe -ent",
                        ))

    return result, corrections
