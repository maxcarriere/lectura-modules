"""Regles d'accord : determinant+nom, determinant+adjectif+nom, etc.

Correspond aux regles 0, 1, 2, 4 du POC.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    COPULES_ALL,
    COPULES_PLURIEL,
    DET_GENRE_MAP,
    INVARIABLES,
    PLUR_DET,
    PREPOSITIONS,
    SING_FEM_DET,
    SING_MASC_DET,
    SUJETS_3PL,
    generer_candidats_3pl,
    generer_candidats_feminin,
    generer_candidats_masculin,
    generer_candidats_pluriel,
    trouver_sujet_genre_nombre,
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
    1b. NOM pluriel + ADJ -> ajouter -s (adj post-nominal)
    2. Det. pluriel + ADJ + NOM -> idem
    4. Det. pluriel + NOM/ADJ(s) + VER -> 3pl (tous groupes)
    5. Copule plurielle + ADJ -> ajouter -s (attribut du sujet)
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

        # Regle 8 : Coherence DET↔NOM en genre (singulier)
        if (
            lexique is not None
            and curr_low in SING_MASC_DET | SING_FEM_DET
            and (pos.startswith("ART") or pos.startswith("DET"))
        ):
            # Trouver le NOM associe (i+1 ou i+2 si ADJ intercale)
            nom_idx = None
            adj_idx = None
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos == "NOM":
                    nom_idx = i + 1
                elif next_pos == "ADJ" and i + 2 < n:
                    adj_idx = i + 1
                    next2_pos = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                    if next2_pos == "NOM":
                        nom_idx = i + 2

            if nom_idx is not None:
                nom_infos = lexique.info(result[nom_idx])
                # Only consider NOM entries for genre (ignore ADJ homographs)
                nom_only = [e for e in nom_infos if e.get("cgram", "").startswith("NOM")]
                if not nom_only:
                    nom_only = nom_infos
                nom_genres = {e.get("genre") for e in nom_only if e.get("genre")}

                if curr_low in SING_MASC_DET and "f" in nom_genres and "m" not in nom_genres:
                    # DET masc + NOM fem → corriger le DET
                    new_det = DET_GENRE_MAP.get(curr_low)
                    if new_det:
                        ancien = result[i]
                        result[i] = transferer_casse(curr, new_det)
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=result[i],
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication=f"DET masc→fem (NOM '{result[nom_idx]}' est feminin)",
                        ))
                        curr = result[i]
                        curr_low = curr.lower()

                elif curr_low in SING_MASC_DET and "m" in nom_genres and adj_idx is not None:
                    # DET masc + ADJ fem + NOM masc → de-feminiser l'ADJ
                    adj_infos = lexique.info(result[adj_idx])
                    adj_genred = [e for e in adj_infos if e.get("genre")]
                    adj_est_fem = adj_genred and all(
                        e.get("genre") == "f" for e in adj_genred
                    )
                    if adj_est_fem:
                        for cand in generer_candidats_masculin(result[adj_idx]):
                            c_infos = lexique.info(cand)
                            if c_infos and any(
                                e.get("genre") == "m" for e in c_infos
                            ):
                                ancien = result[adj_idx]
                                result[adj_idx] = cand
                                corrections.append(Correction(
                                    index=adj_idx,
                                    original=ancien,
                                    corrige=cand,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    explication=f"ADJ fem→masc (DET+NOM '{result[nom_idx]}' sont masculins)",
                                ))
                                break

        # Regle 1 : Det. pluriel -> NOM/ADJ doit avoir -s/-x
        if i > 0 and pos in ("NOM", "ADJ"):
            prev_low = result[i - 1].lower()
            if (
                prev_low in PLUR_DET
                and not curr_low.endswith(("s", "x", "z"))
                and len(curr) > 1
                and curr_low not in INVARIABLES
            ):
                for candidate in generer_candidats_pluriel(curr):
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
                        break

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
                for candidate in generer_candidats_pluriel(result[i]):
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
                        break

        # Regle 7 : NOM feminin + ADJ masculin -> forme feminine
        # (avant Regle 1b pour que "vert" devienne "verte" avant "vertes")
        if i > 0 and pos == "ADJ" and lexique is not None:
            prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            if prev_pos == "NOM":
                prev_infos = lexique.info(result[i - 1])
                nom_est_fem = prev_infos and any(
                    e.get("genre") == "f" for e in prev_infos
                )
                if nom_est_fem:
                    adj_infos = lexique.info(result[i])
                    adj_genred = [e for e in adj_infos if e.get("genre")]
                    adj_est_masc = adj_genred and all(
                        e.get("genre") == "m" for e in adj_genred
                    )
                    if adj_est_masc:
                        for cand in generer_candidats_feminin(result[i]):
                            c_infos = lexique.info(cand)
                            if c_infos and any(
                                e.get("genre") == "f" for e in c_infos
                            ):
                                ancien = result[i]
                                result[i] = cand
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=cand,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    explication="Accord en genre NOM fem + ADJ",
                                ))
                                # Update curr_low for subsequent rules
                                curr = result[i]
                                curr_low = curr.lower()
                                break

        # Regle 1b : NOM pluriel + ADJ -> ajouter -s (adj post-nominal)
        if i > 1 and pos == "ADJ":
            prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            if prev_pos in ("NOM", "ADJ"):
                prev_low = result[i - 1].lower()
                if prev_low.endswith(("s", "x", "z")):
                    # Verifier qu'un det. pluriel existe avant
                    det_found = False
                    for j in range(i - 2, max(-1, i - 5), -1):
                        if result[j].lower() in PLUR_DET:
                            det_found = True
                            break
                    if (
                        det_found
                        and not curr_low.endswith(("s", "x", "z"))
                        and len(curr) > 1
                        and curr_low not in INVARIABLES
                    ):
                        for candidate in generer_candidats_pluriel(curr):
                            if lexique is None or lexique.existe(candidate):
                                ancien = result[i]
                                result[i] = candidate
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=candidate,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    explication="Accord pluriel adj post-nominal",
                                ))
                                break

        # Regle 4 : Sujet pluriel (eventuellement distant) + VER -> 3pl
        if i > 1 and pos in ("VER", "AUX"):
            if (
                not curr_low.endswith(("ent", "nt"))
                and _trouver_sujet_pluriel(result, pos_tags, i)
            ):
                candidats = generer_candidats_3pl(curr)
                for candidate in candidats:
                    if lexique is None or lexique.existe(candidate):
                        ancien = result[i]
                        result[i] = candidate
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=candidate,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="Accord sujet pluriel -> verbe 3pl",
                        ))
                        break

        # Regle 9 : Copule (etre/sembler/devenir...) + ADJ -> accord genre
        # avec le sujet. Applique AVANT Regle 5 (nombre) pour que
        # "elles sont petit" → petite (genre) → petites (nombre).
        if i > 0 and pos == "ADJ" and lexique is not None:
            prev_result_low = result[i - 1].lower()
            if prev_result_low in COPULES_ALL:
                # Trouver le genre du sujet
                gn = trouver_sujet_genre_nombre(
                    result, pos_tags, morpho, i - 1, lexique,
                )
                if gn is not None:
                    s_genre, _s_nombre = gn
                    adj_infos = lexique.info(result[i])
                    adj_genred = [e for e in adj_infos if e.get("genre")]
                    if s_genre == "Fem" and adj_genred and all(
                        e.get("genre") == "m" for e in adj_genred
                    ):
                        for cand in generer_candidats_feminin(result[i]):
                            c_infos = lexique.info(cand)
                            if c_infos and any(
                                e.get("genre") == "f" for e in c_infos
                            ):
                                ancien = result[i]
                                result[i] = cand
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=cand,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    explication="Accord attribut en genre (sujet fem)",
                                ))
                                curr = result[i]
                                curr_low = curr.lower()
                                break
                    elif s_genre == "Masc" and adj_genred and all(
                        e.get("genre") == "f" for e in adj_genred
                    ):
                        for cand in generer_candidats_masculin(result[i]):
                            c_infos = lexique.info(cand)
                            if c_infos and any(
                                e.get("genre") == "m" for e in c_infos
                            ):
                                ancien = result[i]
                                result[i] = cand
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=cand,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    explication="Accord attribut en genre (sujet masc)",
                                ))
                                curr = result[i]
                                curr_low = curr.lower()
                                break

        # Regle 5 : Copule plurielle + ADJ -> ajouter -s/-x (attribut du sujet)
        if i > 0 and pos == "ADJ":
            prev_result_low = result[i - 1].lower()
            if (
                prev_result_low in COPULES_PLURIEL
                and not curr_low.endswith(("s", "x", "z"))
                and len(curr) > 1
            ):
                for candidate in generer_candidats_pluriel(curr):
                    if lexique is None or lexique.existe(candidate):
                        ancien = result[i]
                        result[i] = candidate
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=candidate,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="Accord attribut apres copule plurielle",
                        ))
                        break

        # Regle 6 : DET feminin + NOM/ADJ masculin -> forme feminine
        if i > 0 and pos in ("NOM", "ADJ"):
            prev_low = result[i - 1].lower()
            if prev_low in SING_FEM_DET and lexique is not None:
                infos = lexique.info(result[i])
                genred = [e for e in infos if e.get("genre")]
                if genred and all(e.get("genre") == "m" for e in genred):
                    for candidate in generer_candidats_feminin(result[i]):
                        c_infos = lexique.info(candidate)
                        if c_infos and any(e.get("genre") == "f" for e in c_infos):
                            ancien = result[i]
                            result[i] = candidate
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige=candidate,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                explication=f"Accord en genre apres '{prev_low}'",
                            ))
                            break

    return result, corrections


def _trouver_sujet_pluriel(
    mots: list[str], pos_tags: list[str], idx_verbe: int,
) -> bool:
    """Cherche un determinant pluriel avant le verbe en sautant les complements.

    Saute les groupes prepositionnels PRE (+DET) (+NOM/ADJ).
    En balayant vers la gauche, l'ordre est NOM/ADJ ← DET ← PRE, donc
    quand on rencontre un DET singulier, on verifie si une preposition suit.
    """
    j = idx_verbe - 1
    while j >= 0:
        pos_j = pos_tags[j] if j < len(pos_tags) else ""
        mot_j = mots[j].lower()
        if pos_j in ("NOM", "ADJ"):
            j -= 1
            continue
        # Contractions prepositionnelles : sauter du/au
        if mot_j in ("du", "au"):
            j -= 1
            continue
        if mot_j in PLUR_DET:
            # Verifier que ce n'est pas un DET dans un PP
            # (ex: "pres des maisons" — "des" est DET pluriel dans PP)
            if j > 0:
                prev_pos = pos_tags[j - 1] if j - 1 < len(pos_tags) else ""
                prev_mot = mots[j - 1].lower()
                if prev_pos == "PRE" or prev_mot in PREPOSITIONS:
                    j -= 2  # sauter DET + PRE
                    continue
            return True
        if pos_j.startswith("ART") or pos_j.startswith("DET"):
            # DET/ART singulier : verifier s'il est precede d'une preposition
            if j > 0:
                prev_pos = pos_tags[j - 1] if j - 1 < len(pos_tags) else ""
                prev_mot = mots[j - 1].lower()
                if prev_pos == "PRE" or prev_mot in PREPOSITIONS:
                    j -= 2  # sauter DET + PRE (complement prepositionnel)
                    continue
            return False
        if pos_j == "PRE" or mot_j in PREPOSITIONS:
            j -= 1
            continue
        break
    return False
