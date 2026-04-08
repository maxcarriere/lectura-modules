"""Desambiguation contextuelle des homophones grammaticaux.

Paires traitees : et/est, son/sont, a/à, ou/où, on/ont, ce/se, la/là,
leur/leurs, ça/sa, -er/-é apres aller.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur.grammaire._donnees import ALLER_FORMES

# Sujets pronominaux 3e personne du singulier
_SUJETS_3SG = frozenset({"il", "elle", "on", "qui", "ce", "c'"})

# Pronoms objets (entre sujet et verbe, indiquent que "a" est auxiliaire)
_PRONOMS_OBJETS = frozenset({
    "me", "m'", "te", "t'", "se", "s'", "le", "la", "l'",
    "lui", "nous", "vous", "leur", "en", "y",
})

# Terminaisons de participe passe (pour heuristique morpho sans POS)
_PP_SUFFIXES = ("é", "és", "ée", "ées", "i", "is", "ie", "ies",
                "u", "us", "ue", "ues", "it", "ite", "ites",
                "ert", "erte", "ertes", "erts", "oint", "ointe",
                "eint", "einte", "aint", "ainte", "ort", "orte")


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

        # "et" suivi de VER ou mot en forme de PP, quand precede d'un sujet
        # "il et parti" -> "il est parti"
        if curr_low == "et" and pos == "CON":
            if i > 0 and i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                next_looks_pp = next_pos == "VER" or next_low.endswith(_PP_SUFFIXES)
                if next_looks_pp:
                    prev_low = result[i - 1].lower()
                    prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                    if prev_pos in ("PRO:per", "PRO:dem", "PRO:rel") or prev_low in (
                        "il", "elle", "on", "ils", "elles", "je", "tu",
                        "nous", "vous", "qui", "ce", "c'",
                    ):
                        ancien = result[i]
                        result[i] = "est"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="est",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="'et' -> 'est' (sujet + _ + participe)",
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
        # "son" + VER/PP -> probablement "sont" (quand sujet pluriel avant)
        if curr_low == "son" and pos in ("ADJ:pos", "ADJ"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                _next_verb = next_pos in ("VER", "AUX") or next_low.endswith(_PP_SUFFIXES)
                if _next_verb:
                    # Verifier qu'un sujet pluriel precede
                    _plur_subj = False
                    for _k in range(i - 1, max(-1, i - 4), -1):
                        _w = result[_k].lower()
                        if _w in ("ils", "elles"):
                            _plur_subj = True
                            break
                        _pk = pos_tags[_k] if _k < len(pos_tags) else ""
                        if _pk == "NOM" and _w.endswith(("s", "x", "z")):
                            _plur_subj = True
                            break
                        if _pk not in ("ADV", "PRE", "ADJ:pos", "ADJ"):
                            break
                    if _plur_subj and lexique is not None and lexique.existe("sont"):
                        ancien = result[i]
                        result[i] = "sont"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="sont",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="'son' -> 'sont' (sujet pluriel + participe)",
                        ))
                        continue

        # "sont" etiquete ADJ:pos -> "son" (possessif)
        if curr_low == "sont" and pos in ("ADJ:pos",):
            ancien = result[i]
            result[i] = "son"
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige="son",
                type_correction=TypeCorrection.GRAMMAIRE,
                explication="'sont' -> 'son' (possessif)",
            ))
            continue

        # "sont" apres PRE + suivi de NOM -> "son" (possessif)
        # "avec sont velo" -> "avec son vélo"
        if curr_low == "sont" and pos in ("VER", "AUX"):
            if i > 0 and i + 1 < n:
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if prev_pos == "PRE" and next_pos == "NOM":
                    ancien = result[i]
                    result[i] = "son"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="son",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="'sont' -> 'son' (PRE + possessif + NOM)",
                    ))
                    continue

        # --- a / à ---
        if curr_low == "a" and pos in ("VER", "AUX"):
            # Chercher si "a" est probablement l'auxiliaire avoir (3sg)
            _is_aux = False
            for _k in range(i - 1, max(-1, i - 4), -1):
                _w = result[_k].lower()
                # Pronom sujet 3sg -> "a" = auxiliaire avoir
                if _w in _SUJETS_3SG:
                    _is_aux = True
                    break
                # Pronom objet (me/te/nous/vous/lui/...) -> "a" = auxiliaire
                if _w in _PRONOMS_OBJETS:
                    _is_aux = True
                    break
                _pk = pos_tags[_k] if _k < len(pos_tags) else ""
                # NOM directement avant "a" : verifier si le NOM est sujet
                # (vs objet d'un verbe plus tot dans la phrase)
                if _pk == "NOM":
                    # Chercher s'il y a un VER/AUX avant ce NOM -> NOM = objet
                    _nom_is_subject = True
                    for _j in range(_k - 1, max(-1, _k - 4), -1):
                        _pj = pos_tags[_j] if _j < len(pos_tags) else ""
                        if _pj in ("VER", "AUX"):
                            _nom_is_subject = False
                            break
                        if _pj in ("PRE", "CON"):
                            break
                    if _nom_is_subject:
                        _is_aux = True
                    break
                if _pk in ("VER", "AUX", "ADJ"):
                    break
            if not _is_aux and i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower() if i + 1 < n else ""
                # a + ART, a + PRE, a + elision (l', d'), a + VER (infinitif)
                if next_pos in (
                    "VER", "ART:def", "ART:ind", "PRE",
                ) or next_low.endswith(("'", "\u2019")):
                    ancien = result[i]
                    result[i] = "à"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="à",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="'a' -> 'à' (preposition)",
                    ))
                    continue

        # "à" devant VER quand sujet pronominal precede -> "a" (auxiliaire avoir)
        # "elle à mange" -> "elle a mange"
        if curr_low == "à" and i + 1 < n:
            next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
            next_low = result[i + 1].lower()
            _next_is_verb = next_pos == "VER" or next_low.endswith(_PP_SUFFIXES)
            if _next_is_verb:
                for _k in range(i - 1, max(-1, i - 3), -1):
                    _w = result[_k].lower()
                    if _w in ("il", "elle", "on", "ils", "elles",
                              "je", "j'", "tu", "nous", "vous", "qui"):
                        ancien = result[i]
                        result[i] = "a"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="a",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="'à' -> 'a' (auxiliaire avoir apres sujet)",
                        ))
                        break
                else:
                    _next_is_verb = False  # reset flag
                if result[i] == "a":
                    continue

        # "à" tague VER/AUX -> "a" (auxiliaire avoir)
        if curr_low == "à" and pos in ("VER", "AUX"):
            ancien = result[i]
            result[i] = "a"
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige="a",
                type_correction=TypeCorrection.GRAMMAIRE,
                explication="'à' -> 'a' (auxiliaire avoir)",
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

        # "ou" etiquete ADV -> "où" (adverbe interrogatif)
        if curr_low == "ou" and pos == "ADV":
            ancien = result[i]
            result[i] = "où"
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige="où",
                type_correction=TypeCorrection.GRAMMAIRE,
                explication="'ou' -> 'où' (adverbe interrogatif)",
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

        # "ont" etiquete PRO -> "on" (pronom indefini)
        if curr_low == "ont" and pos in ("PRO:ind", "PRO:per"):
            ancien = result[i]
            result[i] = "on"
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige="on",
                type_correction=TypeCorrection.GRAMMAIRE,
                explication="'ont' -> 'on' (pronom indefini)",
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

        # --- leur / leurs ---
        # "leur" + NOM pluriel -> "leurs"
        if curr_low == "leur" and pos in ("DET", "ADJ:pos"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos == "NOM":
                    next_low = result[i + 1].lower()
                    if next_low.endswith(("s", "x", "z")):
                        ancien = result[i]
                        result[i] = "leurs"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="leurs",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="'leur' -> 'leurs' (NOM pluriel)",
                        ))
                        continue

        # "leurs" + NOM singulier -> "leur"
        if curr_low == "leurs" and pos in ("DET", "ADJ:pos"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos == "NOM":
                    next_low = result[i + 1].lower()
                    if not next_low.endswith(("s", "x", "z")):
                        ancien = result[i]
                        result[i] = "leur"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="leur",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            explication="'leurs' -> 'leur' (NOM singulier)",
                        ))
                        continue

        # --- ça / sa ---
        # "sa" + VER -> probablement "ça"
        if curr_low == "sa" and pos in ("ADJ:pos", "DET"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos in ("VER", "AUX"):
                    ancien = result[i]
                    result[i] = "ça"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="ça",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="'sa' -> 'ça' (suivi d'un verbe)",
                    ))
                    continue

        # "ça" + NOM -> probablement "sa"
        if curr_low in ("ça", "ca") and pos in ("PRO:dem", "PRO:ind"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos == "NOM":
                    ancien = result[i]
                    result[i] = "sa"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="sa",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="'ça' -> 'sa' (suivi d'un nom)",
                    ))
                    continue

        # --- -er/-é apres aller ---
        # ALLER_FORMES + mot en -é -> -er (infinitif requis apres aller)
        if i > 0 and pos == "VER" and curr_low.endswith("é") and not curr_low.endswith("er"):
            prev_low = result[i - 1].lower()
            if prev_low in ALLER_FORMES:
                # Generer l'infinitif : mangé → manger (remplacer é par er)
                candidate = curr_low[:-1] + "er"
                if lexique is None or lexique.existe(candidate):
                    ancien = result[i]
                    result[i] = candidate
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=candidate,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        explication="PP -> infinitif apres aller",
                    ))
                    continue

    return result, corrections
