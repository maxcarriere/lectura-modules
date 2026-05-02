"""Desambiguation contextuelle des homophones grammaticaux.

Paires traitees : et/est, son/sont, a/à, ou/où, on/ont, ce/se, la/là,
leur/leurs, ça/sa, -er/-é apres aller.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur.grammaire._donnees import (
    ALLER_FORMES,
    PLUR_DET,
    PREPOSITIONS,
    SING_DET,
    generer_candidats_pluriel,
)

# Sujets pronominaux 3e personne du singulier
_SUJETS_3SG = frozenset({"il", "elle", "on", "qui", "ce", "c'"})

# Pronoms objets (entre sujet et verbe, indiquent que "a" est auxiliaire)
_PRONOMS_OBJETS = frozenset({
    "me", "m'", "te", "t'", "se", "s'", "le", "la", "l'", "les",
    "lui", "nous", "vous", "leur", "en", "y",
})

# Terminaisons de participe passe (pour heuristique morpho sans POS)
_PP_SUFFIXES = ("é", "és", "ée", "ées", "i", "is", "ie", "ies",
                "u", "us", "ue", "ues", "it", "its", "ite", "ites",
                "ert", "erte", "ertes", "erts", "oint", "oints", "ointe",
                "eint", "eints", "einte", "aint", "aints", "ainte",
                "ort", "orts", "orte")

# Present-tense-only -oit/-ait verbs (no PP entry)
# These match _PP_SUFFIXES but are NOT past participles
_PRESENT_ONLY_IT = frozenset({
    "voit", "boit", "doit", "reçoit", "conçoit", "aperçoit",
    "croit", "croît", "plaît", "paraît", "apparaît", "disparaît",
    "connaît", "reconnaît", "naît", "sait",
    "revoit", "entrevoit", "prévoit", "pourvoit",
})


def verifier_homophones(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
    *,
    pos_confiance: list[float] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Desambigue les homophones grammaticaux par le contexte.

    Args:
        pos_confiance: Confiance POS par position (optionnel).
            Si fourni, certaines regles skipent quand la confiance
            est trop faible pour le POS decisive.

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
        _conf = pos_confiance[i] if pos_confiance and i < len(pos_confiance) else 1.0

        # --- et / est ---
        # Guard: capitalized "Est" mid-sentence = cardinal direction, not verb
        if (
            curr_low == "est"
            and result[i] == "Est"
            and i > 0
            and result[i - 1] not in (".", "!", "?", ";", ":")
        ):
            continue

        # "n et pas" → "n'est pas" (orphan elision of "ne")
        if curr_low == "et" and pos == "CON":
            if i > 0 and i + 1 < n:
                _prev_n = result[i - 1].lower()
                _next_pas = result[i + 1].lower()
                if _prev_n == "n" and _next_pas == "pas":
                    ancien = result[i]
                    result[i] = "est"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="est",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.et_est",
                        explication="'et' -> 'est' (n'est pas)",
                    ))
                    continue

        # sujet + "et" + preposition (en, au, à, ...) → "est" (copule)
        # "on et en retard" → "on est en retard"
        # "il et à la maison" → "il est à la maison"
        # Guard: require subject pronoun or singular NOM before
        if curr_low == "et" and pos == "CON":
            if i > 0 and i + 1 < n:
                _prev_pos_et = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _prev_low_et = result[i - 1].lower()
                _next_pos_et = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _next_low_et = result[i + 1].lower()
                _prev_is_subj = (
                    _prev_low_et in (
                        "il", "elle", "on", "je", "j'", "tu",
                        "nous", "vous", "ce", "c'", "tout",
                    )
                    or _prev_pos_et in ("PRO:per", "PRO:dem")
                )
                _next_is_prep_copule = (
                    _next_pos_et == "PRE"
                    and _next_low_et in (
                        "en", "au", "aux",
                        "à", "a",
                    )
                )
                if _prev_is_subj and _next_is_prep_copule:
                    ancien = result[i]
                    result[i] = "est"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="est",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.et_est",
                        explication="'et' -> 'est' (copule + preposition)",
                    ))
                    continue

        # "et" devant un ADJ/ADV -> probablement "est"
        if curr_low == "et" and pos == "CON":
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low_b1 = result[i + 1].lower()
                # Guard: single-letter next word = orphan elision, skip
                if next_pos in ("ADJ", "ADV") and len(next_low_b1) > 1:
                    if i > 0:
                        prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                        if prev_pos in ("PRO:per", "PRO:dem", "PRO:rel"):
                            ancien = result[i]
                            result[i] = "est"
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige="est",
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="homophone.et_est",
                                explication="'et' -> 'est' (sujet + _ + adjectif)",
                            ))
                            continue
                        # NOM / NOM PROPRE / OOV + et + ADJ/ADV = copule
                        # Coordination ADVs: "et même X" = "and even X"
                        _COORD_ADVS = (
                            "même", "meme", "surtout", "voire",
                            "notamment", "aussi", "également", "puis",
                            "dernièrement", "récemment", "principalement",
                            "particulièrement", "spécialement",
                        )
                        _can_correct_b1 = False
                        if (
                            prev_pos == "NOM PROPRE"
                            and next_pos == "ADV"
                            and next_low_b1 not in _COORD_ADVS
                        ):
                            # Proper nouns don't need DET; check PRE before name
                            _in_pp_np = False
                            if i > 1:
                                prev2_low = result[i - 2].lower()
                                _p2pos = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                                if (
                                    _p2pos == "PRE"
                                    or prev2_low in PREPOSITIONS
                                    or prev2_low in ("du", "des", "aux")
                                ):
                                    _in_pp_np = True
                            if not _in_pp_np:
                                _can_correct_b1 = True
                        elif prev_pos == "NOM" and i > 1:
                            # Regular NOM: need DET at i-2
                            prev2_low = result[i - 2].lower()
                            if prev2_low in SING_DET:
                                _in_pp = False
                                if i > 2:
                                    _prev3_pos = pos_tags[i - 3] if i - 3 < len(pos_tags) else ""
                                    _prev3_low = result[i - 3].lower()
                                    if _prev3_pos == "PRE" or _prev3_low in PREPOSITIONS:
                                        _in_pp = True
                                if prev2_low in ("du", "des"):
                                    _in_pp = True
                                if not _in_pp:
                                    _can_correct_b1 = True
                        elif (
                            prev_pos == ""
                            and next_pos == "ADV"
                            and len(result[i - 1]) >= 3
                            and next_low_b1 not in _COORD_ADVS
                        ):
                            # OOV word (likely proper noun) + et + ADV
                            _in_pp_oov = False
                            if i > 1:
                                prev2_low = result[i - 2].lower()
                                _p2pos = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                                if (
                                    _p2pos == "PRE"
                                    or prev2_low in PREPOSITIONS
                                    or prev2_low in ("du", "des")
                                ):
                                    _in_pp_oov = True
                            if not _in_pp_oov:
                                _can_correct_b1 = True
                        if _can_correct_b1:
                            ancien = result[i]
                            result[i] = "est"
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige="est",
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="homophone.et_est",
                                explication="'et' -> 'est' (NOM singulier + _ + adjectif)",
                            ))
                            continue

        # "et" suivi de VER ou mot en forme de PP, quand precede d'un sujet
        # ou d'un NOM/NOM PROPRE (passif: "le tournage et arrete" -> "est arrete")
        # "il et parti" -> "il est parti"
        if curr_low == "et" and pos == "CON":
            if i > 0 and i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                next_looks_pp = next_pos == "VER" or (
                    len(next_low) >= 4 and next_low.endswith(_PP_SUFFIXES)
                ) or next_low in ("né", "née", "nés", "nées")
                # Pour NOM/NOM PROPRE, double garde: VER tag + suffixe 1er groupe
                # pour eviter FP sur "chaussées"(NOM), "saint"("aint"),
                # "glorifies"(VER present, pas PP)
                _next_nom_safe_pp = (
                    next_pos in ("VER", "AUX")
                    and len(next_low) >= 4
                    and next_low.endswith(
                        ("\u00e9", "\u00e9s", "\u00e9e", "\u00e9es"))
                ) or next_low in ("né", "née", "nés", "nées")
                if next_looks_pp:
                    prev_low = result[i - 1].lower()
                    prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                    _SUBJ_PRONOUNS = frozenset({
                        "il", "elle", "on", "ils", "elles", "je", "tu",
                        "nous", "vous", "qui", "ce", "c'",
                    })
                    _is_pronoun_subj = (
                        prev_pos in ("PRO:per", "PRO:dem", "PRO:rel")
                        or prev_low in _SUBJ_PRONOUNS
                    )
                    # Guard: PRO:per that is NOT a subject pronoun
                    # (me, te, le, la, lui, leur, en) followed by
                    # a non-PP verb form → not "est" context.
                    # Exception: "y" is transparent — "y est" is dominant
                    # pattern ("il y est", "cela y est", "NOM y est PP").
                    if (
                        _is_pronoun_subj
                        and prev_low not in _SUBJ_PRONOUNS
                        and prev_low != "y"
                        and prev_pos == "PRO:per"
                        and lexique is not None
                        and hasattr(lexique, "info")
                    ):
                        _next_infos_subj = lexique.info(next_low)
                        _next_is_pp = any(
                            e.get("cgram") in ("VER", "AUX")
                            and e.get("mode") in ("participe", "par")
                            and e.get("temps") in ("passé", "pas", "past")
                            for e in _next_infos_subj
                        ) if _next_infos_subj else False
                        if not _next_is_pp:
                            _is_pronoun_subj = False
                    _is_oov_subj_b2 = (
                        prev_pos == "" and len(prev_low) >= 3
                    )
                    _is_nom_subj = (
                        (
                            prev_pos in ("NOM", "NOM PROPRE", "ADJ")
                            or _is_oov_subj_b2
                        )
                        and _next_nom_safe_pp
                    )
                    # Guard: NOM in PP is complement, not subject
                    if _is_nom_subj and i > 1:
                        _pp2_pos = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                        _pp2_low = result[i - 2].lower()
                        if _pp2_pos == "PRE" or _pp2_low in PREPOSITIONS:
                            _is_nom_subj = False
                        # "du/des" = prepositional contraction
                        elif _pp2_low in ("du", "des"):
                            _is_nom_subj = False
                        # "aux" = prepositional contraction (à + les)
                        elif _pp2_low in ("aux",):
                            _is_nom_subj = False
                        # ART/DET at i-2, PRE at i-3 → "de la communauté"
                        elif (
                            _pp2_pos.startswith(("ART", "DET"))
                            and i > 2
                        ):
                            _pp3_pos = pos_tags[i - 3] if i - 3 < len(pos_tags) else ""
                            _pp3_low = result[i - 3].lower()
                            if (_pp3_pos == "PRE"
                                    or _pp3_low in PREPOSITIONS
                                    or _pp3_low in ("du", "des", "aux")):
                                _is_nom_subj = False
                        # NOM PROPRE chain: "par elmore leonard et réalisé"
                        # Two consecutive NOM PROPRE with PRE before them
                        # → multi-word proper noun inside PP
                        elif _pp2_pos == "NOM PROPRE" and i > 2:
                            _pp3_pos = pos_tags[i - 3] if i - 3 < len(pos_tags) else ""
                            _pp3_low = result[i - 3].lower()
                            if (_pp3_pos == "PRE"
                                    or _pp3_low in PREPOSITIONS
                                    or _pp3_low in ("du", "des", "aux")):
                                _is_nom_subj = False
                        # OOV-only guards below: extra caution for unknown words
                        elif _is_oov_subj_b2:
                            # Single-letter fragment at i-2 (orphan elision: l, d)
                            # → check i-3 for PRE ("par l usaaf")
                            if (
                                len(_pp2_low) == 1
                                and _pp2_low.isalpha()
                                and i > 2
                            ):
                                _pp3_pos = pos_tags[i - 3] if i - 3 < len(pos_tags) else ""
                                _pp3_low = result[i - 3].lower()
                                if (_pp3_pos == "PRE"
                                        or _pp3_low in PREPOSITIONS
                                        or _pp3_low in ("du", "des", "aux")):
                                    _is_nom_subj = False
                            # NOM/NOM PROPRE at i-2 + PRE at i-3: "par alexandre X"
                            elif _pp2_pos in ("NOM", "NOM PROPRE") and i > 2:
                                _pp3_pos = pos_tags[i - 3] if i - 3 < len(pos_tags) else ""
                                _pp3_low = result[i - 3].lower()
                                if (_pp3_pos == "PRE"
                                        or _pp3_low in PREPOSITIONS
                                        or _pp3_low in ("du", "des", "aux")):
                                    _is_nom_subj = False
                    # Guard: plural NOM/ADJ (-s/-x/-z) without a singular
                    # DET at i-2 cannot be a 3sg subject for "est"
                    # "rêves et dépassé" → "rêves" is plural, not subject
                    # Only for known NOM/ADJ — proper names and OOV
                    # often end in s without being plural
                    if (
                        _is_nom_subj
                        and prev_low.endswith(("s", "x", "z"))
                        and prev_pos in ("NOM", "ADJ")
                    ):
                        _SING_DET_EST = frozenset({
                            "le", "la", "l'", "l\u2019",
                            "ce", "cette", "cet",
                            "mon", "ma", "ton", "ta", "son", "sa",
                            "un", "une", "quel", "quelle",
                        })
                        if i > 1:
                            _det_low_g6 = result[i - 2].lower()
                            if _det_low_g6 not in _SING_DET_EST:
                                _is_nom_subj = False
                        else:
                            _is_nom_subj = False
                    # Guard: ADJ ending in PP suffix + et + PP = coordination
                    # "la langue écrite et parlée" → two coordinated PPs
                    # Both prev and next look like past participles
                    if (
                        _is_nom_subj
                        and prev_pos in ("ADJ", "ADJ:pos")
                        and prev_low.endswith(_PP_SUFFIXES)
                    ):
                        _is_nom_subj = False
                    # Fallback: NOM blocked by PP guard → scan deeper
                    # for a singular subject before the PP chain.
                    # "le barrage et construit" → subject = "le barrage"
                    # Require VER/AUX POS + PP suffix (POS alone is
                    # too broad: "voit", "marque", "reste" are V3s not PP).
                    # Also block when prev is a PP (PP coordination:
                    # "dessinée par X et produite" → et links two PPs).
                    _deep_scan_pp = (
                        (
                            next_pos in ("VER", "AUX")
                            and (
                                next_low.endswith(_PP_SUFFIXES)
                                or next_low in ("né", "née", "nés", "nées")
                                # Infinitive -er: chain et→est then
                                # PP corrects er→é
                                or (next_low.endswith("er")
                                    and len(next_low) >= 5)
                            )
                        )
                        # ADJ with PP suffix: copula + predicate adj
                        # ("le prêt et gratuit"→"est gratuit")
                        or (
                            next_pos in ("ADJ", "ADJ:pos")
                            and len(next_low) >= 4
                            and next_low.endswith(_PP_SUFFIXES)
                        )
                    )
                    _prev_is_pp = (
                        prev_pos in ("VER", "AUX", "ADJ", "ADJ:pos")
                        and prev_low.endswith(_PP_SUFFIXES)
                    )
                    if (
                        not _is_nom_subj
                        and not _is_pronoun_subj
                        and _deep_scan_pp
                        and not _prev_is_pp
                    ):
                        _SING_DET_DEEP = frozenset({
                            "le", "la", "l'", "l\u2019",
                            "ce", "cette", "cet",
                            "mon", "ma", "ton", "ta", "son", "sa",
                            "un", "une", "quel", "quelle",
                        })
                        _j_deep = i - 1
                        _found_earlier_pp = False
                        while _j_deep >= max(0, i - 10):
                            _jdp = pos_tags[_j_deep] if _j_deep < len(pos_tags) else ""
                            _jdl = result[_j_deep].lower()
                            if _jdp in ("NOM", "NOM PROPRE", "ADJ", "?", ""):
                                _j_deep -= 1
                                continue
                            if _jdp == "PRE" or _jdl in PREPOSITIONS:
                                _j_deep -= 1
                                continue
                            if _jdl in ("du", "des", "aux", "d'", "d\u2019"):
                                _j_deep -= 1
                                continue
                            # VER/AUX in the chain
                            if _jdp in ("VER", "AUX"):
                                if _jdl.endswith(_PP_SUFFIXES):
                                    # Earlier PP → likely coordination
                                    _found_earlier_pp = True
                                    _j_deep -= 1
                                    continue
                                # Finite verb = clause boundary
                                break
                            if _jdp.startswith(("ART", "DET")):
                                # DET in a PP — check if PRE before it
                                if _j_deep > 0:
                                    _jdp2 = pos_tags[_j_deep - 1] if _j_deep - 1 < len(pos_tags) else ""
                                    _jdl2 = result[_j_deep - 1].lower()
                                    if _jdp2 == "PRE" or _jdl2 in PREPOSITIONS or _jdl2 in ("du", "des", "aux"):
                                        _j_deep -= 2
                                        continue
                                    # VER/AUX before DET = clause
                                    # boundary ("cause la mort",
                                    # "commençant le mars")
                                    if _jdp2 in ("VER", "AUX"):
                                        break
                                # DET not in PP + no earlier PP coord
                                if (
                                    not _found_earlier_pp
                                    and _jdl in _SING_DET_DEEP
                                ):
                                    _is_nom_subj = True
                                break
                            break
                    if _is_pronoun_subj or _is_nom_subj:
                        ancien = result[i]
                        result[i] = "est"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="est",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.et_est",
                            explication="'et' -> 'est' (sujet + _ + participe)",
                        ))
                        continue

        # "et" precede d'un sujet pronominal + suivi d'ART/DET → "est" (copule)
        # "il et le pere" -> "il est le pere"
        if curr_low == "et" and pos == "CON":
            if i > 0 and i + 1 < n:
                prev_low = result[i - 1].lower()
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                _subj_pronoun = (
                    prev_pos in ("PRO:per", "PRO:dem", "PRO:rel")
                    or prev_low in ("il", "elle", "on", "ce", "c'")
                )
                # NOM/NOM PROPRE/OOV + et + un/une = copula
                # ("le parachutisme est une", "bates est une actrice")
                _subj_nom_copula = False
                if (
                    not _subj_pronoun
                    and next_low in ("un", "une")
                ):
                    if (
                        prev_pos == "NOM PROPRE"
                        or (prev_pos == "" and len(prev_low) >= 3)
                    ):
                        # NOM PROPRE / OOV: no DET needed, check PRE
                        _in_pp_np3 = False
                        if i > 1:
                            _p2l = result[i - 2].lower()
                            _p2p = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                            if (
                                _p2p == "PRE"
                                or _p2l in PREPOSITIONS
                                or _p2l in ("du", "des", "aux")
                            ):
                                _in_pp_np3 = True
                            # ART/DET at i-2 (aux, la, le), PRE at i-3
                            elif (
                                _p2p.startswith(("ART", "DET"))
                                and i > 2
                            ):
                                _p3p = pos_tags[i - 3] if i - 3 < len(pos_tags) else ""
                                _p3l = result[i - 3].lower()
                                if (
                                    _p3p == "PRE"
                                    or _p3l in PREPOSITIONS
                                    or _p3l in ("du", "des", "aux")
                                ):
                                    _in_pp_np3 = True
                        # Extended: scan past multi-word names for PRE
                        # "de Carl Gustaf Wrangel et une" → in PP
                        if not _in_pp_np3 and i > 2:
                            for _kpp in range(i - 2, max(-1, i - 6), -1):
                                _kpp_pos = (
                                    pos_tags[_kpp]
                                    if _kpp < len(pos_tags) else ""
                                )
                                _kpp_low = result[_kpp].lower()
                                if (
                                    _kpp_pos in ("NOM PROPRE", "")
                                    and len(_kpp_low) >= 2
                                ):
                                    continue
                                if (
                                    _kpp_pos == "PRE"
                                    or _kpp_low in PREPOSITIONS
                                    or _kpp_low in ("du", "des", "aux")
                                ):
                                    _in_pp_np3 = True
                                break
                        # Guard: "un X et un Y" = coordination
                        # Check i-2 and i-3 for "un/une"
                        _both_indef_np3 = False
                        if not _in_pp_np3 and i > 1:
                            _p2l_bi = result[i - 2].lower()
                            if _p2l_bi in ("un", "une"):
                                _both_indef_np3 = True
                            elif i > 2:
                                _p3l_bi = result[i - 3].lower()
                                if _p3l_bi in ("un", "une"):
                                    _both_indef_np3 = True
                        if not _in_pp_np3 and not _both_indef_np3:
                            _subj_nom_copula = True
                    elif prev_pos == "NOM" and i > 1:
                        # Regular NOM: need SING_DET at i-2
                        _prev2_low = result[i - 2].lower()
                        if _prev2_low in SING_DET:
                            _in_pp_et = False
                            if i > 2:
                                _p3_pos = pos_tags[i - 3] if i - 3 < len(pos_tags) else ""
                                _p3_low = result[i - 3].lower()
                                if _p3_pos == "PRE" or _p3_low in PREPOSITIONS:
                                    _in_pp_et = True
                            # Guard: "un NOM et un NOM" = coordination
                            _both_indef = (
                                _prev2_low in ("un", "une")
                                and next_low in ("un", "une")
                            )
                            if not _in_pp_et and not _both_indef:
                                _subj_nom_copula = True
                if (
                    (_subj_pronoun or _subj_nom_copula)
                    and next_pos in (
                        "ART:def", "ART:ind", "ART", "DET",
                    )
                ):
                    ancien = result[i]
                    result[i] = "est"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="est",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.et_est",
                        explication="'et' -> 'est' (sujet + _ + article)",
                    ))
                    continue

        # PRO:per (3sg) + "et" + bare NOM = copula + attribut
        # "elle et membre" → "elle est membre", "il et ambassadeur"
        if curr_low == "et" and pos == "CON":
            if i > 0 and i + 1 < n:
                _prev_low_attr = result[i - 1].lower()
                _prev_pos_attr = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _next_pos_attr = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _next_low_attr = result[i + 1].lower()
                if (
                    (
                        _prev_pos_attr == "PRO:per"
                        or _prev_low_attr in ("il", "elle", "on", "ce", "c'")
                    )
                    and _next_pos_attr == "NOM"
                    and not _next_low_attr.endswith(("s", "x", "z"))
                    and len(_next_low_attr) >= 3
                ):
                    ancien = result[i]
                    result[i] = "est"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="est",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.et_est",
                        explication="'et' -> 'est' (sujet + _ + attribut)",
                    ))
                    continue

        # "entre" + "est" → "et" (entre always pairs with et, never est)
        # "entre [est]" → "entre [et]"
        # Extended: "entre X est Y" → "entre X et Y" (up to 4 tokens back)
        if curr_low == "est" and pos in ("VER", "AUX"):
            _has_entre = False
            _entre_idx = -1
            for _je in range(max(0, i - 4), i):
                if result[_je].lower() == "entre":
                    _has_entre = True
                    _entre_idx = _je
                    break
            if _has_entre:
                # Guard: "et" already between "entre" and "est" → pair satisfied
                _has_et_between = any(
                    result[k].lower() == "et"
                    for k in range(_entre_idx + 1, i)
                )
                if _has_et_between:
                    _has_entre = False
                # Guard: "et" immediately after "est" → pair exists via next
                elif i + 1 < n and result[i + 1].lower() == "et":
                    _has_entre = False
                # Guard: DET/ART right after "entre" with >2 tokens gap
                # → PP complement, not paired "entre X et Y"
                # "entre les deux indicateurs est" (distance=4) → PP
                # "entre le est le" (distance=2) → paired, keep
                elif i - _entre_idx > 2 and _entre_idx + 1 < i:
                    _after_entre_pos = (
                        pos_tags[_entre_idx + 1]
                        if _entre_idx + 1 < len(pos_tags) else ""
                    )
                    if _after_entre_pos in (
                        "ART", "ART:def", "ART:ind",
                        "ADJ:pos", "ADJ:dem",
                    ):
                        _has_entre = False
            if _has_entre:
                ancien = result[i]
                result[i] = "et"
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige="et",
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.et_est",
                    explication="'est' -> 'et' (entre...et)",
                ))
                continue

        # "est" (VER/AUX) suivi d'un DET ou PRO -> probablement "et" (coordination)
        if curr_low == "est" and pos in ("VER", "AUX"):
            if i > 0 and i + 1 < n:
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                prev_low = result[i - 1].lower()
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                _next_is_coord = (
                    next_pos in (
                        "ART:def", "ART:ind", "ART", "DET", "PRO:per", "ADJ:pos",
                        "PRO:dem",
                    )
                    or next_low.endswith(("'", "\u2019"))  # elision (l', d', qu')
                    or (
                        next_pos in ("NOM", "NOM PROPRE")
                        and next_low not in ("un", "une", "pas")
                    )
                    # Present participe / gerondif (ayant, etant, etc.)
                    or (
                        next_pos in ("VER", "AUX")
                        and next_low.endswith("ant")
                        and len(next_low) >= 4
                    )
                    # Fragments d'elision orphelins (l, d, n sans apostrophe)
                    or (len(next_low) == 1 and next_low in "dln")
                    # Conjonctions de coordination : "est comme/ou" → "et comme/ou"
                    or next_low in ("comme", "ou")
                    # Verbe conjugue (P6 -ent, P4 -ons, P5 -ez) : jamais PP
                    # "tensions est proposent" → "et proposent"
                    or (
                        next_pos == "VER"
                        and len(next_low) >= 4
                        and next_low.endswith(("ent", "ons", "ez"))
                        and not next_low.endswith("ant")
                    )
                    # AUX apres "est" : "NOM est a/ont" → "et a/ont"
                    # ("est + AUX" n'est jamais copule+auxiliaire)
                    or next_pos == "AUX"
                    # "est non" → "et non" (jamais copule + "non")
                    or next_low == "non"
                )
                # Guard: "est + un/une" = copula ("est une discipline")
                # when preceded by a subject (pronoun, NOM PROPRE, NOM, or ADJ).
                # Exception: si le mot avant est NOM/ADJ pluriel (-s/-x/-z)
                # ET n'est pas dans un PP (prev-1 n'est pas PRE),
                # "est" ne peut pas etre copule 3sg → c'est "et"
                if next_low in ("un", "une"):
                    if prev_pos in ("PRO:per", "PRO:dem", "PRO:rel",
                                    "NOM PROPRE", "NOM", "ADJ"):
                        _prev_is_plur = (
                            prev_low.endswith(("s", "x", "z"))
                            and prev_pos in ("NOM", "ADJ")
                            and len(prev_low) > 2
                        )
                        # Guard: NOM dans un PP (prev-1 = PRE) = complement
                        if _prev_is_plur and i > 1:
                            _pp_pos = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                            _pp_low = result[i - 2].lower()
                            if _pp_pos == "PRE" or _pp_low in PREPOSITIONS:
                                _prev_is_plur = False
                            # Definite singular DET → NOM is singular despite -s
                            # "le tessinois est un" → copula, not coordination
                            # Excludes "un/une" to keep "un repas est un meeting"
                            elif _pp_low in (
                                "le", "la", "l'", "l\u2019",
                                "ce", "cette", "mon", "ton", "son",
                                "sa", "ma", "ta", "notre", "votre",
                            ):
                                _prev_is_plur = False
                        # Extended PP: scan past NOM PROPRE/NOM/ADJ
                        # "des Khmers nationalistes est un" → PP
                        if _prev_is_plur and i > 2:
                            for _kpe in range(i - 2, max(-1, i - 6), -1):
                                _kpe_pos = (
                                    pos_tags[_kpe]
                                    if _kpe < len(pos_tags) else ""
                                )
                                _kpe_low = result[_kpe].lower()
                                if _kpe_pos in (
                                    "NOM PROPRE", "NOM", "ADJ",
                                    "ADJ:pos", "",
                                ):
                                    continue
                                if (
                                    _kpe_pos == "PRE"
                                    or _kpe_low in PREPOSITIONS
                                    or _kpe_low in ("du", "des", "aux")
                                ):
                                    _prev_is_plur = False
                                break
                        # Guard: invariable nouns (pays, bras, etc.)
                        if _prev_is_plur and lexique is not None:
                            _inv_infos = lexique.info(prev_low)
                            if _inv_infos and any(
                                e.get("nombre") in ("singulier", "s", "Sing")
                                for e in _inv_infos
                            ):
                                _prev_is_plur = False
                        # Guard: capitalized NOM/ADJ = likely proper noun or
                        # title ("Caux", "News", "Bourgeoises" in titles)
                        if _prev_is_plur and result[i - 1][0].isupper():
                            _prev_is_plur = False
                        if not _prev_is_plur:
                            _next_is_coord = False
                # Guard: "est + le/la/l'" when preceded by NOM singulier
                # with DET sing → copula ("le fils est le pere")
                if next_low in ("le", "la", "l'", "l\u2019"):
                    if not prev_low.endswith(("s", "x", "z")):
                        _next_is_coord = False
                    # Guard: definite singular DET or PRE at prev-2
                    elif i > 1:
                        _pp2_low_le = result[i - 2].lower()
                        _pp2_pos_le = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                        if _pp2_low_le in (
                            "le", "la", "l'", "l\u2019", "l",
                            "ce", "cette", "mon", "ton", "son",
                            "sa", "ma", "ta", "notre", "votre",
                        ):
                            _next_is_coord = False
                        # NOM in PP (de NOM est le) = complement, not subject
                        elif _pp2_pos_le == "PRE" or _pp2_low_le in PREPOSITIONS:
                            _next_is_coord = False
                # Guard: NOM PROPRE + est + ART(le/la/l'/un/une) = very likely copula
                if (
                    prev_pos == "NOM PROPRE"
                    and next_low in (
                        "le", "la", "l'", "l\u2019", "un", "une",
                    )
                ):
                    _next_is_coord = False
                # Guard: NOM PROPRE + est + NOM/ADJ/VER = very likely copula
                # "gruda est contrainte", "ayala est fondatrice"
                # VER: en texte apprenant, -ez/-ent/-ons = PP mal conjugue
                # ("leal est enlevez" = "est enlevé", pas "et enlevez")
                # Exception: NOM PROPRE inside PP = complement, not subject
                # "chez honda est remporte" → "et remporte"
                if (
                    prev_pos == "NOM PROPRE"
                    and next_pos in ("NOM", "ADJ", "VER")
                ):
                    _np_in_pp = False
                    if i >= 2:
                        _pp2_np = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                        _pp2_np_l = result[i - 2].lower()
                        if (
                            _pp2_np == "PRE"
                            or _pp2_np_l in PREPOSITIONS
                            or _pp2_np_l in ("du", "des", "aux")
                        ):
                            _np_in_pp = True
                    if not _np_in_pp:
                        _next_is_coord = False
                # Guard: DET_def_sg + NOM + est + bare_NOM = copula (passive/predicative)
                # "la ville est prise", "le site est leadeur"
                # Definite article guarantees singular subject → copula
                # Only definite articles (not possessives: "son fils et élève")
                if (
                    next_pos == "NOM"
                    and prev_pos == "NOM"
                    and i > 1
                ):
                    _pp_det_cop = result[i - 2].lower()
                    if _pp_det_cop in (
                        "le", "la", "l'", "l\u2019", "l",
                    ):
                        _next_is_coord = False
                # Guard: DET_def_sg + NOM(sg) + est + VER = copula
                # "la mort est conforment" → copula (la = singular subject)
                # Conjugated verb after copula is likely a misspelling
                if (
                    next_pos == "VER"
                    and prev_pos == "NOM"
                    and not prev_low.endswith(("s", "x", "z"))
                    and i > 1
                ):
                    _pp_det_nv = result[i - 2].lower()
                    if _pp_det_nv in (
                        "le", "la", "l'", "l\u2019", "l",
                    ):
                        _next_is_coord = False
                # Guard: ADJ + est + l/d elision = copula
                # "sérique est d'une", "marquantes est l'introduction"
                if (
                    prev_pos in ("ADJ", "ADJ:pos")
                    and len(next_low) == 1
                    and next_low in "dln"
                ):
                    _next_is_coord = False
                # Guard: ADJ + est + PRO:dem/PRO:rel = copula
                # "le cas le plus complexe est celui des" → copula
                # Exception: if another "est" already serves as copula earlier
                # "pelage est noir est celui" → second "est" = coordination
                if (
                    prev_pos in ("ADJ", "ADJ:pos")
                    and next_pos in ("PRO:dem", "PRO:rel")
                ):
                    _prior_est = any(
                        result[k].lower() == "est"
                        for k in range(max(0, i - 8), i)
                    )
                    if not _prior_est:
                        _next_is_coord = False
                # Guard: est + d'/l' + VER = copula + infinitive clause
                # "le programme est d'améliorer" → copula, not coordination
                # Also handles bare "d" / "l" tokens (missing apostrophe)
                if (
                    next_low in ("d'", "d\u2019", "d", "l'", "l\u2019", "l")
                    and i + 2 < n
                ):
                    _n2p_inf = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                    if _n2p_inf in ("VER", "AUX"):
                        _next_is_coord = False
                # Guard: common NOM(singular) + est + NOM PROPRE = copula
                # "famille est zhang" (freq=269) → copula
                # "gibert est nathalie" (freq=0.14) → coordination (low-freq = likely name)
                if (
                    prev_pos == "NOM"
                    and next_pos == "NOM PROPRE"
                    and not prev_low.endswith(("s", "x", "z"))
                ):
                    _prev_freq_np = (
                        lexique.frequence(prev_low)
                        if hasattr(lexique, "frequence") else 0.0
                    )
                    if _prev_freq_np > 5:
                        _next_is_coord = False
                # Guard: DET_def_sg + ADJ + est = copula
                # "la mort est conforment" → copula (la = singular subject)
                # Exception: "est non" is always "et non"
                if (
                    prev_pos in ("ADJ", "ADJ:pos")
                    and next_low != "non"
                    and i > 1
                ):
                    _pp_det_adj = result[i - 2].lower()
                    if _pp_det_adj in (
                        "le", "la", "l'", "l\u2019", "l",
                    ):
                        _next_is_coord = False
                # Guard: ADJ + est + ADJ:pos (possessive) = copula
                # "moderne est son meilleur ami"
                if (
                    prev_pos in ("ADJ", "ADJ:pos")
                    and next_pos == "ADJ:pos"
                ):
                    _next_is_coord = False
                # Guard: ADJ:pos + NOM(sg) + est + NOM = copula
                # "son marché est lieu" → possessive subject = singular
                if (
                    next_pos == "NOM"
                    and prev_pos == "NOM"
                    and not prev_low.endswith(("s", "x", "z"))
                    and i > 1
                ):
                    _pp_det_pos_tag = (
                        pos_tags[i - 2]
                        if i - 2 < len(pos_tags) else ""
                    )
                    _pp_det_pos_low = result[i - 2].lower()
                    if (
                        _pp_det_pos_tag == "ADJ:pos"
                        or _pp_det_pos_low in (
                            "mon", "ton", "son", "sa", "ma", "ta",
                            "notre", "votre",
                        )
                    ):
                        _next_is_coord = False
                # Guard: est + "en" + ADJ/ADV/NOM = copula expression
                # "travaux est en général de" → copula
                if (
                    next_low == "en"
                    and i + 2 < n
                ):
                    _n2_pos_en = (
                        pos_tags[i + 2]
                        if i + 2 < len(pos_tags) else ""
                    )
                    if _n2_pos_en in ("ADJ", "ADV", "NOM"):
                        _next_is_coord = False
                # "de est un" → "de et un": preposition can never be
                # copula subject, so "est" after bare "de" = coordination
                _prev_is_bare_de = (
                    prev_pos == "PRE" and prev_low == "de"
                )
                if (
                    (
                        prev_pos in ("NOM", "NOM PROPRE", "ADJ", "ADJ:pos")
                        or _prev_is_bare_de
                    )
                    and _next_is_coord
                    # Guard: elision remnants (n, l, s, etc.) mistagged as NOM
                    and not (len(prev_low) == 1 and prev_low.isalpha())
                ):
                    ancien = result[i]
                    result[i] = "et"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="et",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.et_est",
                        explication="'est' -> 'et' (coordination NOM/ADJ + DET/PRO)",
                    ))
                    continue

        # Plural NOM/ADJ + "est" + plural ADJ = coordination
        # Both sides must be plural: "économiques est financières" → "et"
        # If next ADJ is singular, "est" is likely a copula with distant subject
        if curr_low == "est" and pos in ("VER", "AUX"):
            if i > 0 and i + 1 < n:
                _prev_pos_pa = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _prev_low_pa = result[i - 1].lower()
                _next_pos_pa = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _next_low_pa = result[i + 1].lower()
                if (
                    _prev_pos_pa in ("NOM", "ADJ", "ADJ:pos")
                    and _prev_low_pa.endswith(("s", "x", "z"))
                    and len(_prev_low_pa) > 2
                    and _next_pos_pa in ("ADJ", "ADJ:pos")
                    and _next_low_pa.endswith(("s", "x", "z"))
                    and len(_next_low_pa) > 2
                ):
                    ancien = result[i]
                    result[i] = "et"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="et",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.et_est",
                        explication="'est' -> 'et' (pluriel + ADJ = coordination)",
                    ))
                    continue

        # OOV + est + (OOV/NOM PROPRE/plural NOM) → name coordination → "et"
        # Handles name lists: "moog est oberheim", "patty est selma",
        # "athletes est personnages"
        if curr_low == "est" and pos in ("VER", "AUX"):
            if i > 0 and i + 1 < n:
                _prev_pos_nc = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _prev_low_nc = result[i - 1].lower()
                _next_pos_nc = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _next_low_nc = result[i + 1].lower()
                if _prev_pos_nc in ("", "NOM PROPRE") and len(_prev_low_nc) >= 2:
                    _is_name_coord = False
                    # OOV + est + OOV/NOM PROPRE
                    if _prev_pos_nc == "" and (
                        _next_pos_nc == "NOM PROPRE"
                        or (_next_pos_nc == "" and len(_next_low_nc) >= 2)
                    ):
                        _is_name_coord = True
                    # NOM PROPRE + est + NOM PROPRE (strict: both must be names)
                    elif (
                        _prev_pos_nc == "NOM PROPRE"
                        and _next_pos_nc == "NOM PROPRE"
                    ):
                        _is_name_coord = True
                    # OOV + est + plural NOM (OOV prev only)
                    elif (
                        _prev_pos_nc == ""
                        and _next_pos_nc == "NOM"
                        and _next_low_nc.endswith(("s", "x", "z"))
                        and len(_next_low_nc) > 2
                    ):
                        _is_name_coord = True
                    # Guard C1: NOM_PROPRE + OOV + est + non-name
                    # "nora krug est née" → copula (next is not a name)
                    # Exception: "akira natori est takeshi" → coordination
                    if _is_name_coord and i > 1:
                        _pp2_nc = (
                            pos_tags[i - 2]
                            if i - 2 < len(pos_tags) else ""
                        )
                        if (
                            _pp2_nc == "NOM PROPRE"
                            and _next_pos_nc not in ("NOM PROPRE", "")
                        ):
                            _is_name_coord = False
                    if _is_name_coord:
                        ancien = result[i]
                        result[i] = "et"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="et",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.et_est",
                            explication="'est' -> 'et' (noms propres coordination)",
                        ))
                        continue

        # VER + "est" + VER = coordination of verbal forms
        # "connaissaient est chantaient" → "et", "accompagna est mourut" → "et"
        # Guards: skip if either neighbor ends in -é/-ée/-és/-ées (likely passive)
        #         skip if next is infinitive (VER_conj + être + INF = passive/aux)
        if curr_low == "est" and pos in ("VER", "AUX"):
            if i > 0 and i + 1 < n:
                _prev_pos_vv = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _next_pos_vv = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _prev_low_vv = result[i - 1].lower()
                _next_low_vv = result[i + 1].lower()
                _pp_vv_suffixes = (
                    "\u00e9", "\u00e9s", "\u00e9e", "\u00e9es",
                    "i", "is",
                    "u", "us",
                )
                _either_pp = (
                    _prev_low_vv.endswith(_pp_vv_suffixes)
                    or _next_low_vv.endswith(_pp_vv_suffixes)
                )
                # Exclude imparfait/conditionnel (-ait/-aient) which
                # accidentally match -it suffix
                if _either_pp:
                    if (
                        _prev_low_vv.endswith(("ait", "aient", "ais"))
                        and not _next_low_vv.endswith(_pp_vv_suffixes)
                    ):
                        _either_pp = False
                    elif (
                        _next_low_vv.endswith(("ait", "aient", "ais"))
                        and not _prev_low_vv.endswith(_pp_vv_suffixes)
                    ):
                        _either_pp = False
                _modes = morpho.get("mode", [])
                _next_mode = _modes[i + 1] if i + 1 < len(_modes) else "_"
                if (
                    _prev_pos_vv == "VER"
                    and _next_pos_vv in ("VER", "AUX")
                    and not _either_pp
                    and _next_mode != "inf"
                ):
                    ancien = result[i]
                    result[i] = "et"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="et",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.et_est",
                        explication="'est' -> 'et' (VER + VER coordination)",
                    ))
                    continue

        # Parallel prepositional structure: "en X est en Y" → "et"
        # When the same preposition appears both before and after "est",
        # it's coordination, not copula.
        # "de" requires ART after it ("de la faune est de la flore")
        # because "est de VALUE" is a copula ("la superficie est de 100 km²").
        _SAFE_PARALLEL_PREP = frozenset({
            "en", "au", "aux", "par", "pour", "avec", "dans",
            "sur", "sous", "sans", "entre", "vers", "contre",
            "à", "chez",
        })
        if curr_low == "est" and pos in ("VER", "AUX"):
            if i + 1 < n:
                _next_pos_pp = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _next_low_pp = result[i + 1].lower()
                _is_safe_prep = (
                    _next_pos_pp == "PRE"
                    and (
                        _next_low_pp in _SAFE_PARALLEL_PREP
                        or (
                            _next_low_pp == "de"
                            and i + 2 < n
                            and (pos_tags[i + 2] if i + 2 < len(pos_tags) else "").startswith(("ART", "DET"))
                        )
                        # "de X est de Y": accept "de" without ART when
                        # matching de-form precedes (tight parallel)
                        # Guard: require NOM/PROPRE at i+2 to exclude
                        # copula "est de VALUE" (est de 800mm, est de plus)
                        or (
                            _next_low_pp in ("de", "d'", "d\u2019", "d")
                            and i >= 2
                            and result[i - 2].lower()
                            in ("de", "d'", "d\u2019", "d", "du", "des")
                            and i + 2 < n
                            and (pos_tags[i + 2] if i + 2 < len(pos_tags) else "")
                            in ("NOM", "NOM PROPRE")
                        )
                    )
                )
                if _is_safe_prep:
                    # Search for the same preposition in the 5 tokens before "est"
                    # Contractions count: du/des/d'/d ≈ de, au/aux ≈ à
                    _DE_FORMS = frozenset({"de", "du", "des", "d'", "d\u2019", "d"})
                    _A_FORMS = frozenset({"à", "au", "aux"})
                    _found_same_prep = False
                    for _kpp in range(i - 1, max(-1, i - 6), -1):
                        _kpp_low = result[_kpp].lower()
                        _kpp_pos = pos_tags[_kpp] if _kpp < len(pos_tags) else ""
                        # Exact match (normal case)
                        if _kpp_pos == "PRE" and _kpp_low == _next_low_pp:
                            _found_same_prep = True
                            break
                        # Bidirectional contraction matching for de-forms
                        if _next_low_pp in _DE_FORMS and _kpp_low in _DE_FORMS:
                            _found_same_prep = True
                            break
                        # Bidirectional contraction matching for à-forms
                        if _next_low_pp in _A_FORMS and _kpp_low in _A_FORMS:
                            _found_same_prep = True
                            break
                    if _found_same_prep:
                        ancien = result[i]
                        result[i] = "et"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="et",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.et_est",
                            explication="'est' -> 'et' (parallel PRE structure)",
                        ))
                        continue

        # PREP + NOM/PROPRE + est + VER(non-PP, non-INF) → coordination
        # When prev word is inside a PP, it cannot be subject for "est"
        # "chez honda est remporte" → "et remporte"
        # Guard: next must not be PP (copula: "est construite")
        # Guard: next must not be infinitive
        if curr_low == "est" and pos in ("VER", "AUX"):
            if i >= 2 and i + 1 < n:
                _p1_pnv = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _p2_pnv = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                _p2_low_pnv = result[i - 2].lower()
                _n1_pnv = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _n1_low_pnv = result[i + 1].lower()
                _modes_pnv = morpho.get("mode", [])
                _n1_mode_pnv = _modes_pnv[i + 1] if i + 1 < len(_modes_pnv) else "_"
                # Guard: skip if next is also ADJ in lexique (copula)
                # "en richardson est present" → copula, not coordination
                _n1_also_adj_pnv = False
                if lexique is not None:
                    _n1i_pnv = lexique.info(_n1_low_pnv)
                    _n1_also_adj_pnv = any(
                        (e.get("cgram") or "").startswith("ADJ")
                        for e in _n1i_pnv
                    )
                if (
                    _p1_pnv in ("NOM", "NOM PROPRE", "")
                    and (
                        _p2_pnv == "PRE"
                        or _p2_low_pnv in PREPOSITIONS
                        or _p2_low_pnv in ("du", "des", "aux")
                    )
                    and _n1_pnv in ("VER", "AUX")
                    and not _n1_low_pnv.endswith(_PP_SUFFIXES)
                    and _n1_mode_pnv != "inf"
                    and not _n1_also_adj_pnv
                ):
                    ancien = result[i]
                    result[i] = "et"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="et",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.et_est",
                        explication="'est' -> 'et' (NOM in PP + VER coordination)",
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
                regle="homophone.et_est",
                explication="'est' -> 'et' (conjonction attendue)",
            ))
            continue

        # --- son / sont ---
        # "ne son pas/plus/jamais" → "ne sont pas" (negation de etre 3pl)
        # Pattern tres sur : ne/n' + son + mot negatif
        if curr_low == "son" and i > 0 and i + 1 < n:
            if (result[i - 1].lower() in ("ne", "n'")
                    and result[i + 1].lower() in (
                        "pas", "plus", "jamais", "rien", "point",
                    )):
                ancien = result[i]
                result[i] = "sont"
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige="sont",
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.son_sont",
                    explication="'son' -> 'sont' (negation ne...pas)",
                ))
                continue

        # "son" precede directement par un sujet pluriel (sans verbe entre) →
        # probablement "sont" (copule etre 3pl).
        # Ex: "les enfants son gentil" → "les enfants sont gentils"
        # Guard: "de son" / "à son" = possessif, pas copule
        # Guard: "son" + NOM = possessif ("son tempérament")
        if curr_low == "son" and pos in ("ADJ:pos", "ADJ"):
            _prev_pre = (
                i > 0
                and (pos_tags[i - 1] if i - 1 < len(pos_tags) else "") == "PRE"
            )
            # Guard: mot suivant est NOM/NOM PROPRE → possessif, pas copule
            # Override: si NOM est clairement pluriel, "son" (singulier)
            # ne peut pas etre un possessif ("son morts" impossible)
            _next_is_nom_son = (
                i + 1 < n
                and (pos_tags[i + 1] if i + 1 < len(pos_tags) else "")
                in ("NOM", "NOM PROPRE")
            )
            if _next_is_nom_son and i + 1 < n and lexique is not None:
                _nsn_low = result[i + 1].lower()
                if (
                    _nsn_low.endswith(("s", "x"))
                    and len(_nsn_low) >= 4
                ):
                    _nsn_base = _nsn_low[:-1]
                    if lexique.existe(_nsn_base):
                        _next_is_nom_son = False
            # Guard: "du son" = "de+le son" (son=NOM, not verb)
            _prev_du_son = (
                i > 0
                and result[i - 1].lower() in ("du", "au")
            )
            _plur_subj_direct = False
            if not _prev_pre and not _next_is_nom_son and not _prev_du_son:
                # "son" + ART → impossible possessive (DET+DET),
                # must be "sont" (e.g. "ils son les premiers")
                if i + 1 < n:
                    _next_pos_son = (
                        pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                    )
                    if _next_pos_son.startswith(("ART", "DET")):
                        _plur_subj_direct = True
                # "son" + ADV → often "sont" with adverb
                # "ils son aussi partis" → "sont aussi"
                if (
                    not _plur_subj_direct
                    and i + 1 < n
                ):
                    _next_pos_son = (
                        pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                    )
                    _next_low_son = result[i + 1].lower()
                    if (
                        _next_pos_son == "ADV"
                        and _next_low_son in (
                            "aussi", "encore", "toujours",
                            "souvent", "donc", "alors",
                            "vraiment", "tous", "toutes",
                            "beaucoup", "également",
                            "très", "plus", "moins",
                        )
                    ):
                        # Check for plural subject before (traverse PP chains)
                        # Don't treat "qui" as plural (relative pronoun
                        # with unknown antecedent number); scan past it.
                        for _k_adv in range(i - 1, max(-1, i - 10), -1):
                            _w_adv = result[_k_adv].lower()
                            _pk_adv = (
                                pos_tags[_k_adv]
                                if _k_adv < len(pos_tags) else ""
                            )
                            if _w_adv in ("ils", "elles"):
                                _plur_subj_direct = True
                                break
                            if (
                                _pk_adv in ("NOM", "ADJ")
                                and _w_adv.endswith(("s", "x", "z"))
                            ):
                                _plur_subj_direct = True
                                break
                            if _pk_adv in (
                                "ART:def", "ART:ind", "ART",
                                "DET", "ADJ:pos", "ADJ", "ADV",
                                "PRO:rel",
                                # Traverse PP chains (a chevet plat, de X)
                                "NOM", "NOM PROPRE", "PRE",
                            ) or _w_adv in (
                                "du", "des", "aux", "au",
                            ):
                                continue
                            break
                # "son" + clearly plural ADJ/VER → "sont" (no subject needed)
                # "son" is singular possessive, cannot precede a plural word.
                if not _plur_subj_direct and i + 1 < n and lexique is not None:
                    _nlow_sc = result[i + 1].lower()
                    _npos_sc = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                    if (
                        _npos_sc in ("ADJ", "ADJ:pos", "VER", "AUX")
                        and _nlow_sc.endswith("s")
                        and len(_nlow_sc) >= 4
                    ):
                        _base_sc = _nlow_sc[:-1]
                        _base_infos_sc = lexique.info(_base_sc)
                        if any(
                            (e.get("cgram") or "").startswith(
                                ("ADJ", "VER", "AUX"))
                            for e in _base_infos_sc
                        ):
                            _plur_subj_direct = True
                            # Guard: if word at i+2 is singular NOM,
                            # likely "son ADJ(typo) NOM" not "sont ADJ NOM(sg)"
                            if i + 2 < n:
                                _n2p = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                                _n2l = result[i + 2].lower()
                                if (
                                    _n2p == "NOM"
                                    and not _n2l.endswith(("s", "x", "z"))
                                ):
                                    _plur_subj_direct = False
                if not _plur_subj_direct:
                    for _k in range(i - 1, max(-1, i - 10), -1):
                        _w = result[_k].lower()
                        _pk = pos_tags[_k] if _k < len(pos_tags) else ""
                        if _w in ("ils", "elles"):
                            _plur_subj_direct = True
                            break
                        if _pk in ("NOM", "ADJ") and _w.endswith(("s", "x", "z")):
                            _plur_subj_direct = True
                            break
                        if _pk in (
                            "ART:def", "ART:ind", "ART", "DET",
                            "ADJ:pos", "ADJ", "ADV", "PRO:rel",
                            "NOM", "NOM PROPRE", "PRE",
                        ) or _w in ("du", "des", "aux", "au"):
                            continue
                        break  # Verbe ou autre → arreter
            if _plur_subj_direct and lexique is not None and lexique.existe("sont"):
                ancien = result[i]
                result[i] = "sont"
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige="sont",
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.son_sont",
                    explication="'son' -> 'sont' (sujet pluriel, copule)",
                ))
                # Pluraliser aussi le mot suivant s'il est ADJ dans le lexique
                if i + 1 < n:
                    _nw = result[i + 1]
                    _nlow = _nw.lower()
                    if (
                        not _nlow.endswith(("s", "x", "z"))
                        and len(_nw) > 1
                        and lexique is not None
                    ):
                        _infos = lexique.info(_nlow)
                        _is_adj = _infos and any(
                            e.get("cgram", "").startswith("ADJ") for e in _infos
                        )
                        if _is_adj:
                            for _cand in generer_candidats_pluriel(_nw):
                                if lexique.existe(_cand):
                                    _anc = result[i + 1]
                                    result[i + 1] = _cand
                                    corrections.append(Correction(
                                        index=i + 1,
                                        original=_anc,
                                        corrige=_cand,
                                        type_correction=TypeCorrection.GRAMMAIRE,
                                        regle="homophone.son_sont",
                                        explication="Accord pluriel apres 'sont'",
                                    ))
                                    break
                continue

        # "son" + VER/PP -> probablement "sont" (quand sujet pluriel avant)
        # Guard: "de son" / "à son" = possessif, pas copule
        # Guard: "son" + NOM = possessif ("son mépris")
        # Guard: "du son" = son=NOM (not verb)
        if curr_low == "son" and pos in ("ADJ:pos", "ADJ"):
            _prev_pre = (
                i > 0
                and (pos_tags[i - 1] if i - 1 < len(pos_tags) else "") == "PRE"
            )
            _prev_du_son2 = (
                i > 0
                and result[i - 1].lower() in ("du", "au")
            )
            _next_pos_son2 = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
            _next_is_nom_son2 = (
                i + 1 < n
                and _next_pos_son2 in ("NOM", "NOM PROPRE")
            )
            if not _prev_pre and not _prev_du_son2 and not _next_is_nom_son2 and i + 1 < n:
                next_pos = _next_pos_son2
                next_low = result[i + 1].lower()
                _next_verb = next_pos in ("VER", "AUX") or (
                    len(next_low) >= 4 and next_low.endswith(_PP_SUFFIXES)
                    and next_pos not in (
                        "NOM", "NOM PROPRE", "ADJ", "ADV",
                        "PRE", "CON", "ART:def", "ART:ind",
                    )
                )
                # "son" + clearly plural PP → "sont" (no subject scan needed)
                # "son" is singular possessive, cannot precede a plural PP.
                # Clearly plural PP: remove final 's' and check if base exists
                # as VER in lexique (e.g. intervenus→intervenu exists)
                _next_clearly_plural = False
                if (
                    _next_verb
                    and next_low.endswith("s")
                    and len(next_low) >= 4
                    and lexique is not None
                ):
                    _base = next_low[:-1]
                    _base_infos = lexique.info(_base)
                    _next_clearly_plural = any(
                        (e.get("cgram") or "").startswith(("VER", "AUX"))
                        for e in _base_infos
                    )
                if _next_verb:
                    # Verifier qu'un sujet pluriel precede
                    _plur_subj = _next_clearly_plural
                    if not _plur_subj:
                        for _k in range(i - 1, max(-1, i - 10), -1):
                            _w = result[_k].lower()
                            if _w in ("ils", "elles"):
                                _plur_subj = True
                                break
                            _pk = pos_tags[_k] if _k < len(pos_tags) else ""
                            if _pk in ("NOM", "ADJ") and _w.endswith(("s", "x", "z")):
                                _plur_subj = True
                                break
                            if _pk in (
                                "ADV", "PRE", "ADJ:pos", "ADJ",
                                "PRO:rel", "NOM", "NOM PROPRE",
                                "ART:def", "ART:ind", "ART", "DET",
                            ) or _w in ("du", "des", "aux", "au"):
                                continue
                            break
                    if _plur_subj and lexique is not None and lexique.existe("sont"):
                        ancien = result[i]
                        result[i] = "sont"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="sont",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.son_sont",
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
                regle="homophone.son_sont",
                explication="'sont' -> 'son' (possessif)",
            ))
            continue

        # "sont" apres PRE + suivi de NOM -> "son" (possessif)
        # "avec sont velo" -> "avec son vélo"
        if curr_low == "sont" and pos in ("VER", "AUX"):
            if i > 0 and i + 1 < n:
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if prev_pos == "PRE" and next_pos in ("NOM", "ADJ", "ADJ:pos"):
                    ancien = result[i]
                    result[i] = "son"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="son",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.son_sont",
                        explication="'sont' -> 'son' (PRE + possessif + NOM)",
                    ))
                    continue

        # "sont" suivi de NOM/ADJ singulier (pas de sujet pluriel avant) -> "son"
        # "nomme en aout sont jeune fils" -> "son jeune fils"
        # Guard: ne pas convertir si un sujet pluriel precede
        # Guard: ne pas convertir si le mot suivant est pluriel (sont+ADJplur = copule)
        if curr_low == "sont" and pos in ("VER", "AUX"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                # Check if NOM ending in "s" is invariable (repas, bras, etc.)
                # Invariable: has a singulier entry with same form
                _looks_plural = next_low.endswith(("s", "x", "z"))
                if _looks_plural and lexique is not None:
                    _inv_infos = lexique.info(next_low)
                    _has_sing = any(
                        e.get("nombre") in ("singulier", "s", "Sing")
                        and e.get("cgram") in ("NOM", "ADJ")
                        for e in _inv_infos
                    ) if _inv_infos else False
                    if _has_sing:
                        _looks_plural = False  # Invariable
                if (
                    next_pos in ("NOM", "ADJ", "ADJ:pos")
                    and not _looks_plural
                ):
                    # Verifier qu'il n'y a pas de sujet pluriel avant
                    _has_plur_subj = False
                    for _k in range(i - 1, max(-1, i - 5), -1):
                        _w = result[_k].lower()
                        _pk = pos_tags[_k] if _k < len(pos_tags) else ""
                        if _w in ("ils", "elles"):
                            _has_plur_subj = True
                            break
                        if _pk == "NOM" and _w.endswith(("s", "x", "z")):
                            _has_plur_subj = True
                            break
                        if _pk in ("VER", "AUX", "CON"):
                            break
                    if not _has_plur_subj:
                        ancien = result[i]
                        result[i] = "son"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="son",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.son_sont",
                            explication="'sont' -> 'son' (possessif + NOM/ADJ)",
                        ))
                        continue

        # --- a / à ---
        if curr_low == "a" and pos in ("VER", "AUX"):
            # Guard: "a contrario", "a priori", "a posteriori" etc.
            _next_low_a = result[i + 1].lower() if i + 1 < n else ""
            if _next_low_a in (
                "contrario", "priori", "posteriori", "fortiori", "minima",
                "maxima", "cappella",
            ):
                continue  # latin expression, skip
            # Chercher si "a" est probablement l'auxiliaire avoir (3sg)
            _is_aux = False
            for _k in range(i - 1, max(-1, i - 5), -1):
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
                if _pk in ("NOM", "NOM PROPRE"):
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
                if _pk == "ADJ":
                    # ADJ before NOM is a noun phrase modifier
                    # ("le jeune ecclesiastique a") — continue scanning
                    continue
                if _pk in ("VER", "AUX"):
                    # Capitalized word at sentence start tagged VER but
                    # with NOM entries → probably a proper noun (subject)
                    if (
                        _k == 0
                        and result[_k][0].isupper()
                        and lexique is not None
                    ):
                        _cap_infos = lexique.info(result[_k])
                        if _cap_infos and any(
                            (e.get("cgram") or "") in ("NOM", "NOM PROPRE")
                            for e in _cap_infos
                        ):
                            _is_aux = True
                    break
            # Guard: "a" at sentence start (position 0) without subject
            # — too risky to convert (truncated sentences, e.g. "[Objet] a une magnitude...")
            _orig_a = originaux[i] if originaux and i < len(originaux) else result[i]
            if not _is_aux and i == 0:
                pass  # skip — no subject possible at sentence start
            elif not _is_aux and i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower() if i + 1 < n else ""
                # Guard: a + PP form (laissé, pris, fait...) = aux avoir
                _next_is_pp_a = (
                    next_pos in ("VER", "AUX")
                    and len(next_low) >= 3
                    and next_low.endswith(_PP_SUFFIXES)
                )
                # Guard: "et a un/une" → avoir, not preposition
                # "la tribune ... et a un toit" = has a roof
                _prev_low_a = result[i - 1].lower() if i > 0 else ""
                _et_a_art = (
                    _prev_low_a in ("et", "ou")
                    and next_pos in ("ART:ind", "ART")
                    and next_low in ("un", "une")
                )
                # a + ART, a + PRE, a + NOM, a + elision (l', d'), a + VER (inf)
                if not _next_is_pp_a and not _et_a_art and (
                    next_pos in (
                        "VER", "ART:def", "ART:ind", "ART", "PRE",
                        "NOM", "NOM PROPRE",
                    ) or next_low.endswith(("'", "\u2019"))
                ):
                    ancien = result[i]
                    result[i] = "à"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="à",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.a_a",
                        explication="'a' -> 'à' (preposition)",
                    ))
                    continue

        # "à" devant VER quand sujet pronominal precede -> "a" (auxiliaire avoir)
        # "elle à mange" -> "elle a mange"
        if curr_low == "à" and i + 1 < n:
            next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
            next_low = result[i + 1].lower()
            _next_is_verb = next_pos == "VER" or (
                len(next_low) >= 4 and next_low.endswith(_PP_SUFFIXES)
            )
            if _next_is_verb:
                for _k in range(i - 1, max(-1, i - 2), -1):
                    _w = result[_k].lower()
                    _pk = pos_tags[_k] if _k < len(pos_tags) else ""
                    # Only accept actual subject pronouns, skip object pronouns
                    if _w in ("il", "elle", "on", "ils", "elles",
                              "je", "j'", "tu", "nous", "vous", "qui"):
                        ancien = result[i]
                        result[i] = "a"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="a",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.a_a",
                            explication="'à' -> 'a' (auxiliaire avoir apres sujet)",
                        ))
                        break
                    # Stop at verbs (another verb means "à" is a preposition)
                    if _pk in ("VER", "AUX"):
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
                regle="homophone.a_a",
                explication="'à' -> 'a' (auxiliaire avoir)",
            ))
            continue

        # --- ou / où ---
        # "ou" etiquete PRO:rel -> "où" (pronom relatif lieu/temps)
        # Guard: ne pas re-accentuer si l'original etait deja "où" (ortho a corrige)
        if curr_low == "ou" and pos == "PRO:rel":
            _orig_low = originaux[i].lower() if originaux and i < len(originaux) else ""
            if _orig_low != "où":
                ancien = result[i]
                result[i] = "où"
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige="où",
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.ou_ou",
                    explication="'ou' -> 'où' (pronom relatif)",
                ))
                continue

        # "ou" etiquete ADV -> "où" (adverbe interrogatif)
        if curr_low == "ou" and pos == "ADV":
            _orig_low = originaux[i].lower() if originaux and i < len(originaux) else ""
            if _orig_low != "où":
                ancien = result[i]
                result[i] = "où"
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige="où",
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.ou_ou",
                    explication="'ou' -> 'où' (adverbe interrogatif)",
                ))
                continue

        # "ou" (CON) at sentence start + VER/AUX or inversion -> "où"
        # "ou est la gare" -> "où est la gare"
        # "ou vas-tu" -> "où vas-tu"
        _INVERSION_SUFFIXES = (
            "-tu", "-il", "-elle", "-on", "-nous", "-vous",
            "-ils", "-elles", "-je", "-t-il", "-t-elle", "-t-on",
        )
        if curr_low == "ou" and pos == "CON":
            _is_sent_start = (
                i == 0
                or (i > 0 and result[i - 1] in (".", "!", "?", ";", ":"))
            )
            if _is_sent_start and i + 1 < n:
                _next_pos_ou = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _next_low_ou = result[i + 1].lower()
                _next_is_verb_ou = _next_pos_ou in ("VER", "AUX")
                _next_is_inversion = any(
                    _next_low_ou.endswith(s) for s in _INVERSION_SUFFIXES
                )
                # Guard: "ou bien" = disjonction, pas interrogatif
                if (
                    (_next_is_verb_ou or _next_is_inversion)
                    and _next_low_ou not in ("bien", "alors", "sinon")
                ):
                    ancien = result[i]
                    result[i] = "où"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="où",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.ou_ou",
                        explication="'ou' -> 'où' (interrogatif en debut de phrase)",
                    ))
                    continue

        # "ou" (CON) + pronom sujet + VER, precede par VER -> "où"
        # (interrogatif indirect : "je sais ou il habite")
        # Guard: ne pas re-accentuer si l'original etait deja "où"
        if curr_low == "ou" and pos == "CON":
            _orig_low_ou = originaux[i].lower() if originaux and i < len(originaux) else ""
            if _orig_low_ou == "où":
                pass  # skip, ortho already corrected où→ou
            elif i > 0 and i + 2 < n:
                next_low = result[i + 1].lower()
                next2_pos = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                _is_pronom_sujet = next_low in (
                    "il", "elle", "on", "je", "j'", "tu",
                    "nous", "vous", "ils", "elles",
                )
                if _is_pronom_sujet and next2_pos in ("VER", "AUX"):
                    # Verifier qu'un VER precede (contexte interro indirecte)
                    # Skip ADV, PRO:dem/PRO:rel ("celui ou il est")
                    _ver_before = False
                    for _k in range(i - 1, max(-1, i - 4), -1):
                        _pk = pos_tags[_k] if _k < len(pos_tags) else ""
                        if _pk in ("VER", "AUX"):
                            _ver_before = True
                            break
                        if _pk in ("ADV", "PRO:dem", "PRO:rel"):
                            continue
                        break
                    if _ver_before:
                        ancien = result[i]
                        result[i] = "où"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="où",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.ou_ou",
                            explication="'ou' -> 'où' (interrogatif indirect)",
                        ))
                        continue

        # "où" en contexte de coordination (NOM/ADJ + où + ART/PRE/elision)
        # -> "ou" (conjonction disjonctive)
        # "phylactères où des écus" -> "ou des écus"
        if curr_low == "où" and pos in ("PRO:rel", "ADV"):
            if i > 0 and i + 1 < n:
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                _prev_is_nominal = prev_pos in (
                    "NOM", "ADJ", "ADJ:pos", "VER", "PRE",
                )
                _next_is_coord = (
                    next_pos in (
                        "ART:def", "ART:ind", "ART", "DET", "PRE",
                        "NOM",  # "syndicat où mouvement" = coordination
                        "ADV",  # "peut où non" = coordination
                    )
                    or next_low.endswith(("'", "\u2019"))
                    # Fragments d'elision sans apostrophe (d, l, n, s)
                    or (len(next_low) == 1 and next_low in "dlns")
                    # VER as PP = coordination ("friandises où décortiquées")
                    # Guard: only PP forms, not conjugated verbs
                    or (
                        next_pos in ("VER", "AUX")
                        and len(next_low) >= 4
                        and next_low.endswith(_PP_SUFFIXES)
                    )
                )
                # Guard: "où + ART:def + NOM" = relative clause, not coordination
                # "l'arrondissement où le mélange" = relative (definite = specific)
                # "phylactères où des écus" = coordination (indefinite)
                _is_relative_clause = False
                if (
                    _next_is_coord
                    and next_low in ("le", "la", "l'", "l\u2019", "les")
                    and i + 2 < n
                ):
                    _next2_pos = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                    if _next2_pos in ("NOM", "ADJ"):
                        _is_relative_clause = True
                # Guard: "où + ART + NOM ... VER" = relative clause
                # "moment où un orage éclate" = relative (VER nearby)
                # "occasions où des communautés réussissent" = relative
                if (
                    not _is_relative_clause
                    and _next_is_coord
                    and next_pos.startswith(("ART", "DET"))
                    and i + 2 < n
                ):
                    _n2p = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                    if _n2p in ("NOM", "ADJ"):
                        # Look ahead for conjugated VER within 3 positions
                        for _k_ou in range(i + 3, min(i + 6, n)):
                            _kp = pos_tags[_k_ou] if _k_ou < len(pos_tags) else ""
                            if _kp in ("VER", "AUX"):
                                _is_relative_clause = True
                                break
                # Guard: "où + NOM" when prev is a location-like word
                # "la ville où paris" = relative clause
                # Only allow NOM coordination if prev also looks like a list item
                # Exclude single-letter NOM (orphan elision fragments like "d")
                if (
                    _next_is_coord
                    and next_pos == "NOM"
                    and len(next_low) > 1
                    and prev_pos not in ("NOM", "ADJ")
                ):
                    _next_is_coord = False
                # Guard: "NOM" followed by ART/DET → likely inverted
                # verb-subject, not coordination
                # "où figure un rang" = relative ("where a row appears")
                if (
                    _next_is_coord
                    and next_pos == "NOM"
                    and i + 2 < n
                ):
                    _n2p_ou = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                    if _n2p_ou in (
                        "ART:def", "ART:ind", "ART", "DET",
                    ):
                        _next_is_coord = False
                if _prev_is_nominal and _next_is_coord and not _is_relative_clause:
                    ancien = result[i]
                    result[i] = "ou"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="ou",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.ou_ou",
                        explication="'où' -> 'ou' (coordination)",
                    ))
                    continue

        # "ou" (CON) suivi de PRO:per -> probablement "où" (relatif)
        # "italie ou il réside" -> "italie où il réside"
        if curr_low == "ou" and pos == "CON":
            if i > 0 and i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                if (
                    prev_pos in ("NOM", "NOM PROPRE", "?", "")
                    and (
                        next_pos == "PRO:per"
                        # "un/une" after place name = relative clause
                        # "Persépolis ou un incendie" → "où"
                        # Guard: "diplôme ou un baccalauréat" = coordination
                        or (
                            next_low in ("un", "une")
                            and prev_pos == "NOM PROPRE"
                        )
                    )
                ):
                    ancien = result[i]
                    result[i] = "où"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="où",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.ou_ou",
                        explication="'ou' -> 'où' (relatif apres NOM)",
                    ))
                    continue

        # "ou" (CON) apres NOM, suivi de VER conjugue -> "où" (relatif)
        # "paris ou réside son frère" -> "paris où réside son frère"
        # Guard: infinitif ou participe passe = coordination, pas relatif
        if curr_low == "ou" and pos == "CON":
            if i > 0 and i + 1 < n:
                _prev_pos_r = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _next_pos_r = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _next_low_r = result[i + 1].lower()
                _is_inf_or_pp = (
                    _next_low_r.endswith(("er", "ir", "re", "oir"))
                    or _next_low_r.endswith((
                        "é", "ée", "és", "ées",
                        "is", "ise", "ises",
                        "us", "ue", "ues",
                        "it", "ite", "ites",
                    ))
                )
                if (
                    _prev_pos_r in ("NOM", "NOM PROPRE")
                    and _next_pos_r in ("VER", "AUX")
                    and not _is_inf_or_pp
                ):
                    ancien = result[i]
                    result[i] = "où"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="où",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.ou_ou",
                        explication="'ou' -> 'où' (relatif + verbe conjugue)",
                    ))
                    continue

        # --- elles / elle ---
        # "elles" + auxiliaire suppletif singulier → "elle"
        # Seulement pour les formes dont le pluriel est suppletif
        # (a→ont, est→sont, va→vont) : l'erreur est sur le pronom.
        if curr_low == "elles" and pos == "PRO:per":
            if i + 1 < n:
                _next_low_el = result[i + 1].lower()
                if _next_low_el in (
                    "a", "est", "va", "fait", "peut", "sait",
                    "veut", "doit", "avait", "était", "devait",
                    "pouvait", "sera", "fera", "ira", "aura",
                    "fut",
                ):
                    # Guard: si un mot proche apres le verbe porte un accord
                    # pluriel feminin, c'est le verbe qui est faux, pas le pronom
                    # Ex: "elles est marquées" → garder "elles", CONJ fixera le verbe
                    _plur_agree_el = False
                    for _k_el in range(i + 2, min(n, i + 5)):
                        _kw_el = result[_k_el].lower()
                        if _kw_el.endswith(("ées", "ées", "ées")):
                            _plur_agree_el = True
                            break
                        if _kw_el.endswith(("és", "ies", "ues", "ites", "ertes")):
                            _plur_agree_el = True
                            break
                    if _plur_agree_el:
                        continue
                    ancien = result[i]
                    _elle = "Elle" if ancien[0].isupper() else "elle"
                    result[i] = _elle
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=_elle,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.elle_elles",
                        explication="'elles' -> 'elle' (verbe singulier)",
                    ))
                    continue

        # --- on / ont ---
        # "on" apres NOM/PRO + suivi de VER/ADV -> "ont" (3pl avoir)
        # Guard: "qu on" = "qu'on" (que + on), pas un vrai NOM avant
        if curr_low == "on" and pos in ("PRO:ind", "PRO:per"):
            if i > 0 and i + 1 < n:
                prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                prev_low = result[i - 1].lower() if i > 0 else ""
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _prev_is_fragment = prev_low in ("qu", "qu'", "qu\u2019", "l", "l'", "l\u2019")
                _clitic_3pl = False
                # Override: clitic + on after ils/elles
                # "ils l on hissé" → "ils l'ont hissé"
                if _prev_is_fragment and i > 1:
                    _p2_low_ont = result[i - 2].lower()
                    if _p2_low_ont in ("ils", "elles"):
                        _prev_is_fragment = False
                        _clitic_3pl = True
                # Strong rule: "ils/elles on" → always "ont"
                _prev_3pl = prev_low in ("ils", "elles")
                # Guard: si le mot suivant est un verbe conjugue (pas un PP),
                # "on" est probablement le pronom sujet, pas l'auxiliaire "ont"
                # Ex: "commerçants on parlons" → garder "on"
                # -es = P2s (parles, manges), -eux = P1/P2 (peux, veux)
                _next_is_conjugated = (
                    next_pos in ("VER", "AUX")
                    and i + 1 < n
                    and result[i + 1].lower().endswith((
                        "ons", "ez", "ent", "ais", "ait", "aient",
                        "es", "eux",
                    ))
                )
                _prev_in_pp = False
                # "on lieu" → "ont lieu" (expression idiomatique)
                _next_ont_lieu = (
                    result[i + 1].lower() == "lieu"
                    and prev_pos in ("NOM", "NOM PROPRE", "???", "")
                )
                # "on été" is never valid French → always "ont été"
                _next_on_ete = (result[i + 1].lower() == "été")
                # "on" + ART/DET → "ont" (pronoun can't
                # precede article: "on un aspect" → "ont un")
                # Guard: require plural evidence before "on"
                # (NOM ending in s/x/z, or 3pl pronoun)
                # Without it, sentence-start "On les..." would FP
                _next_is_art = (
                    next_pos.startswith(("ART", "DET"))
                    and (
                        _prev_3pl
                        or _clitic_3pl
                        or (
                            prev_pos in ("NOM", "NOM PROPRE")
                            and prev_low.endswith(("s", "x", "z"))
                            and len(prev_low) > 2
                        )
                    )
                )
                if (
                    not _prev_is_fragment
                    and not _next_is_conjugated
                    and not _prev_in_pp
                    and (
                        _prev_3pl
                        or _clitic_3pl
                        or _next_ont_lieu
                        or _next_on_ete
                        or _next_is_art
                        or (
                            prev_pos in ("NOM", "NOM PROPRE", "PRO:per")
                            and (
                                # VER/AUX: require PP form (passe compose)
                                # "on voit" = pronom+VER, "on culminé" = ont+PP
                                # Guard: exclude present-tense-only verbs
                                # that match PP suffixes (voit, boit, doit...)
                                (
                                    next_pos in ("VER", "AUX")
                                    and result[i + 1].lower().endswith(
                                        _PP_SUFFIXES)
                                    and result[i + 1].lower()
                                    not in _PRESENT_ONLY_IT
                                )
                                # ADV: only with explicit 3pl subject
                                or (
                                    next_pos == "ADV"
                                    and prev_low in ("ils", "elles")
                                )
                                # ADV + PP: scan past adverb for PP form
                                # "esclandres on encore eu" → "ont encore eu"
                                or (
                                    next_pos == "ADV"
                                    and i + 2 < n
                                    and (pos_tags[i + 2] if i + 2 < len(pos_tags) else "")
                                    in ("VER", "AUX")
                                    and result[i + 2].lower().endswith(
                                        _PP_SUFFIXES)
                                    and result[i + 2].lower()
                                    not in _PRESENT_ONLY_IT
                                )
                            )
                        )
                        # ADJ (plural) used as nominal subject
                        # "espagnols on introduit" → "ont introduit"
                        or (
                            prev_pos in ("ADJ", "ADJ:pos")
                            and prev_low.endswith(("s", "x", "z"))
                            and len(prev_low) > 2
                            and (
                                (
                                    next_pos in ("VER", "AUX")
                                    and result[i + 1].lower().endswith(
                                        _PP_SUFFIXES)
                                    and result[i + 1].lower()
                                    not in _PRESENT_ONLY_IT
                                )
                                or (
                                    next_pos == "ADV"
                                    and i + 2 < n
                                    and (pos_tags[i + 2] if i + 2 < len(pos_tags) else "")
                                    in ("VER", "AUX")
                                    and result[i + 2].lower().endswith(
                                        _PP_SUFFIXES)
                                    and result[i + 2].lower()
                                    not in _PRESENT_ONLY_IT
                                )
                            )
                        )
                        # "qui on culminé" → "qui ont culminé"
                        # PRO:rel + on + PP = relative clause with avoir
                        or (
                            prev_pos == "PRO:rel"
                            and next_pos in ("VER", "AUX")
                            and result[i + 1].lower().endswith(_PP_SUFFIXES)
                            and result[i + 1].lower()
                            not in _PRESENT_ONLY_IT
                        )
                    )
                ):
                    ancien = result[i]
                    result[i] = "ont"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="ont",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.on_ont",
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
                regle="homophone.on_ont",
                explication="'ont' -> 'on' (pronom indefini)",
            ))
            continue

        # "ont" at sentence start → "on" (3pl auxiliary needs a subject)
        # "ont va partir" → "on va partir"
        if curr_low == "ont" and pos in ("AUX", "VER"):
            _is_sent_start_ont = (
                i == 0
                or (i > 0 and result[i - 1] in (".", "!", "?", ";"))
            )
            if _is_sent_start_ont:
                ancien = result[i]
                result[i] = "on"
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige="on",
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.on_ont",
                    explication="'ont' -> 'on' (debut de phrase sans sujet)",
                ))
                continue

        # "ont" + negation → "on" (negation goes BEFORE auxiliary:
        # "n'ont pas", never "ont ne"/"ont n'")
        # "ont ne sait rien" → "on ne sait rien"
        if curr_low == "ont" and pos in ("AUX", "VER"):
            if i + 1 < n:
                _next_ont = result[i + 1].lower()
                if _next_ont in ("ne", "n'", "n\u2019"):
                    ancien = result[i]
                    result[i] = "on"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="on",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.on_ont",
                        explication="'ont' -> 'on' (negation apres = pronom)",
                    ))
                    continue

        # "ont" (AUX) suivi d'un VER non-PP → probablement "on" + present
        # "ont trouve" → "on trouve", "ont les imagine" → "on les imagine"
        if curr_low == "ont" and pos in ("AUX", "VER"):
            # Chercher le verbe apres, en sautant les pronoms objets
            _verb_idx = None
            for _k in range(i + 1, min(n, i + 4)):
                _kw = result[_k].lower()
                if _kw in _PRONOMS_OBJETS:
                    continue
                _kp = pos_tags[_k] if _k < len(pos_tags) else ""
                if _kp in ("VER", "AUX"):
                    _verb_idx = _k
                break
            if _verb_idx is not None:
                _vw = result[_verb_idx].lower()
                # Si le verbe ne ressemble pas a un PP, c'est "on" + present
                # Guard: -aient/-ait = imparfait 3pl/3sg, sujet=ils/elles
                # "ont survivaient" = probably "ont + survécu" (broken grammar)
                if (
                    not _vw.endswith(_PP_SUFFIXES)
                    and not _vw.endswith(("er", "ir", "oir", "re"))
                    and not _vw.endswith(("aient", "ait"))
                ):
                    ancien = result[i]
                    result[i] = "on"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="on",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.on_ont",
                        explication="'ont' -> 'on' (suivi verbe non-PP)",
                    ))
                    continue

        # --- peu / peut ---
        # "peu" + infinitif → "peut" (pouvoir 3sg)
        # "on peu voir" → "on peut voir"
        # Guard: only after subject pronoun or NOM
        if curr_low == "peu" and pos == "ADV":
            if i + 1 < n and i > 0:
                _next_pos_peu = (
                    pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                )
                _next_low_peu = result[i + 1].lower()
                _prev_low_peu = result[i - 1].lower()
                _prev_pos_peu = (
                    pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                )
                # Next must be infinitive or "être"/"pas"
                _is_inf = (
                    _next_pos_peu in ("VER", "AUX")
                    and (
                        _next_low_peu.endswith(
                            ("er", "ir", "oir", "re")
                        )
                        or _next_low_peu in (
                            "être", "avoir", "faire",
                        )
                    )
                )
                # Also allow "peu" + "pas"/"plus" → "peut pas"
                _is_neg = _next_low_peu in ("pas", "plus")
                if (
                    (_is_inf or _is_neg)
                    and (
                        _prev_pos_peu in (
                            "PRO:ind", "PRO:per", "NOM",
                            "NOM PROPRE",
                        )
                        or _prev_low_peu in (
                            "on", "il", "elle", "qui",
                            "ne", "n'", "n\u2019",
                        )
                    )
                ):
                    ancien = result[i]
                    result[i] = "peut"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="peut",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.peu_peut",
                        explication="'peu' -> 'peut' (pouvoir + infinitif)",
                    ))
                    continue

        # --- ce / se ---
        # "ce" (DET ou PRO:dem) + VER/AUX -> "se" (pronom reflexif)
        # Guard: ne pas convertir "ce" + copule/modal (c'est, ce peut etre, etc.)
        if curr_low == "ce" and (pos.startswith("DET") or pos == "PRO:dem"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low = result[i + 1].lower()
                _ce_copule_modal = (
                    "est", "était", "sera", "serait",
                    "sont", "étaient", "fut", "fût",
                    "furent", "soit", "soient",
                    "seront", "serons", "serez",
                    "seraient", "serions", "seriez",
                    # Modaux: "ce peut etre", "ce doit etre"
                    "peut", "doit", "pourrait", "devrait",
                    "pouvait", "devait", "pourra", "devra",
                )
                # Guard: "de ce fait" = locution, not reflexive
                _is_de_ce_fait = (
                    next_low == "fait"
                    and i > 0
                    and result[i - 1].lower() in ("de", "du")
                )
                # Override: "faire ce peut" → "faire se peut" (idiom)
                _is_faire_se_peut = (
                    next_low == "peut"
                    and i > 0
                    and result[i - 1].lower() == "faire"
                )
                if (
                    next_pos in ("VER", "AUX")
                    and (
                        next_low not in _ce_copule_modal
                        or _is_faire_se_peut
                    )
                    and not _is_de_ce_fait
                ):
                    ancien = result[i]
                    result[i] = "se"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="se",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.ce_se",
                        explication="'ce' -> 'se' (pronom reflexif devant verbe)",
                    ))
                    continue
                # Extension: "ce" + NOM qui est aussi VER feminin
                # "ce rencontre" → "se rencontre" (rencontre=NOM_fem, VER)
                # Guard: le mot precedent doit etre un sujet (NOM/PRO)
                if (
                    next_pos == "NOM"
                    and lexique is not None
                    and i > 0
                ):
                    _prev_pos_ce = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                    if _prev_pos_ce in ("NOM", "NOM PROPRE", "ADJ",
                                        "PRO:per", "PRO:dem", "PRO:rel"):
                        _next_infos_ce = lexique.info(result[i + 1])
                        _has_ver_ce = any(
                            e.get("cgram") in ("VER", "AUX")
                            for e in _next_infos_ce
                        )
                        _nom_fem_only = (
                            any(
                                e.get("cgram") == "NOM" and e.get("genre") == "f"
                                for e in _next_infos_ce
                            )
                            and not any(
                                e.get("cgram") == "NOM" and e.get("genre") == "m"
                                for e in _next_infos_ce
                            )
                        )
                        if _has_ver_ce and _nom_fem_only:
                            ancien = result[i]
                            result[i] = "se"
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige="se",
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="homophone.ce_se",
                                explication="'ce' -> 'se' (NOM feminin = verbe)",
                            ))
                            continue

        # "se" + copule (sont/est) → "ce" (demonstratif)
        # "se sont les seules" → "ce sont les seules"
        # Guard: si mot apres copule est PP/VER, c'est reflexif (se sont écoulés)
        if curr_low == "se" and pos == "PRO:per":
            if i + 1 < n:
                next_low = result[i + 1].lower()
                if next_low in ("sont", "est", "sera", "serait",
                                "était", "étaient", "fut", "furent"):
                    _after_cop = i + 2
                    _is_reflexive = False
                    # Skip ADV (également, aussi, toujours, etc.)
                    while (
                        _after_cop < n
                        and (pos_tags[_after_cop] if _after_cop < len(pos_tags) else "") == "ADV"
                    ):
                        _after_cop += 1
                    if _after_cop < n:
                        _ac_pos = pos_tags[_after_cop] if _after_cop < len(pos_tags) else ""
                        _ac_low = result[_after_cop].lower()
                        if _ac_pos in ("VER", "AUX") or _ac_low.endswith(_PP_SUFFIXES):
                            _is_reflexive = True
                    if not _is_reflexive:
                        ancien = result[i]
                        result[i] = "ce"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="ce",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.ce_se",
                            explication="'se' -> 'ce' (demonstratif devant copule)",
                        ))
                        continue

        # "se" (PRO:per) + NOM/ADJ -> "ce" (determinant)
        # Guard: si le mot suivant est aussi VER dans le lexique, c'est un
        # infinitif (se faire, se dire, etc.) et "se" est correct
        # Exception: PRE + "se" = impossible reflexif (avec se titre → ce)
        if curr_low == "se" and pos == "PRO:per":
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                # Accept ADV-tagged words that also have ADJ/NOM entry
                # (e.g. "nouveau" mis-tagged ADV but is really ADJ)
                _eff_nom_adj = next_pos in ("NOM", "ADJ", "ADJ:pos")
                if (
                    not _eff_nom_adj
                    and next_pos == "ADV"
                    and lexique is not None
                ):
                    _ni_adv = lexique.info(result[i + 1])
                    if any(
                        (e.get("cgram") or "").startswith(("ADJ", "NOM"))
                        for e in _ni_adv
                    ):
                        _eff_nom_adj = True
                _next_also_ver = False
                if _eff_nom_adj and lexique is not None:
                    _ni = lexique.info(result[i + 1])
                    _next_also_ver = any(
                        e.get("cgram", "").startswith("VER")
                        or e.get("cgram", "").startswith("AUX")
                        for e in _ni
                    )
                # Override: PRE before "se" → reflexive impossible
                # "avec se titre" → "ce titre", "de se centre" → "ce"
                # Exception: PRE + se + infinitif = reflexif valide
                # "de se faire" → garder "se", "pour se cacher" → garder "se"
                if _next_also_ver and i > 0:
                    _prev_pos_se = (
                        pos_tags[i - 1]
                        if i - 1 < len(pos_tags) else ""
                    )
                    _prev_low_se = result[i - 1].lower()
                    if (
                        _prev_pos_se == "PRE"
                        or _prev_low_se in PREPOSITIONS
                        or _prev_low_se in (
                            "du", "des", "aux", "au",
                        )
                    ):
                        # Check if next word is infinitive in lexique
                        # If so, PRE + se + INF is valid reflexive
                        _next_low_se = result[i + 1].lower()
                        _is_inf_se = False
                        if lexique is not None and hasattr(lexique, "info"):
                            _ni_inf = lexique.info(_next_low_se)
                            _is_inf_se = any(
                                e.get("cgram") in ("VER", "AUX")
                                and e.get("mode") in (
                                    "infinitif", "Inf", "inf",
                                )
                                for e in _ni_inf
                            ) if _ni_inf else False
                        if not _is_inf_se:
                            _next_also_ver = False
                if _eff_nom_adj and not _next_also_ver:
                    ancien = result[i]
                    result[i] = "ce"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="ce",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.ce_se",
                        explication="'se' -> 'ce' (determinant devant nom/adj)",
                    ))
                    continue

        # "se" + qui/que → "ce" (demonstratif relatif)
        # "se qui mene" → "ce qui mene", "se que madonna" → "ce que madonna"
        if curr_low == "se" and pos == "PRO:per":
            if i + 1 < n:
                _next_se = result[i + 1].lower()
                if _next_se in ("qui", "que", "qu'"):
                    ancien = result[i]
                    result[i] = "ce"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="ce",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.ce_se",
                        explication="'se' -> 'ce' (demonstratif relatif)",
                    ))
                    continue

        # --- la / là ---
        # "la" etiquete ADV -> "là" (adverbe de lieu)
        # Guard: ne pas re-accentuer si l'original etait deja "là" (ortho a corrige)
        if curr_low == "la" and pos == "ADV":
            _orig_low = originaux[i].lower() if originaux and i < len(originaux) else ""
            if _orig_low != "là":
                ancien = result[i]
                result[i] = "là"
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige="là",
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.la_la",
                    explication="'la' -> 'là' (adverbe de lieu)",
                ))
                continue

        # "là" etiquete ART/DET -> "la" (article)
        if curr_low == "là" and pos in ("ART:def", "ART:ind", "DET", "ART"):
            ancien = result[i]
            result[i] = "la"
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige="la",
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="homophone.la_la",
                explication="'là' -> 'la' (article)",
            ))
            continue

        # "la" (ART) apres etre/habiter/rester/vivre → "là" (adverbe)
        # Pattern 1: VER + "la" + PRE/ADV/fin → "là"
        # "je suis la depuis" → "je suis là depuis"
        # Pattern 2: VER + "la" at end → "là"
        # "il habite la" → "il habite là"
        _VERBES_LOCATIFS = frozenset({
            "suis", "es", "est", "sommes", "êtes", "sont",
            "étais", "étais", "était", "étions", "étiez", "étaient",
            "serai", "seras", "sera", "serons", "serez", "seront",
            "habite", "habites", "habitent", "habitons", "habitez",
            "reste", "restes", "restent", "restons", "restez",
            "vis", "vit", "vivons", "vivez", "vivent",
            "va", "vas", "vais", "allons", "allez", "vont",
        })
        if curr_low == "la" and pos in ("ART", "ART:def", "PRO:per"):
            if i > 0:
                _prev_low_la = result[i - 1].lower()
                _is_locatif = _prev_low_la in _VERBES_LOCATIFS
                if _is_locatif:
                    _la_at_end = (i == n - 1)
                    _la_before_prep_adv = False
                    if i + 1 < n:
                        _npos_la = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                        _la_before_prep_adv = _npos_la in ("PRE", "ADV")
                    if _la_at_end or _la_before_prep_adv:
                        ancien = result[i]
                        result[i] = "là"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="là",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.la_la",
                            explication="'la' -> 'là' (adverbe apres verbe locatif)",
                        ))
                        continue

        # --- leur / leurs ---
        # "leurre(s)" + NOM/ADJ/VER → "leur" (possessif/pronom)
        # "leurres ancien emplacement" → "leur ancien emplacement"
        # "leurre tournée" → "leur tournée"
        # Guard: preceded by DET → valid NOM "un leurre" (keep)
        if curr_low in ("leurre", "leurres") and pos == "NOM":
            if i + 1 < n:
                _next_pos_lr = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _prev_pos_lr = pos_tags[i - 1] if i > 0 and i - 1 < len(pos_tags) else ""
                if (
                    _next_pos_lr in ("NOM", "ADJ", "ADJ:pos", "VER", "AUX")
                    and not _prev_pos_lr.startswith(("ART", "DET"))
                ):
                    ancien = result[i]
                    result[i] = "leur"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="leur",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.leur_leurs",
                        explication="'leurre(s)' -> 'leur' (possessif/pronom)",
                    ))
                    continue

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
                            regle="homophone.leur_leurs",
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
                            regle="homophone.leur_leurs",
                            explication="'leurs' -> 'leur' (NOM singulier)",
                        ))
                        continue

        # "leurs" + VER → "leur" (pronom COI, pas possessif)
        # "nationaux leurs seraient" → "leur seraient"
        if curr_low == "leurs" and pos in ("DET", "ADJ:pos"):
            if i + 1 < n:
                _next_pos_lr = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if _next_pos_lr in ("VER", "AUX"):
                    ancien = result[i]
                    result[i] = "leur"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="leur",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.leur_leurs",
                        explication="'leurs' -> 'leur' (pronom COI devant verbe)",
                    ))
                    continue

        # "leurs" + ADJ singulier + NOM singulier → "leur"
        # "leurs unique indice" → "leur unique indice"
        if curr_low == "leurs" and pos in ("DET", "ADJ:pos"):
            if i + 1 < n and i + 2 < n:
                _next_pos_la = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if _next_pos_la in ("ADJ",):
                    _next2_pos_la = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                    _next2_low = result[i + 2].lower()
                    if (
                        _next2_pos_la == "NOM"
                        and not _next2_low.endswith(("s", "x", "z"))
                    ):
                        ancien = result[i]
                        result[i] = "leur"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="leur",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.leur_leurs",
                            explication="'leurs' -> 'leur' (ADJ + NOM singulier)",
                        ))
                        continue

        # --- ça / sa ---
        # "sa" + VER -> probablement "ça"
        # Guard: skip if next word is also NOM in lexique (tagger ambiguity)
        if curr_low == "sa" and pos in ("ADJ:pos", "DET"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos in ("VER", "AUX"):
                    _next_also_nom = False
                    if lexique is not None:
                        _next_infos = lexique.info(result[i + 1])
                        _next_also_nom = any(
                            e.get("cgram", "").startswith("NOM")
                            for e in _next_infos
                        )
                    if not _next_also_nom:
                        ancien = result[i]
                        result[i] = "ça"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="ça",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="homophone.ca_sa",
                            explication="'sa' -> 'ça' (suivi d'un verbe)",
                        ))
                        continue

        # "ça" + NOM/ADJ -> probablement "sa" (possessif)
        # "de ça marraine" → "de sa marraine", "ça deuxième année" → "sa deuxième année"
        if curr_low in ("ça", "ca") and pos in ("PRO:dem", "PRO:ind", "PRO"):
            if i + 1 < n:
                next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if next_pos in ("NOM", "ADJ", "ADJ:num"):
                    ancien = result[i]
                    result[i] = "sa"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="sa",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.ca_sa",
                        explication="'ça' -> 'sa' (suivi d'un nom/adj)",
                    ))
                    continue

        # --- peux / peu / peut ---
        # "peux connu" → "peu connu" (adverbe de degre)
        # "il peux" → "il peut" (conjugaison P3s)
        # "elle peux pivoter" → "elle peut pivoter" (modal + infinitif)
        # Guard: "je peux", "tu peux" = verbe (correct)
        if curr_low == "peux" and pos in ("VER", "AUX"):
            _peux_target = None  # "peu" ou "peut"
            _peux_next_pos = pos_tags[i + 1] if i + 1 < n and i + 1 < len(pos_tags) else ""
            # Chercher le pronom sujet en remontant (sauter ne/n'/y)
            _peux_subj = None
            for _kp in range(i - 1, max(-1, i - 4), -1):
                _kp_low = result[_kp].lower()
                if _kp_low in ("ne", "n'", "n\u2019", "y", "se", "s'", "s\u2019"):
                    continue
                _kp_low = _kp_low.rstrip("''\u2019")
                if _kp_low in ("je", "tu", "j"):
                    _peux_subj = "12"
                elif _kp_low in ("il", "elle", "on"):
                    _peux_subj = "3"
                break
            # Guard: plural nominal subject → leave for conjugaison Rule 3
            # "les hélicoptères peux" → "peuvent" (not "peu")
            _peux_plur_subj = False
            if _peux_subj is None and i > 0:
                _pp_prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _pp_prev_low = result[i - 1].lower()
                if _pp_prev_pos == "NOM" and _pp_prev_low.endswith(("s", "x", "z")):
                    _peux_plur_subj = True
            if _peux_subj == "12" or _peux_plur_subj:
                pass  # correct: je/tu peux OR plural subject → conjugaison
            elif _peux_subj == "3":
                # 3e personne → "peut" (conjugaison)
                _peux_target = "peut"
            elif _peux_next_pos in ("VER", "AUX"):
                # peux + VER/AUX: "peut" si infinitif, "peu" si PP
                _peux_next_low = result[i + 1].lower() if i + 1 < n else ""
                _is_infinitif = _peux_next_low.endswith((
                    "er", "ir", "re", "oir", "être",
                ))
                if _is_infinitif:
                    _peux_target = "peut"
                else:
                    _peux_target = "peu"
            elif _peux_next_pos in ("ADJ", "ADV"):
                # peux + ADJ/ADV = "peu" (adverbe de degre)
                _peux_target = "peu"
            if _peux_target:
                ancien = result[i]
                result[i] = _peux_target
                _peux_expl = (
                    "'peux' -> 'peut' (conjugaison P3s)"
                    if _peux_target == "peut"
                    else "'peux' -> 'peu' (adverbe de degre)"
                )
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=_peux_target,
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.peux_peut",
                    explication=_peux_expl,
                ))
                continue

        # --- tout / tous ---
        # "tout" + ART:def pluriel "les" → "tous" (quantificateur pluriel)
        # "tout les enfants" → "tous les enfants"
        # Guard: "tout le" (singulier) = correct ("tout le monde")
        if curr_low == "tout" and pos in ("PRO:ind", "ADJ:ind", "ADV"):
            if i + 1 < n:
                _next_low_tout = result[i + 1].lower()
                if _next_low_tout == "les":
                    ancien = result[i]
                    result[i] = "tous"
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige="tous",
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="homophone.tout_tous",
                        explication="'tout' -> 'tous' (quantificateur + les)",
                    ))
                    continue

        # --- mes / mais ---
        # "mes" (ADJ:pos) en contexte adversatif → "mais"
        # Contextes: ADJ + mes (incombustible mes onéreux),
        # NOM + mes + PRO:per/ADV/VER (pont-euxin mes également),
        # VER/PP + mes + PRO/NOM (créé mes pittier)
        # Guard: "mes" devant NOM = possessif correct (mes chats)
        if curr_low == "mes" and pos in ("ADJ:pos", "DET", "ADJ"):
            _mes_is_mais = False
            if i + 1 < n:
                next_pos_m = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                next_low_m = result[i + 1].lower()
                # "mes" + ADJ/ADV = adversatif "mais onéreux", "mais également"
                if next_pos_m in ("ADJ", "ADV"):
                    _mes_is_mais = True
                # "mes" + PRO:per/VER = adversatif "mais il est"
                elif next_pos_m in ("PRO:per", "PRO:dem", "VER", "AUX"):
                    _mes_is_mais = True
                # "mes" + NOM PROPRE = adversatif "mais Pierre"
                elif next_pos_m == "NOM PROPRE":
                    _mes_is_mais = True
                # "mes" + ART/DET = impossible possessif → "mais"
                # "mes la performance", "mes les cardinaux"
                elif next_pos_m.startswith(("ART", "DET")):
                    _mes_is_mais = True
                # "mes" + PRE = impossible possessif → "mais"
                # "mes avec retenue", "mes pour trouvé"
                elif next_pos_m == "PRE":
                    _mes_is_mais = True
                # "mes" + CON = impossible possessif → "mais"
                # "mes comme le pouvoir"
                elif next_pos_m == "CON":
                    _mes_is_mais = True
                # "mes" + NOM = possessif (correct) → pas de correction
            if _mes_is_mais:
                ancien = result[i]
                result[i] = "mais"
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige="mais",
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="homophone.mes_mais",
                    explication="'mes' -> 'mais' (conjonction adversative)",
                ))
                continue

        # --- -er/-é apres preposition/aller/faire ---
        # PRE/ALLER/FAIRE + mot en -é -> -er (infinitif requis)
        if i > 0 and pos == "VER" and curr_low.endswith("é") and not curr_low.endswith("er"):
            prev_low = result[i - 1].lower()
            prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            # Rule 1: preposition pour/de/à + PP → infinitif
            _is_prep_inf = (
                prev_low in ("pour", "de", "d'", "d\u2019", "à", "a")
                and prev_pos == "PRE"
            )
            # Rule 2: faire/laisser/voir/entendre + PP → infinitif
            _is_causative = prev_low in (
                "fait", "faire", "fais", "font", "fit",
                "laisser", "laissé", "laisse",
                "vu", "voir", "voit", "voyait",
                "entendu", "entendre",
            ) or prev_low.endswith("ant") and prev_pos == "VER"
            # Guard: if the word is also a NOM in the lexique,
            # don't convert after de/à (de + NOM is very common: "de combiné")
            _also_nom = False
            if _is_prep_inf and prev_low in ("de", "d'", "d\u2019", "à", "a"):
                if lexique is not None:
                    _infos = lexique.info(curr_low)
                    _also_nom = any(
                        e.get("cgram") == "NOM" for e in _infos
                    )
            if (
                (prev_low in ALLER_FORMES or _is_prep_inf or _is_causative)
                and not _also_nom
            ):
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
                        regle="homophone.pp_infinitif",
                        explication="PP -> infinitif apres aller/PRE/faire",
                    ))
                    continue

        # --- listent / liste ---
        # "listent" (VER 3pl) before ART/DET → "liste" (NOM)
        # "listent des aéroports" → "liste des aéroports"
        # Guard: preceded by plural subject (ils/elles/NOM_plur) → keep verb
        if curr_low == "listent" and pos in ("VER", "AUX"):
            if i + 1 < n:
                _next_pos_li = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if _next_pos_li in (
                    "ART:def", "ART:ind", "ART", "DET", "PRE",
                ):
                    _has_plur_subj_li = False
                    if i > 0:
                        _prev_low_li = result[i - 1].lower()
                        _prev_pos_li = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                        if _prev_low_li in ("ils", "elles"):
                            _has_plur_subj_li = True
                        elif (
                            _prev_pos_li == "NOM"
                            and _prev_low_li.endswith(("s", "x", "z"))
                        ):
                            _has_plur_subj_li = True
                    if not _has_plur_subj_li:
                        ancien = result[i]
                        result[i] = "liste"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="liste",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="accord.det_nom_ver",
                            explication="'listent' -> 'liste' (nom)",
                        ))
                        continue

        # --- entrent / entre ---
        # "entrent" (VER 3pl) before ART/DET/NOM PROPRE → "entre" (preposition)
        # "entrent les deux" → "entre les deux", "entrent barby" → "entre barby"
        # Guard: preceded by plural subject (ils/elles/NOM_plur) → keep verb
        if curr_low == "entrent" and pos in ("VER", "AUX"):
            if i + 1 < n:
                _next_pos_ent = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if _next_pos_ent in (
                    "ART:def", "ART:ind", "ART", "DET", "NOM PROPRE",
                    "ADJ:num",  # "entre deux" (numeral)
                ):
                    # Guard: plural subject before → keep as verb
                    _has_plur_subj_ent = False
                    if i > 0:
                        _prev_low_ent = result[i - 1].lower()
                        _prev_pos_ent = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                        if _prev_low_ent in ("ils", "elles"):
                            _has_plur_subj_ent = True
                        elif (
                            _prev_pos_ent == "NOM"
                            and _prev_low_ent.endswith(("s", "x", "z"))
                        ):
                            _has_plur_subj_ent = True
                    if not _has_plur_subj_ent:
                        ancien = result[i]
                        result[i] = "entre"
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige="entre",
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="accord.det_nom_ver",
                            explication="'entrent' -> 'entre' (preposition)",
                        ))
                        continue

    return result, corrections
