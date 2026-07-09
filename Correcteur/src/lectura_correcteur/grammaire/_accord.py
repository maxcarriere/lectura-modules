"""Regles d'accord : determinant+nom, determinant+adjectif+nom, etc.

Correspond aux regles 0, 1, 2, 4 du POC.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    COPULES_ALL,
    COPULES_PLURIEL,
    COPULES_SINGULIER,
    DET_GENRE_MAP,
    ETRE_FORMES,
    INVARIABLES,
    PLUR_DET,
    PREPOSITIONS,
    SING_DET,
    SING_FEM_DET,
    SING_MASC_DET,
    SUJETS_3PL,
    generer_candidats_3pl,
    generer_candidats_feminin,
    generer_candidats_masculin,
    generer_candidats_pluriel,
    generer_candidats_singulier_nom,
    trouver_sujet_genre_nombre,
)

_TRANSPARENTS = frozenset({
    "ne", "n'", "pas", "plus", "jamais", "rien", "point", "y", "en",
})

# Mots-outils a ne JAMAIS modifier par accord.
# Uniquement les mots qui sont prepositions/conjonctions 99%+ du temps
# et dont la forme "accordee" est un faux ami (sur→surs, peut→peux, etc.)
_ACCORD_EXCLUS = frozenset({
    # Prepositions pures
    "sur", "sous", "par", "pour", "avec", "dans", "chez", "sans",
    "entre", "vers", "contre", "depuis", "pendant", "avant", "après",
    "de", "du",
    # Conjonctions
    "et", "ou", "mais", "donc", "car", "ni",
    # Pronoms qui ne doivent pas etre modifies par accord
    "se", "ce", "on",
    # leur/leurs : distinction pronom COI (invariable) vs possessif (accord
    # avec le nom possede) est un probleme d'homophonie, pas d'accord.
    "leur",
    # Verbes courants pris pour ADJ
    "peut", "sont",
    # Participes presents / gerondifs (invariables)
    "étant", "ayant",
    # Adverbes invariables parfois tagues ADJ
    # "même" retire : s'accorde comme ADJ ("les mêmes règles")
})

# Articles definis : ne pas corriger le→la ou la→le par accord genre
# (ces erreurs sont quasi inexistantes dans les corpus reels, et la regle
# produit beaucoup de FP quand le NOM est ambigu en genre)
_DET_GENRE_EXCLUS = frozenset({"le", "la", "l'"})


def verifier_accords(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
    *,
    pm_guidance=None,
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

    # Positions deja corrigees par le module PM
    _pm_handled: set[int] = set()
    if pm_guidance:
        _pm_handled = {g.index for g in pm_guidance}

    origs = originaux if originaux else mots
    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    for i in range(n):
        if i in _pm_handled:
            continue
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]
        curr_low = curr.lower()

        # Regle 11c (early) : "ce" + NOM feminin → "cette"
        # Placed BEFORE _ACCORD_EXCLUS guard since "ce" is excluded
        # "ce saison" → "cette saison", "ce espèce" → "cette espèce"
        if curr_low == "ce" and pos in ("ADJ:dem", "DET:dem", "DET", "ADJ", "PRO:dem"):
            if i + 1 < n and lexique is not None:
                _next_11c = result[i + 1].lower()
                _next_pos_11c = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if (
                    _next_pos_11c == "NOM"
                    and len(_next_11c) > 2
                ):
                    _infos_11c = lexique.info(result[i + 1])
                    # Use multext[2] for gender (genre field is unreliable in v4)
                    _nom_genres_11c = [
                        e.get("multext", "")[2:3] for e in _infos_11c
                        if e.get("cgram") == "NOM" and len(e.get("multext", "")) >= 3
                    ]
                    if _nom_genres_11c and all(g == "f" for g in _nom_genres_11c):
                        ancien = result[i]
                        result[i] = transferer_casse(curr, "cette")
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=result[i],
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="accord.genre_det",
                            explication="'ce' -> 'cette' (NOM feminin)",
                        ))

        # Regle 11d : Possessif + NOM → accord genre
        # "son mère" → "sa mère", "sa père" → "son père"
        # + euphonie : "sa école" → "son école", "ma amie" → "mon amie"
        _POSS_MASC = {"son": "sa", "mon": "ma", "ton": "ta"}
        _POSS_FEM = {"sa": "son", "ma": "mon", "ta": "ton"}
        if curr_low in _POSS_MASC or curr_low in _POSS_FEM:
            if i + 1 < n and lexique is not None:
                # Find the NOM (direct or through intercalated ADJ)
                _nom_idx_r11d = None
                _next_pos_r11d = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                if _next_pos_r11d == "NOM":
                    _nom_idx_r11d = i + 1
                elif _next_pos_r11d in ("ADJ", "ADJ:pos") and i + 2 < n:
                    _next2_pos_r11d = (
                        pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
                    )
                    if _next2_pos_r11d == "NOM":
                        _nom_idx_r11d = i + 2
                # Use immediate next for euphonic check
                _next_r11d = result[i + 1].lower()
                _nom_infos_r11d = (
                    lexique.info(result[_nom_idx_r11d])
                    if _nom_idx_r11d is not None else []
                )
                _nom_only_r11d = [
                    e for e in _nom_infos_r11d
                    if e.get("cgram") == "NOM"
                ]
                if _nom_only_r11d:
                    _nom_genres_r11d = {
                        e.get("genre") for e in _nom_only_r11d
                        if e.get("genre")
                    }
                    _starts_vowel = (
                        _next_r11d
                        and _next_r11d[0] in "aeiouyàâéèêëïîôùûüæœh"
                    )
                    _new_poss = None
                    # Case 1: euphonie — sa/ma/ta + voyelle → son/mon/ton
                    if curr_low in _POSS_FEM and _starts_vowel:
                        _new_poss = _POSS_FEM[curr_low]
                    # Case 2: son/mon/ton + NOM fem (consonant) → sa/ma/ta
                    elif (
                        curr_low in _POSS_MASC
                        and not _starts_vowel
                        and len(_nom_genres_r11d) == 1
                        and "f" in _nom_genres_r11d
                    ):
                        _new_poss = _POSS_MASC[curr_low]
                    # Case 3: sa/ma/ta + NOM masc (consonant) → son/mon/ton
                    elif (
                        curr_low in _POSS_FEM
                        and not _starts_vowel
                        and len(_nom_genres_r11d) == 1
                        and "m" in _nom_genres_r11d
                    ):
                        _new_poss = _POSS_FEM[curr_low]
                    if _new_poss and _new_poss != curr_low:
                        ancien = result[i]
                        result[i] = transferer_casse(curr, _new_poss)
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=result[i],
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="accord.genre_possessif",
                            explication=(
                                f"'{ancien}' -> '{result[i]}'"
                                f" (NOM '{result[_nom_idx_r11d]}')"
                            ),
                        ))

        # Guard : ne jamais modifier les mots-outils par accord
        if curr_low in _ACCORD_EXCLUS:
            continue

        # Regle 0 : Restaurer ils/elles si corrige en il/elle par erreur
        # Guard: ne pas restaurer si le verbe suivant est un auxiliaire
        # suppletif 3sg (a/est/va/fait/peut) dont le 3pl est different;
        # c'est le pronom qui est faux, pas le verbe.
        orig_low = origs[i].lower() if i < len(origs) else ""
        if orig_low in SUJETS_3PL and curr_low in ("il", "elle"):
            _skip_restore = False
            if i + 1 < n:
                _nw0_low = result[i + 1].lower()
                if _nw0_low in (
                    "a", "est", "va", "fait", "peut", "sait", "veut",
                    "doit", "avait", "était", "devait", "pouvait",
                    "sera", "fera", "ira", "aura", "fut",
                ):
                    _skip_restore = True
            if not _skip_restore:
                ancien = curr
                result[i] = transferer_casse(curr, origs[i])
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.pronom",
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
                # Guard: le/la — ne pas flipper le DET sauf si NOM unambigu
                # et ADJ intercale confirme le desaccord
                if curr_low in _DET_GENRE_EXCLUS:
                    if adj_idx is not None:
                        nom_infos_ex = lexique.info(result[nom_idx])
                        nom_only_ex = [
                            e for e in nom_infos_ex
                            if e.get("cgram", "").startswith("NOM")
                        ]
                        if not nom_only_ex:
                            nom_only_ex = nom_infos_ex
                        nom_genres_ex = {
                            e.get("genre") for e in nom_only_ex
                            if e.get("genre")
                        }
                        det_genre = (
                            "m" if curr_low in SING_MASC_DET else "f"
                        )
                        adj_infos_ex = lexique.info(result[adj_idx])
                        adj_genred_ex = [
                            e for e in adj_infos_ex if e.get("genre")
                        ]
                        adj_genre_ex = {
                            e.get("genre") for e in adj_genred_ex
                        } if adj_genred_ex else set()

                        # Fix 1: DET+NOM agree, ADJ disagrees → correct ADJ
                        if (
                            det_genre in nom_genres_ex
                            and adj_genre_ex
                            and det_genre not in adj_genre_ex
                        ):
                            target = det_genre
                            gen_fn = (
                                generer_candidats_masculin
                                if target == "m"
                                else generer_candidats_feminin
                            )
                            for cand in gen_fn(result[adj_idx]):
                                c_infos = lexique.info(cand)
                                if c_infos and any(
                                    e.get("genre") == target
                                    for e in c_infos
                                ):
                                    ancien = result[adj_idx]
                                    result[adj_idx] = cand
                                    corrections.append(Correction(
                                        index=adj_idx,
                                        original=ancien,
                                        corrige=cand,
                                        type_correction=TypeCorrection.GRAMMAIRE,
                                        regle="accord.genre_adj",
                                        explication=(
                                            f"ADJ→{target} "
                                            f"(DET+NOM '{result[nom_idx]}' "
                                            f"sont {'masculins' if target == 'm' else 'feminins'})"
                                        ),
                                    ))
                                    break

                        # Fix 2: NOM unambigu, DET+ADJ both wrong → flip DET + correct ADJ
                        elif (
                            len(nom_genres_ex) == 1
                            and det_genre not in nom_genres_ex
                        ):
                            nom_g = next(iter(nom_genres_ex))
                            # Flip DET
                            new_det = DET_GENRE_MAP.get(curr_low)
                            if new_det:
                                ancien_det = result[i]
                                result[i] = transferer_casse(curr, new_det)
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien_det,
                                    corrige=result[i],
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    regle="accord.genre_det",
                                    explication=(
                                        f"DET {det_genre}→{nom_g} "
                                        f"(NOM '{result[nom_idx]}' unambigu)"
                                    ),
                                ))
                                curr = result[i]
                                curr_low = curr.lower()
                            # Correct ADJ if also wrong genre
                            if adj_genre_ex and nom_g not in adj_genre_ex:
                                gen_fn2 = (
                                    generer_candidats_masculin
                                    if nom_g == "m"
                                    else generer_candidats_feminin
                                )
                                for cand in gen_fn2(result[adj_idx]):
                                    c_infos = lexique.info(cand)
                                    if c_infos and any(
                                        e.get("genre") == nom_g
                                        for e in c_infos
                                    ):
                                        ancien_adj = result[adj_idx]
                                        result[adj_idx] = cand
                                        corrections.append(Correction(
                                            index=adj_idx,
                                            original=ancien_adj,
                                            corrige=cand,
                                            type_correction=TypeCorrection.GRAMMAIRE,
                                            regle="accord.genre_adj",
                                            explication=(
                                                f"ADJ→{nom_g} "
                                                f"(NOM '{result[nom_idx]}' unambigu)"
                                            ),
                                        ))
                                        break
                    # Fallback: le/la + NOM (no adj intercalé)
                    # Correct if NOM is unambiguous high-freq
                    # "le table" → "la table", "la soleil" → "le soleil"
                    else:
                        _nom_infos_le = lexique.info(result[nom_idx])
                        _nom_only_le = [
                            e for e in _nom_infos_le
                            if e.get("cgram", "").startswith("NOM")
                        ]
                        if _nom_only_le:
                            _nom_genres_le = {
                                e.get("genre") for e in _nom_only_le
                                if e.get("genre")
                            }
                            _nom_freq_le = max(
                                float(e.get("freq") or 0)
                                for e in _nom_only_le
                            )
                            _det_g_le = (
                                "m" if curr_low in SING_MASC_DET else "f"
                            )
                            if (
                                len(_nom_genres_le) == 1
                                and _det_g_le not in _nom_genres_le
                                and _nom_freq_le > 10.0
                            ):
                                _new_det_le = DET_GENRE_MAP.get(curr_low)
                                if _new_det_le:
                                    ancien = result[i]
                                    result[i] = transferer_casse(
                                        curr, _new_det_le,
                                    )
                                    corrections.append(Correction(
                                        index=i,
                                        original=ancien,
                                        corrige=result[i],
                                        type_correction=TypeCorrection.GRAMMAIRE,
                                        regle="accord.genre_det",
                                        explication=(
                                            f"DET {_det_g_le}→"
                                            f"{next(iter(_nom_genres_le))} "
                                            f"(NOM '{result[nom_idx]}')"
                                        ),
                                    ))
                                    curr = result[i]
                                    curr_low = curr.lower()
                else:
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
                                regle="accord.genre_det",
                                explication=f"DET masc→fem (NOM '{result[nom_idx]}' est feminin)",
                            ))
                            curr = result[i]
                            curr_low = curr.lower()

                    elif curr_low in SING_FEM_DET and "m" in nom_genres and "f" not in nom_genres:
                        # DET fem + NOM masc → corriger le DET
                        new_det = DET_GENRE_MAP.get(curr_low)
                        if new_det:
                            ancien = result[i]
                            result[i] = transferer_casse(curr, new_det)
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige=result[i],
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="accord.genre_det",
                                explication=f"DET fem→masc (NOM '{result[nom_idx]}' est masculin)",
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
                                        regle="accord.genre_adj",
                                        explication=f"ADJ fem→masc (DET+NOM '{result[nom_idx]}' sont masculins)",
                                    ))
                                    break

        # Regle 1 : Det. pluriel -> NOM/ADJ doit avoir -s/-x
        # Extension : chiffre >= 2 traite comme declencheur pluriel
        # Extension : VER tague par erreur mais ayant des entrees NOM dans le lexique
        _is_nom_or_adj = pos in ("NOM", "ADJ")
        # Detecter si le mot precedent est un chiffre >= 2
        _prev_is_digit_plur = False
        if i > 0:
            _prev_raw = result[i - 1]
            if _prev_raw.isdigit():
                try:
                    _prev_is_digit_plur = int(_prev_raw) >= 2
                except ValueError:
                    pass
        if (
            not _is_nom_or_adj
            and pos in ("VER", "AUX")
            and i > 0
            and (result[i - 1].lower() in PLUR_DET or _prev_is_digit_plur)
            and lexique is not None
        ):
            _r1_infos = lexique.info(curr)
            _r1_has_nom = _r1_infos and any(
                e.get("cgram") == "NOM" for e in _r1_infos
            )
            # Guard: if VER/AUX freq >> NOM freq, keep as VER
            # (e.g. "les avoir" → "avoir" is AUX, not NOM)
            if _r1_has_nom:
                _r1_nom_freq = max(
                    (float(e.get("freq", 0) or 0)
                     for e in _r1_infos if e.get("cgram") == "NOM"),
                    default=0,
                )
                _r1_ver_freq = max(
                    (float(e.get("freq", 0) or 0)
                     for e in _r1_infos
                     if e.get("cgram") in ("VER", "AUX")),
                    default=0,
                )
                # Guard: if VER freq is substantial (>=50),
                # the tagger is likely right → keep as VER
                # (e.g. "les appelle" = verb, VER freq=144)
                if (
                    _r1_ver_freq <= 5 * _r1_nom_freq
                    and _r1_ver_freq < 50
                ):
                    _is_nom_or_adj = True
        if i > 0 and _is_nom_or_adj:
            prev_low = result[i - 1].lower()
            if (
                (prev_low in PLUR_DET or _prev_is_digit_plur)
                and not curr_low.endswith(("s", "x", "z"))
                and len(curr) > 1
                and curr_low not in INVARIABLES
            ):
                # Guard: si un verbe singulier suit le NOM (1-2 pos),
                # le sujet est probablement singulier → ne pas pluraliser
                # Ex: "ces disposition était" → NOM + VER_sing = sujet sing
                _verb_sing_after = False
                _VERBES_SING = frozenset({
                    "est", "a", "était", "avait", "fut", "sera",
                    "aura", "va", "fait", "peut", "doit",
                })
                for _look in range(i + 1, min(i + 3, n)):
                    _look_low = result[_look].lower()
                    _look_pos = pos_tags[_look] if _look < len(pos_tags) else ""
                    if _look_low in _VERBES_SING and _look_pos in ("VER", "AUX"):
                        _verb_sing_after = True
                        break
                    # Stop scanning at NOM/VER boundary
                    if _look_pos in ("VER", "AUX", "NOM", "CON"):
                        break
                if not _verb_sing_after:
                    # Fix 3a : ADJ antepose + NOM feminin → feminiser avant pluraliser
                    if pos == "ADJ" and lexique is not None and i + 1 < n:
                        next_pos = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                        if next_pos == "NOM":
                            nom_infos = lexique.info(result[i + 1])
                            # Require UNAMBIGUOUS feminine (all NOM entries)
                            _nom_entries_3a = [
                                e for e in nom_infos
                                if e.get("cgram") == "NOM"
                            ]
                            nom_est_fem = _nom_entries_3a and all(
                                e.get("genre") == "f"
                                for e in _nom_entries_3a
                                if e.get("genre")
                            )
                            adj_infos = lexique.info(result[i])
                            adj_genred = [e for e in adj_infos if e.get("genre")]
                            adj_est_masc = adj_genred and all(
                                e.get("genre") == "m" for e in adj_genred
                            )
                            if nom_est_fem and adj_est_masc:
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
                                            regle="accord.genre_adj",
                                            explication="Accord en genre ADJ antepose + NOM fem",
                                        ))
                                        curr = result[i]
                                        curr_low = curr.lower()
                                        break

                    # Now pluralize (curr may have been feminized above)
                    # Guard: coordinated singular pattern (same as Rule 1b)
                    _coord_r1 = False
                    if i + 1 < n and result[i + 1].lower() in (
                        "et", "ou", "ni", "puis",
                    ):
                        _coord_r1 = True
                    # Guard: compound hyphenated words (invariable plurals)
                    _is_compound_r1 = "-" in curr
                    # Guard: OOV words should not be pluralized
                    # (proper names, foreign words, abbreviations)
                    _oov_r1 = (
                        lexique is not None
                        and not lexique.existe(curr_low)
                    )
                    # Guard: words primarily used as proper nouns
                    # (bourbon, bel, mossi — NOM freq < 5 + has NOM PROPRE)
                    # or words that are ONLY SIGLE (jo → JO, invariable)
                    _has_propre_r1 = False
                    if lexique is not None:
                        _r1_all_infos = lexique.info(curr_low)
                        if _r1_all_infos:
                            _r1_nom_freq_max = max(
                                (float(e.get("freq", 0) or 0)
                                 for e in _r1_all_infos
                                 if e.get("cgram") in ("NOM", "ADJ")),
                                default=0,
                            )
                            _r1_has_np = any(
                                "PROPRE" in (e.get("cgram") or "")
                                for e in _r1_all_infos
                            )
                            _r1_all_sigle = all(
                                ":sigle" in (e.get("cgram") or "")
                                or ":acronyme" in (e.get("cgram") or "")
                                or "PROPRE" in (e.get("cgram") or "")
                                for e in _r1_all_infos
                            )
                            if (_r1_has_np and _r1_nom_freq_max < 5.0) \
                                    or _r1_all_sigle:
                                _has_propre_r1 = True
                            # Capitalized mid-sentence + NOM PROPRE
                            # → proper noun (les Guise, les Bonaparte)
                            if (
                                not _has_propre_r1
                                and _r1_has_np
                                and i > 0
                                and curr[0].isupper()
                            ):
                                _has_propre_r1 = True
                    # Guard: NOM/VER-ambiguous word followed by NOM
                    # → likely a verb with direct object
                    # "les appelle pierres" = verb, not NOM
                    _ver_direct_obj = False
                    if (
                        lexique is not None
                        and i + 1 < n
                        and not _has_propre_r1
                    ):
                        _r1v_infos = lexique.info(curr_low)
                        if _r1v_infos and any(
                            e.get("cgram") in ("VER", "AUX")
                            and float(e.get("freq") or 0) > 30
                            for e in _r1v_infos
                        ):
                            _r1v_next_pos = (
                                pos_tags[i + 1]
                                if i + 1 < len(pos_tags) else ""
                            )
                            if _r1v_next_pos in ("NOM", "NOM PROPRE"):
                                _ver_direct_obj = True
                    if (
                        not curr_low.endswith(("s", "x", "z"))
                        and len(curr) > 1
                        and curr_low not in INVARIABLES
                        and not _coord_r1
                        and not _is_compound_r1
                        and not _oov_r1
                        and not _has_propre_r1
                        and not _ver_direct_obj
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
                                    regle="accord.nombre_nom",
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
            # Guard R2: low-freq loanwords (gimmick freq=0.11)
            _r2_low_freq = False
            if lexique is not None:
                _r2_freq = lexique.frequence(result[i].lower())
                if _r2_freq < 1.0:
                    _r2_low_freq = True
            if (
                prev2_low in PLUR_DET
                and not result[i].lower().endswith(("s", "x", "z"))
                and len(result[i]) > 1
                and result[i].lower() not in INVARIABLES
                and not _r2_low_freq
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
                            regle="accord.nombre_nom",
                            explication=f"Accord pluriel (det+adj+nom) apres '{prev2_low}'",
                        ))
                        break

        # Regle 2b : PLUR_DET + word(ADJ_lexique) + NOM → full accord cascade
        # Handles cases where P2G mistagged the ADJ as NOM, or ADJ ends in
        # s/x/z (skipped by Regle 1). Uses formes_de for irregular forms.
        # "les vieux maison" → "les vieilles maisons"
        # "les blanc chemise" → "les blanches chemises"
        if i > 1 and lexique is not None:
            _prev2_low_r2b = result[i - 2].lower()
            if _prev2_low_r2b in PLUR_DET:
                _prev_low_r2b = result[i - 1].lower()
                _curr_low_r2b = result[i].lower()
                # Check prev word has ADJ entries in lexique
                _prev_infos_r2b = lexique.info(_prev_low_r2b)
                _prev_adj_r2b = [
                    e for e in _prev_infos_r2b
                    if (e.get("cgram") or "").startswith("ADJ")
                ]
                # Check current word has NOM entries in lexique.
                # Use ORIGINAL form for gender (avoids pluralized form
                # having different entries, e.g. "vagues" only m but
                # "vague" has both m and f)
                _orig_low_r2b = (
                    origs[i].lower() if i < len(origs) else _curr_low_r2b
                )
                _curr_infos_r2b = lexique.info(_orig_low_r2b)
                if not _curr_infos_r2b:
                    _curr_infos_r2b = lexique.info(_curr_low_r2b)
                _curr_nom_r2b = [
                    e for e in _curr_infos_r2b
                    if e.get("cgram") == "NOM"
                ]
                if (
                    _prev_adj_r2b
                    and _curr_nom_r2b
                    and _prev_low_r2b not in _ACCORD_EXCLUS
                    and _prev_low_r2b not in INVARIABLES
                ):
                    # Get NOM gender (must be unambiguous)
                    _nom_genres_r2b = {
                        e.get("genre") for e in _curr_nom_r2b
                        if e.get("genre")
                    }
                    if len(_nom_genres_r2b) == 1:
                        _tgt_g_r2b = next(iter(_nom_genres_r2b))
                        # Get ADJ gender(s)
                        _adj_genres_r2b = {
                            e.get("genre") for e in _prev_adj_r2b
                            if e.get("genre")
                        }
                        # Check if gender flip needed
                        _needs_flip_r2b = (
                            _adj_genres_r2b
                            and _tgt_g_r2b not in _adj_genres_r2b
                        )
                        # Check if pluralization needed
                        _needs_plur_r2b = not _prev_low_r2b.endswith(
                            ("s", "x", "z"),
                        )
                        if _needs_flip_r2b or _needs_plur_r2b:
                            # Find target form via formes_de
                            _lemme_r2b = (
                                _prev_adj_r2b[0].get("lemme")
                                or _prev_adj_r2b[0].get("lemma")
                                or _prev_low_r2b
                            )
                            _RAW_G = {"f": ("f", "feminin", "Fem"),
                                      "m": ("m", "masculin", "Masc")}
                            _RAW_N = ("p", "pluriel", "Plur")
                            _RAW_S = ("s", "singulier", "Sing")
                            _tgt_form_r2b = None
                            if hasattr(lexique, "formes_de"):
                                _formes = lexique.formes_de(_lemme_r2b)
                                # Try target gender + plural directly
                                for _f in _formes:
                                    if (
                                        _f.get("genre") in _RAW_G.get(_tgt_g_r2b, ())
                                        and _f.get("nombre") in _RAW_N
                                        and (
                                            _f.get("cgram") or ""
                                        ).startswith("ADJ")
                                    ):
                                        _tgt_form_r2b = _f.get("ortho")
                                        break
                                # Fallback: target gender + singular, then pluralize
                                if not _tgt_form_r2b:
                                    for _f in _formes:
                                        if (
                                            _f.get("genre")
                                            in _RAW_G.get(_tgt_g_r2b, ())
                                            and _f.get("nombre") in _RAW_S
                                            and (
                                                _f.get("cgram") or ""
                                            ).startswith("ADJ")
                                        ):
                                            _fem_sing = _f.get("ortho", "")
                                            if _fem_sing:
                                                for _pc in generer_candidats_pluriel(
                                                    _fem_sing,
                                                ):
                                                    if lexique.existe(_pc):
                                                        _tgt_form_r2b = _pc
                                                        break
                                            break
                            # Fallback: generer_candidats + pluralize
                            if not _tgt_form_r2b and _needs_flip_r2b:
                                _gen_fn_r2b = (
                                    generer_candidats_feminin
                                    if _tgt_g_r2b == "f"
                                    else generer_candidats_masculin
                                )
                                for _c in _gen_fn_r2b(_prev_low_r2b):
                                    _c_infos = lexique.info(_c)
                                    if _c_infos and any(
                                        e.get("genre") == _tgt_g_r2b
                                        and (
                                            e.get("cgram") or ""
                                        ).startswith("ADJ")
                                        for e in _c_infos
                                    ):
                                        # Pluralize the feminized form
                                        for _pc in generer_candidats_pluriel(
                                            _c,
                                        ):
                                            if lexique.existe(_pc):
                                                _tgt_form_r2b = _pc
                                                break
                                        if not _tgt_form_r2b:
                                            _tgt_form_r2b = _c
                                        break
                            # Only pluralization needed (right gender)
                            if (
                                not _tgt_form_r2b
                                and not _needs_flip_r2b
                                and _needs_plur_r2b
                            ):
                                for _pc in generer_candidats_pluriel(
                                    _prev_low_r2b,
                                ):
                                    if lexique.existe(_pc):
                                        _tgt_form_r2b = _pc
                                        break
                            # Apply ADJ correction
                            if (
                                _tgt_form_r2b
                                and _tgt_form_r2b.lower()
                                != result[i - 1].lower()
                            ):
                                _anc_r2b = result[i - 1]
                                result[i - 1] = transferer_casse(
                                    _anc_r2b, _tgt_form_r2b,
                                )
                                corrections.append(Correction(
                                    index=i - 1,
                                    original=_anc_r2b,
                                    corrige=result[i - 1],
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    regle="accord.adj_antepose",
                                    explication=(
                                        f"ADJ antepose → accord "
                                        f"{'fem' if _tgt_g_r2b == 'f' else 'masc'}"
                                        f" plur (NOM '{result[i]}')"
                                    ),
                                ))
                    # Also pluralize the NOM at i if needed
                    # Guard: don't pluralize VER/AUX as NOM (est→ests)
                    _pos_r2b_i = pos_tags[i] if i < len(pos_tags) else ""
                    if (
                        _pos_r2b_i not in ("VER", "AUX")
                        and not _curr_low_r2b.endswith(("s", "x", "z"))
                        and len(_curr_low_r2b) > 1
                        and _curr_low_r2b not in INVARIABLES
                    ):
                        for _pc_nom in generer_candidats_pluriel(
                            result[i],
                        ):
                            if lexique.existe(_pc_nom):
                                _anc_nom_r2b = result[i]
                                result[i] = _pc_nom
                                corrections.append(Correction(
                                    index=i,
                                    original=_anc_nom_r2b,
                                    corrige=_pc_nom,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    regle="accord.nombre_nom",
                                    explication=(
                                        f"Accord pluriel (adj+nom) "
                                        f"apres '{_prev2_low_r2b}'"
                                    ),
                                ))
                                break

        # Regle 7 : NOM feminin + ADJ masculin -> forme feminine
        # (avant Regle 1b pour que "vert" devienne "verte" avant "vertes")
        if i > 0 and pos == "ADJ" and lexique is not None:
            prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            # Accept NOM tag or lexique-confirmed NOM (OOV words fixed by ortho)
            _prev_is_nom = prev_pos == "NOM"
            if not _prev_is_nom and not prev_pos:
                _prev_infos = lexique.info(result[i - 1])
                if _prev_infos:
                    _cgrams = {e.get("cgram") for e in _prev_infos if e.get("cgram")}
                    _prev_is_nom = _cgrams == {"NOM"}
            if _prev_is_nom:
                prev_infos = lexique.info(result[i - 1])
                # Filtrer uniquement les entrees NOM pour verifier le genre
                # (sinon des entrees VER/ADJ fem parasitent : "musee" VER:f)
                _nom_entries = [e for e in prev_infos if e.get("cgram") == "NOM"]
                nom_est_fem = _nom_entries and all(
                    e.get("genre") == "f" for e in _nom_entries
                )
                # Guard: si le mot a aussi des entrees ADJ, il peut etre
                # un ADJ tague NOM (ex: "embarras politique majeur")
                # → ne pas forcer l'accord feminin
                if nom_est_fem and any(
                    e.get("cgram") == "ADJ" for e in prev_infos
                ):
                    nom_est_fem = False
                # Guard: si un DET masculin precede le NOM, le NOM est
                # employe au masculin malgre ses entrees feminines
                # (ex: "le journaliste sportif")
                if nom_est_fem and i >= 2:
                    _MASC_DET = frozenset({
                        "le", "l'", "l\u2019", "un", "ce", "cet",
                        "mon", "ton", "son", "notre", "votre",
                        "du",
                    })
                    for _k in range(i - 2, max(-1, i - 7), -1):
                        _pk = pos_tags[_k] if _k < len(pos_tags) else ""
                        if _pk.startswith(("ART", "DET")) or _pk == "ADJ:dem":
                            if mots[_k].lower() in _MASC_DET:
                                nom_est_fem = False
                            break
                        if _pk not in ("NOM", "NOM PROPRE", "ADJ", "CON"):
                            break
                # Guard: ADJ pre-nominal — si le mot suivant est NOM masculin,
                # l'ADJ modifie probablement le NOM suivant, pas le precedent
                # "noblesse premier pas" → "premier" modifie "pas" (m), pas "noblesse" (f)
                if nom_est_fem and i + 1 < n:
                    _next_pos_r7 = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                    _next_infos_r7 = lexique.info(result[i + 1])
                    _next_nom_r7 = [e for e in _next_infos_r7 if e.get("cgram") == "NOM"]
                    # Guard by POS tag or by lexique NOM entries
                    if _next_nom_r7 and all(
                        e.get("genre") == "m" for e in _next_nom_r7
                    ):
                        # Accept if tagged NOM, or if NOM is the highest-freq
                        # cgram for the next word (ex: "pas" tagged ADV but
                        # really NOM masc in "premier pas")
                        if _next_pos_r7 == "NOM":
                            nom_est_fem = False
                        elif _next_infos_r7:
                            _best_cgram_r7 = max(
                                _next_infos_r7,
                                key=lambda e: float(e.get("freq") or 0),
                            ).get("cgram", "")
                            if _best_cgram_r7 == "NOM":
                                nom_est_fem = False
                # Guard: epicene nouns (secrétaire, guitariste, photographe)
                # are lexique-marked as feminine only, but often used for males
                # in biographical contexts. Require a FEM_DET before the NOM.
                _EPICENE_NOMS = frozenset({
                    "secrétaire", "guitariste", "photographe",
                    "journaliste", "artiste", "pianiste", "violoniste",
                    "touriste", "dentiste", "spécialiste", "fleuriste",
                    "diplomate", "athlète", "pilote",
                    "ministre", "juge", "arbitre", "capitaine",
                    "propriétaire", "locataire", "partenaire",
                    "comptable", "responsable",
                    # Job titles often used with masc DET for both genders
                    "essayiste", "bassiste", "saxophoniste",
                    "biologiste", "économiste", "linguiste",
                    "écrivain", "auteur", "professeur", "médecin",
                    "ingénieur", "maire", "architecte",
                })
                if (
                    nom_est_fem
                    and result[i - 1].lower() in _EPICENE_NOMS
                ):
                    # Only feminize if a feminine DET exists nearby
                    _has_fem_det_r7 = False
                    _FEM_DET = frozenset({
                        "la", "une", "cette", "ma", "ta", "sa",
                    })
                    for _k7e in range(i - 2, max(-1, i - 6), -1):
                        _k7e_pos = pos_tags[_k7e] if _k7e < len(pos_tags) else ""
                        if _k7e_pos.startswith(("ART", "DET")):
                            if mots[_k7e].lower() in _FEM_DET:
                                _has_fem_det_r7 = True
                            break
                        if _k7e_pos in ("VER", "AUX", "CON"):
                            break
                    if not _has_fem_det_r7:
                        nom_est_fem = False
                # Guard: skip if ADJ word has NOM PROPRE entries
                # (proper nouns: François, Besson, etc.)
                if nom_est_fem:
                    _adj_has_np = any(
                        (e.get("cgram") or "") == "NOM PROPRE"
                        for e in (lexique.info(result[i]) or [])
                    )
                    if _adj_has_np:
                        nom_est_fem = False
                # Guard: NOM inside PP → ADJ modifies subject, not PP NOM
                # "metteur en scène français" → scène in PP, français=metteur
                if nom_est_fem and i >= 2:
                    _pre_nom_pos = (
                        pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                    )
                    if _pre_nom_pos == "PRE":
                        nom_est_fem = False
                if nom_est_fem:
                    adj_infos = lexique.info(result[i])
                    adj_genred = [e for e in adj_infos if e.get("genre")]
                    adj_est_masc = adj_genred and all(
                        e.get("genre") == "m" for e in adj_genred
                    )
                    if adj_est_masc:
                        found_fem = False
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
                                    regle="accord.genre_adj",
                                    explication="Accord en genre NOM fem + ADJ",
                                ))
                                # Update curr_low for subsequent rules
                                curr = result[i]
                                curr_low = curr.lower()
                                found_fem = True
                                break
                        # Fallback morphologique : mot + "e" pour OOV
                        if not found_fem and not result[i].lower().endswith("e"):
                            fallback = result[i] + "e"
                            if lexique.existe(fallback):
                                ancien = result[i]
                                result[i] = fallback
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=fallback,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    regle="accord.genre_adj",
                                    explication="Accord en genre NOM fem + ADJ (fallback)",
                                ))
                                curr = result[i]
                                curr_low = curr.lower()

        # Regle 7b : NOM masculin + ADJ feminin -> forme masculine
        # "siège épiscopale" → "siège épiscopal", "parc nationale" → "national"
        if i > 0 and pos == "ADJ" and lexique is not None:
            prev_pos_7b = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            _prev_is_nom_7b = prev_pos_7b == "NOM"
            if _prev_is_nom_7b:
                prev_infos_7b = lexique.info(result[i - 1])
                _nom_entries_7b = [e for e in prev_infos_7b if e.get("cgram") == "NOM"]
                nom_est_masc_7b = _nom_entries_7b and all(
                    e.get("genre") == "m" for e in _nom_entries_7b
                )
                # Guard: si le NOM est dans un PP ("de fer principale" → "fer"
                # modifie par PP, l'ADJ ne s'accorde pas avec lui)
                _PP_MARKERS_7B = frozenset({
                    "de", "du", "des", "d'", "en", "au", "aux",
                })
                if nom_est_masc_7b and i >= 2:
                    _pre_nom_7b = result[i - 2].lower()
                    _pre_nom_pos_7b = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                    if _pre_nom_pos_7b == "PRE" or _pre_nom_7b in _PP_MARKERS_7B:
                        nom_est_masc_7b = False
                # Guard: si un DET feminin precede le NOM, ne pas masculiniser
                if nom_est_masc_7b and i >= 2:
                    _FEM_DET_7B = frozenset({
                        "la", "une", "cette", "sa", "ma", "ta",
                    })
                    for _k7b in range(i - 2, max(-1, i - 5), -1):
                        _pk7b = pos_tags[_k7b] if _k7b < len(pos_tags) else ""
                        if _pk7b.startswith(("ART", "DET", "ADJ:")) or _pk7b == "ADJ:dem":
                            if mots[_k7b].lower() in _FEM_DET_7B:
                                nom_est_masc_7b = False
                            break
                        if not _pk7b.startswith(("NOM", "ADJ")):
                            break
                if nom_est_masc_7b:
                    adj_infos_7b = lexique.info(result[i])
                    adj_genred_7b = [e for e in adj_infos_7b if e.get("genre")]
                    adj_est_fem_7b = adj_genred_7b and all(
                        e.get("genre") == "f" for e in adj_genred_7b
                    )
                    if adj_est_fem_7b:
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
                                    regle="accord.genre_adj",
                                    explication="Accord en genre NOM masc + ADJ",
                                ))
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
                    # Guard: invariable ADJ expressions
                    # "meilleur marché" = cheaper (invariable)
                    _invar_expr = False
                    if i + 1 < n:
                        _next_1b = result[i + 1].lower()
                        if curr_low == "meilleur" and _next_1b == "marché":
                            _invar_expr = True
                    # Guard: coordinated singular adjectives
                    # "les sections française et indigène" → each ADJ
                    # qualifies a separate element, don't pluralize
                    _coord_r1b = False
                    if i + 1 < n and result[i + 1].lower() in (
                        "et", "ou", "ni", "puis",
                    ):
                        _coord_r1b = True
                    # Guard: present participle -ant + preposition
                    # "domaines dépendant des temples" → gerund, invariable
                    _part_pres_r1b = False
                    if (
                        not _coord_r1b
                        and curr_low.endswith("ant")
                        and lexique is not None
                        and i + 1 < n
                    ):
                        _r1b_infos = lexique.info(curr_low)
                        if _r1b_infos and any(
                            e.get("cgram") == "VER"
                            for e in _r1b_infos
                        ):
                            _next_r1b_low = result[i + 1].lower()
                            if _next_r1b_low in (
                                "de", "du", "des", "d'",
                                "en", "par", "à", "au", "aux",
                            ):
                                _part_pres_r1b = True
                    if (
                        det_found
                        and not curr_low.endswith(("s", "x", "z"))
                        and len(curr) > 1
                        and curr_low not in INVARIABLES
                        and not _invar_expr
                        and not _coord_r1b
                        and not _part_pres_r1b
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
                                    regle="accord.nombre_adj",
                                    explication="Accord pluriel adj post-nominal",
                                ))
                                break

        # Regle 1f : ADJ pre-nominal singulier + NOM pluriel → pluraliser l'ADJ
        # "de jolie femmes" → "de jolies femmes", "de petit enfants" → "de petits enfants"
        # Guard: le mot avant l'ADJ doit etre une preposition (de, d', des)
        _PREP_PRE_ADJ = frozenset({"de", "d'", "des", "d\u2019"})
        if (
            i > 0
            and pos == "ADJ"
            and not curr_low.endswith(("s", "x", "z"))
            and len(curr) > 1
            and curr_low not in INVARIABLES
            and i + 1 < n
        ):
            _next_pos_1f = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
            if _next_pos_1f == "NOM":
                _next_low_1f = result[i + 1].lower()
                # Le NOM suivant doit etre au pluriel
                if _next_low_1f.endswith(("s", "x", "z")) and len(result[i + 1]) > 2:
                    _prev_low_1f = result[i - 1].lower()
                    if _prev_low_1f in _PREP_PRE_ADJ:
                        for candidate in generer_candidats_pluriel(curr):
                            if lexique is None or lexique.existe(candidate):
                                ancien = result[i]
                                result[i] = candidate
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=candidate,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    regle="accord.nombre_adj",
                                    explication="Accord pluriel ADJ pre-nominal + NOM pluriel",
                                ))
                                curr = result[i]
                                curr_low = curr.lower()
                                break

        # Regle 1c : NOM singulier + ADJ pluriel → singulariser l'ADJ
        # "année suivantes" → "année suivante", "poste actuels" → "poste actuel"
        # Guard: un DET singulier doit preceder le NOM (confirme le contexte)
        if (
            i > 1
            and pos == "ADJ"
            and curr_low.endswith(("s", "x"))
            and not curr_low.endswith(("ss", "eux", "oux"))
            and len(curr) > 2
            and curr_low not in INVARIABLES
            and lexique is not None
        ):
            prev_pos_1c = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            if prev_pos_1c in ("NOM", "NOM PROPRE"):
                prev_low_1c = result[i - 1].lower()
                # Le NOM doit etre au singulier (pas de -s/-x/-z)
                if not prev_low_1c.endswith(("s", "x", "z")):
                    # Chercher un DET singulier avant le NOM
                    # Guard: exclure les DET prepositionnels (du, au)
                    # qui introduisent un PP — l'ADJ peut accorder avec un NOM plus loin
                    _sing_det_1c = False
                    _PREP_DET = frozenset({"du", "au"})
                    for _k in range(i - 2, max(-1, i - 5), -1):
                        _kw = result[_k].lower()
                        if _kw in SING_DET and _kw not in PLUR_DET:
                            if _kw not in _PREP_DET:
                                # Guard PP : si une PRE precede ce DET,
                                # le NOM est dans un complement prepositionnel
                                # (ex: "contre la poitrine, incapables")
                                if _k > 0:
                                    _pre_pos = pos_tags[_k - 1] if _k - 1 < len(pos_tags) else ""
                                    _pre_mot = result[_k - 1].lower()
                                    if _pre_pos == "PRE" or _pre_mot in (
                                        "de", "dans", "sur", "sous", "pour",
                                        "par", "avec", "sans", "contre",
                                        "entre", "vers", "chez", "devant",
                                        "derrière", "après", "avant",
                                    ):
                                        break
                                _sing_det_1c = True
                            break
                        _kp = pos_tags[_k] if _k < len(pos_tags) else ""
                        if _kp not in ("ADJ", "ADV"):
                            break
                    # Guard coordination: "NOM1 et/ou NOM2 ADJ_plur"
                    # L'ADJ peut accorder au pluriel avec deux NOM coordonnes
                    # Ex: "la production et la direction théâtrales"
                    _coord_1c = False
                    if _sing_det_1c:
                        for _kc1c in range(i - 2, max(-1, i - 10), -1):
                            _kc1c_low = result[_kc1c].lower()
                            if _kc1c_low in ("et", "ou"):
                                _coord_1c = True
                                break
                            _kc1c_pos = pos_tags[_kc1c] if _kc1c < len(pos_tags) else ""
                            if _kc1c_pos in (
                                "VER", "AUX", "CON",
                            ) or _kc1c_low in (".", ",", ";", ":"):
                                break
                    if _sing_det_1c and not _coord_1c:
                        for candidate in generer_candidats_singulier_nom(curr):
                            c_infos = lexique.info(candidate)
                            if c_infos and any(
                                e.get("cgram") == "ADJ" for e in c_infos
                            ):
                                ancien = result[i]
                                result[i] = candidate
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=candidate,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    regle="accord.nombre_adj",
                                    explication="Depluralization ADJ apres NOM singulier",
                                ))
                                curr = result[i]
                                curr_low = curr.lower()
                                break

        # Regle 1d : ADJ singulier + NOM pluriel → singulariser le NOM
        # "ancien joueurs" → "ancien joueur", "première femmes" → "première femme"
        # Guard: un DET singulier doit preceder l'ADJ
        # Guard: acronymes et mots courts exclus (<=3 chars)
        # Guard: candidat doit etre NOM dans le lexique
        # Guard: ADJ introductifs de liste exclus (suivante, précédente...)
        _LIST_INTRO_ADJ = frozenset({
            "suivant", "suivante", "suivants", "suivantes",
            "précédent", "précédente", "précédents", "précédentes",
        })
        if (
            i > 1
            and pos == "NOM"
            and curr_low.endswith(("s", "x"))
            and not curr_low.endswith(("ss", "eux", "oux"))
            and len(curr) > 3
            and curr_low not in INVARIABLES
            and lexique is not None
        ):
            prev_pos_1d = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            if prev_pos_1d == "ADJ":
                prev_low_1d = result[i - 1].lower()
                # L'ADJ doit etre au singulier (pas de -s/-x/-z)
                # et ne pas etre un ADJ introductif de liste
                if (
                    not prev_low_1d.endswith(("s", "x", "z"))
                    and prev_low_1d not in _LIST_INTRO_ADJ
                ):
                    # Chercher un DET singulier avant l'ADJ
                    _sing_det_1d = False
                    for _k in range(i - 2, max(-1, i - 5), -1):
                        _kw = result[_k].lower()
                        if _kw in SING_DET and _kw not in PLUR_DET:
                            _sing_det_1d = True
                            break
                        _kp = pos_tags[_k] if _k < len(pos_tags) else ""
                        if _kp not in ("ADJ", "ADV"):
                            break
                    if _sing_det_1d:
                        for candidate in generer_candidats_singulier_nom(curr):
                            c_infos = lexique.info(candidate)
                            if c_infos and any(
                                e.get("cgram") == "NOM" for e in c_infos
                            ):
                                ancien = result[i]
                                result[i] = candidate
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=candidate,
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    regle="accord.nombre_nom",
                                    explication="Depluralization NOM apres ADJ singulier",
                                ))
                                curr = result[i]
                                curr_low = curr.lower()
                                break

        # Regle 1e : SING_DET + [ADJ]? + NOM/ADJ pluriel → singulariser
        # "au bouts" → "au bout", "ce postes" → "ce poste"
        # "une autre sculptures" → "sculpture"
        if (
            i > 0
            and pos in ("NOM", "ADJ")
            and curr_low.endswith(("s", "x"))
            and not curr_low.endswith(("ss", "eux", "oux"))
            and len(curr) > 2
            and curr_low not in INVARIABLES
            and "-" not in curr  # compound words (lance-flammes)
            and lexique is not None
        ):
            # Check immediate prev or 1 intermediate ADJ
            _sing_det_1e = False
            _det_word_1e = ""
            _prev_1e = result[i - 1].lower()
            if _prev_1e in SING_DET and _prev_1e not in PLUR_DET:
                _sing_det_1e = True
                _det_word_1e = _prev_1e
            elif i > 1:
                # Allow 1 intermediate word: ADJ, ADV, or DET-like
                # that is not in PLUR_DET (e.g. "autre" tagged ART:ind)
                # Guard: skip predicative ADJ ("la suivante", "le premier")
                _PREDICATIVE_ADJ = frozenset({
                    "suivant", "suivante", "suivants", "suivantes",
                    "précédent", "précédente", "précédents", "précédentes",
                    "premier", "première", "premiers", "premières",
                    "dernier", "dernière", "derniers", "dernières",
                })
                _prev_1e_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                if (
                    _prev_1e not in _PREDICATIVE_ADJ
                    and (
                        _prev_1e_pos in ("ADJ", "ADV")
                        or (
                            _prev_1e_pos.startswith(("ART", "DET"))
                            and _prev_1e not in PLUR_DET
                            and _prev_1e not in SING_DET
                        )
                    )
                ):
                    _prev2_1e = result[i - 2].lower()
                    if _prev2_1e in SING_DET and _prev2_1e not in PLUR_DET:
                        _sing_det_1e = True
                        _det_word_1e = _prev2_1e
            if _sing_det_1e:
                # Guard: si le mot-avec-s a des entrees NOM singulier
                # dans le lexique → mot invariable (fonds, bras, temps)
                _is_sing_with_s = False
                _curr_infos_1e = lexique.info(curr_low)
                if _curr_infos_1e and any(
                    e.get("cgram") in ("NOM", "ADJ")
                    and e.get("nombre") in ("singulier", "s")
                    for e in _curr_infos_1e
                ):
                    _is_sing_with_s = True
                # Guard OOV: not in lexique → likely proper noun
                if not _curr_infos_1e:
                    _is_sing_with_s = True
                # Guard NOM PROPRE: pure proper nouns
                if _curr_infos_1e and all(
                    "PROPRE" in (e.get("cgram") or "")
                    for e in _curr_infos_1e
                ):
                    _is_sing_with_s = True
                # Guard NOM PROPRE + low NOM freq (mans=Le Mans)
                if (
                    _curr_infos_1e
                    and any(
                        "PROPRE" in (e.get("cgram") or "")
                        for e in _curr_infos_1e
                    )
                    and max(
                        (float(e.get("freq", 0) or 0)
                         for e in _curr_infos_1e
                         if e.get("cgram") in ("NOM", "ADJ")),
                        default=0,
                    ) < 3.0
                ):
                    _is_sing_with_s = True
                # Guard: coordination après → skip
                _coord_after_1e = False
                if i + 1 < n:
                    _n1e_low = result[i + 1].lower()
                    if _n1e_low in ("et", "ou"):
                        _coord_after_1e = True
                if not _coord_after_1e and not _is_sing_with_s:
                    for candidate in generer_candidats_singulier_nom(curr):
                        c_infos = lexique.info(candidate)
                        if c_infos and any(
                            e.get("cgram") in ("NOM", "ADJ") for e in c_infos
                        ):
                            ancien = result[i]
                            result[i] = candidate
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige=candidate,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="accord.nombre_nom",
                                explication=f"Depluralization apres DET singulier '{_det_word_1e}'",
                            ))
                            curr = result[i]
                            curr_low = curr.lower()
                            break

        # Regle 4 : Sujet pluriel (eventuellement distant) + VER -> 3pl
        # Ne pas appliquer si etre precede (laisser la regle PP_etre gerer)
        # Guard: VER directement apres PRE = probable nom propre (à Vienne)
        _prev_is_pre_r4 = (
            i > 0
            and (pos_tags[i - 1] if i - 1 < len(pos_tags) else "") == "PRE"
        )
        # Guard R4: mot VER qui a aussi des entrees NOM apres un DET
        # = probablement un NOM pluriel, pas un VER (ex: "ses œuvres")
        _skip_nom_r4 = False
        if (
            i > 1
            and pos in ("VER", "AUX")
            and not _prev_is_pre_r4
            and lexique is not None
            and curr_low.endswith(("es", "s"))
        ):
            _prev_r4_low = result[i - 1].lower()
            _prev_r4_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            if (
                _prev_r4_pos.startswith(("ART", "DET", "ADJ:pos"))
                or _prev_r4_low in SING_DET | PLUR_DET
            ):
                _r4_infos = lexique.info(curr)
                if any(e.get("cgram") == "NOM" for e in _r4_infos):
                    _skip_nom_r4 = True
        if i > 1 and pos in ("VER", "AUX") and not _prev_is_pre_r4 and not _skip_nom_r4:
            _etre_before = False
            for _j in range(i - 1, max(-1, i - 4), -1):
                _w = result[_j].lower()
                if _w in ETRE_FORMES:
                    _etre_before = True
                    break
                if _w not in _TRANSPARENTS:
                    break
            # Guard est→sont : si le contexte suggere coordination
            # (NOM/ADJ + est + NOM/ART/DET/PRE), laisser homophones decider
            _skip_est_coord = False
            if curr_low == "est" and i + 1 < n:
                _npc = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _nlc = result[i + 1].lower()
                if _npc in (
                    "NOM", "NOM PROPRE", "ART", "ART:def", "ART:ind",
                    "DET", "?", "",
                ) or _nlc.endswith(("'", "\u2019")):
                    _skip_est_coord = True
                # Guard: single-letter fragment (orphan elision "l", "d")
                elif len(_nlc) == 1 and _nlc.isalpha():
                    _skip_est_coord = True
                # Guard: ADJ already plural → coordination ("fédéraux et municipaux")
                elif (
                    _npc in ("ADJ", "ADJ:pos")
                    and _nlc.endswith(("s", "x", "z"))
                ):
                    _skip_est_coord = True
                # Guard: VER/AUX not a PP form → conjugated verb = coordination
                elif _npc in ("VER", "AUX") and not _nlc.endswith((
                    "\u00e9", "\u00e9s", "\u00e9e", "\u00e9es",
                    "i", "is", "ie", "ies",
                    "u", "us", "ue", "ues",
                    "it", "ite", "ites",
                    "ert", "erte", "ertes", "erts",
                )):
                    _skip_est_coord = True
                # Guard: est + singular ADJ = copula
                # "les indicateurs est complexe" → copula, not *sont
                elif (
                    _npc in ("ADJ", "ADJ:pos")
                    and not _nlc.endswith(("s", "x", "z"))
                ):
                    _skip_est_coord = True
                # Guard: est + singular PP = passive voice
                # "est située", "est construit" → singular, not coordination
                elif (
                    _npc in ("VER", "AUX")
                    and not _nlc.endswith(("s", "x", "z"))
                    and _nlc.endswith((
                        "\u00e9", "\u00e9e",  # é, ée
                        "i", "ie", "u", "ue",
                        "it", "ite", "ert", "erte",
                    ))
                ):
                    _skip_est_coord = True

            # Guard: causatif "fait/faire + infinitif" — ne pas pluraliser
            _skip_causatif_r4 = False
            if curr_low in ("fait", "fais") and i + 1 < n:
                _next_caus_r4 = result[i + 1].lower()
                if _next_caus_r4.endswith(("er", "ir", "re", "oir")):
                    if lexique is None or lexique.existe(_next_caus_r4):
                        _skip_causatif_r4 = True

            # Guard: NOM/ADJ homograph — si la forme singuliere du mot
            # est principalement NOM/ADJ, c'est probablement un nom/adj
            # au pluriel et non un verbe a conjuguer.
            # Ex: "arts graphiques" → "graphique" = NOM/ADJ, pas VER
            _skip_nom_adj_r4 = False
            if (
                not _etre_before
                and not _skip_est_coord
                and not _skip_causatif_r4
                and lexique is not None
                and hasattr(lexique, "info")
            ):
                if curr_low.endswith("s") and len(curr_low) > 3:
                    _sing_r4 = curr_low[:-1]
                    _sing_infos_r4 = lexique.info(_sing_r4)
                    if _sing_infos_r4:
                        _best_sing_r4 = max(
                            _sing_infos_r4,
                            key=lambda e: float(e.get("freq") or 0),
                        )
                        if (_best_sing_r4.get("cgram") or "") in ("NOM", "ADJ"):
                            _skip_nom_adj_r4 = True

            if not _etre_before and not _skip_est_coord and not _skip_causatif_r4 and not _skip_nom_adj_r4 and (
                not curr_low.endswith(("ent", "nt"))
                and _trouver_sujet_pluriel(result, pos_tags, i)
            ):
                candidats = generer_candidats_3pl(curr)
                for candidate in candidats:
                    if lexique is None:
                        _cand_ok = True
                    else:
                        _c_infos = lexique.info(candidate)
                        _cand_ok = any(
                            e.get("cgram") in ("VER", "AUX")
                            and float(e.get("freq") or 0) > 0.05
                            for e in _c_infos
                        )
                    if _cand_ok:
                        ancien = result[i]
                        result[i] = candidate
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=candidate,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="accord.sujet_verbe",
                            explication="Accord sujet pluriel -> verbe 3pl",
                        ))
                        break

        # Regle 9 : Copule (etre/sembler/devenir...) + ADJ -> accord genre
        # avec le sujet. Applique AVANT Regle 5 (nombre) pour que
        # "elles sont petit" → petite (genre) → petites (nombre).
        # Also handle NOM when the word has higher-freq ADJ entries
        # (P2G sometimes tags "chaud", "froid" etc. as NOM instead of ADJ)
        _is_adj_r9 = pos == "ADJ"
        if not _is_adj_r9 and pos == "NOM" and lexique is not None:
            _r9_infos = lexique.info(result[i])
            _r9_adj = [e for e in _r9_infos if e.get("cgram") == "ADJ"]
            _r9_nom = [e for e in _r9_infos if e.get("cgram") == "NOM"]
            if _r9_adj and _r9_nom:
                _r9_adj_freq = max(float(e.get("freq") or 0) for e in _r9_adj)
                _r9_nom_freq = max(float(e.get("freq") or 0) for e in _r9_nom)
                if _r9_adj_freq > _r9_nom_freq:
                    _is_adj_r9 = True
        if i > 0 and _is_adj_r9 and lexique is not None:
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
                        _found_fem = False
                        # Essayer d'abord par formes_de (gere les irreguliers)
                        if hasattr(lexique, "formes_de"):
                            _lem_attr = adj_genred[0].get("lemme") or adj_genred[0].get("lemma") or result[i].lower()
                            for f in lexique.formes_de(_lem_attr):
                                if (
                                    f.get("genre") in ("f", "feminin", "Fem")
                                    and f.get("nombre") in ("s", "singulier", "Sing", _s_nombre)
                                    and f.get("cgram") == "ADJ"
                                ):
                                    cand = f.get("ortho", "")
                                    if cand and cand.lower() != result[i].lower():
                                        ancien = result[i]
                                        result[i] = transferer_casse(ancien, cand)
                                        corrections.append(Correction(
                                            index=i,
                                            original=ancien,
                                            corrige=result[i],
                                            type_correction=TypeCorrection.GRAMMAIRE,
                                            regle="accord.genre_attribut",
                                            explication="Accord attribut en genre (sujet fem)",
                                        ))
                                        curr = result[i]
                                        curr_low = curr.lower()
                                        _found_fem = True
                                        break
                        # Fallback par suffixe
                        if not _found_fem:
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
                                        regle="accord.genre_attribut",
                                        explication="Accord attribut en genre (sujet fem)",
                                    ))
                                    curr = result[i]
                                    curr_low = curr.lower()
                                    break
                    elif s_genre == "Masc" and adj_genred and all(
                        e.get("genre") == "f" for e in adj_genred
                    ):
                        _found_masc = False
                        if hasattr(lexique, "formes_de"):
                            _lem_attr_m = adj_genred[0].get("lemme") or adj_genred[0].get("lemma") or result[i].lower()
                            for f in lexique.formes_de(_lem_attr_m):
                                if (
                                    f.get("genre") in ("m", "masculin", "Masc")
                                    and f.get("nombre") in ("s", "singulier", "Sing", _s_nombre)
                                    and f.get("cgram") == "ADJ"
                                ):
                                    cand = f.get("ortho", "")
                                    if cand and cand.lower() != result[i].lower():
                                        ancien = result[i]
                                        result[i] = transferer_casse(ancien, cand)
                                        corrections.append(Correction(
                                            index=i,
                                            original=ancien,
                                            corrige=result[i],
                                            type_correction=TypeCorrection.GRAMMAIRE,
                                            regle="accord.genre_attribut",
                                            explication="Accord attribut en genre (sujet masc)",
                                        ))
                                        curr = result[i]
                                        curr_low = curr.lower()
                                        _found_masc = True
                                        break
                        if not _found_masc:
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
                                        regle="accord.genre_attribut",
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
                # Guard: if the copule was itself corrected in this pass
                # (e.g., est→sont by Rule 4), the plural form may be wrong
                # — don't cascade the error to the attribut
                _copule_was_corrected = any(
                    c.index == i - 1
                    and c.corrige.lower() in COPULES_PLURIEL
                    for c in corrections
                )
                if not _copule_was_corrected:
                    for candidate in generer_candidats_pluriel(curr):
                        if lexique is None or lexique.existe(candidate):
                            ancien = result[i]
                            result[i] = candidate
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige=candidate,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="accord.nombre_attribut",
                                explication="Accord attribut apres copule plurielle",
                            ))
                            break

        # Regle 5b : Copule singuliere + (ADV*) + ADJ/PP pluriel -> singulariser
        # "est consommés" → "consommé", "est traditionnellement consommés"
        # Guard: si le sujet est pluriel (DET pluriel avant la copule),
        # c'est la copule qui est fausse, pas l'attribut
        _is_pp_like_5b = pos in ("ADJ", "VER")
        # NOM tague mais qui a des entrees VER participe passe
        # (ex: "élus" tague NOM freq=18.9, mais a aussi VER PP freq=4.3)
        if not _is_pp_like_5b and pos == "NOM" and lexique is not None:
            _nom_infos_5b = lexique.info(curr_low)
            if any(
                e.get("cgram") in ("VER", "AUX")
                and e.get("mode") in ("participe", "par")
                for e in _nom_infos_5b
            ):
                _is_pp_like_5b = True
        if i > 0 and _is_pp_like_5b and lexique is not None:
            # Chercher la copule en sautant les ADV
            _copule_5b = None
            for _j5b in range(i - 1, max(-1, i - 4), -1):
                _j5b_low = result[_j5b].lower()
                _j5b_pos = pos_tags[_j5b] if _j5b < len(pos_tags) else ""
                if _j5b_low in COPULES_SINGULIER:
                    _copule_5b = _j5b
                    break
                if _j5b_pos != "ADV":
                    break
            if (
                _copule_5b is not None
                and curr_low.endswith(("s", "x"))
                and not curr_low.endswith(("ss", "eux", "oux"))
                and len(curr) > 2
                and curr_low not in INVARIABLES
            ):
                # Guard: verifier qu'aucun DET pluriel n'existe avant la copule
                _plur_subj_5b = False
                for _k in range(_copule_5b - 1, max(-1, _copule_5b - 7), -1):
                    _kw = result[_k].lower()
                    if _kw in PLUR_DET or _kw in ("ils", "elles"):
                        _plur_subj_5b = True
                        break
                    # Pronom sujet singulier → sujet trouve, pas besoin
                    # de chercher plus loin (evite de croiser une clause)
                    if _kw in ("il", "elle", "on", "je", "tu",
                               "ce", "c'", "ça"):
                        break
                    _kp = pos_tags[_k] if _k < len(pos_tags) else ""
                    if _kp in ("VER", "AUX", "CON") or _kw in (".", ",", ";"):
                        break
                # Guard: "est" preceded by plural ADJ/NOM = likely "et"
                # "originales est sophistiquées" → "et" not copula
                _est_likely_et_5b = False
                if (
                    _copule_5b is not None
                    and result[_copule_5b].lower() == "est"
                    and _copule_5b > 0
                ):
                    _pre_est_5b = result[_copule_5b - 1].lower()
                    _pre_est_pos_5b = (
                        pos_tags[_copule_5b - 1]
                        if _copule_5b - 1 < len(pos_tags) else ""
                    )
                    if (
                        _pre_est_pos_5b in ("ADJ", "NOM", "VER")
                        and _pre_est_5b.endswith(("s", "x", "es"))
                        and len(_pre_est_5b) > 2
                    ):
                        _est_likely_et_5b = True
                    # Symmetric: current word also has plural morphology
                    # "oeuvres est considérés" → both sides plural = "et"
                    # Require pre-est to be ADJ/NOM/VER (not OOV/PROPRE)
                    if (
                        not _est_likely_et_5b
                        and curr_low.endswith(("és", "ées"))
                        and _pre_est_pos_5b in ("ADJ", "NOM", "VER")
                        and _pre_est_5b.endswith(("s", "x", "es"))
                        and len(_pre_est_5b) > 2
                    ):
                        _est_likely_et_5b = True
                if not _plur_subj_5b and not _est_likely_et_5b:
                    for candidate in generer_candidats_singulier_nom(curr):
                        c_infos = lexique.info(candidate)
                        if c_infos and any(
                            e.get("cgram") in ("ADJ", "VER", "AUX")
                            for e in c_infos
                        ):
                            ancien = result[i]
                            result[i] = candidate
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige=candidate,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="accord.nombre_attribut",
                                explication="Depluralization attribut apres copule singuliere",
                            ))
                            curr = result[i]
                            curr_low = curr.lower()
                            break

        # Regle 10 : Accord ADJ coordonne apres copule
        # "sont intelligentes et serieuse" -> "serieuses"
        if (
            i > 2
            and pos == "ADJ"
            and result[i - 1].lower() == "et"
        ):
            prev2_pos = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
            if prev2_pos == "ADJ":
                prev2_low = result[i - 2].lower()
                # Propager le nombre : si l'ADJ precedent est pluriel
                if (
                    prev2_low.endswith(("s", "x"))
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
                                regle="accord.nombre_adj",
                                explication="Accord ADJ coordonne (pluriel)",
                            ))
                            curr = result[i]
                            curr_low = curr.lower()
                            break

                # Propager le genre : chercher copule + sujet plus a gauche
                # ou propager depuis le premier ADJ coordonne
                if lexique is not None:
                    target_genre = None
                    # Chercher la copule avant l'ADJ coordonne
                    copule_idx = None
                    for _k in range(i - 3, max(-1, i - 6), -1):
                        if result[_k].lower() in COPULES_ALL:
                            copule_idx = _k
                            break
                    if copule_idx is not None:
                        gn = trouver_sujet_genre_nombre(
                            result, pos_tags, morpho, copule_idx, lexique,
                        )
                        if gn is not None:
                            s_genre, _s_nombre = gn
                            if s_genre == "Fem":
                                target_genre = "f"
                    # Fallback: propager le genre du premier ADJ coordonne
                    if target_genre is None:
                        prev2_infos = lexique.info(result[i - 2])
                        prev2_genred = [e for e in prev2_infos if e.get("genre")]
                        if prev2_genred and all(
                            e.get("genre") == "f" for e in prev2_genred
                        ):
                            target_genre = "f"
                    if target_genre == "f":
                        adj_infos = lexique.info(result[i])
                        adj_genred = [e for e in adj_infos if e.get("genre")]
                        if adj_genred and all(
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
                                        regle="accord.genre_adj",
                                    explication="Accord ADJ coordonne en genre (fem)",
                                    ))
                                    curr = result[i]
                                    curr_low = curr.lower()
                                    break

        # Regle 6 : DET feminin + NOM/ADJ masculin -> forme feminine
        if i > 0 and pos in ("NOM", "ADJ"):
            prev_low = result[i - 1].lower()
            if prev_low in SING_FEM_DET and lexique is not None:
                infos = lexique.info(result[i])
                genred = [e for e in infos if e.get("genre")]
                if genred and all(e.get("genre") == "m" for e in genred):
                    _found_r6 = False
                    # Try formes_de first (handles irregulars: beau→belle)
                    if hasattr(lexique, "formes_de"):
                        _lem_r6 = genred[0].get("lemme") or genred[0].get("lemma") or result[i].lower()
                        for f in lexique.formes_de(_lem_r6):
                            if (
                                f.get("genre") in ("f", "feminin", "Fem")
                                and f.get("nombre") in ("s", "singulier", "Sing")
                                and f.get("cgram") in ("ADJ", "NOM")
                            ):
                                cand = f.get("ortho", "")
                                if cand and cand.lower() != result[i].lower():
                                    ancien = result[i]
                                    result[i] = transferer_casse(ancien, cand)
                                    corrections.append(Correction(
                                        index=i,
                                        original=ancien,
                                        corrige=result[i],
                                        type_correction=TypeCorrection.GRAMMAIRE,
                                        regle="accord.genre_adj",
                                        explication=f"Accord en genre apres '{prev_low}'",
                                    ))
                                    _found_r6 = True
                                    break
                    # Fallback: suffix-based candidates
                    if not _found_r6:
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
                                    regle="accord.genre_adj",
                                    explication=f"Accord en genre apres '{prev_low}'",
                                ))
                                _found_r6 = True
                                break

        # Regle 11 : DET singulier + NOM/ADJ pluriel → depluralize
        # "une parties" → "une partie", "le généraux" → "le général"
        # Guard: ne pas depluralize si le mot est invariable
        # Guard: ne pas depluralize si le DET est aussi dans PLUR_DET
        # Guard: ne pas depluralize si le mot-avec-s a des entrees singulier
        #        (mot invariable en -s : fonds, bras, temps, corps)
        if i > 0 and pos in ("NOM", "ADJ"):
            prev_low = result[i - 1].lower()
            if (
                prev_low in SING_DET
                and prev_low not in PLUR_DET
                and curr_low.endswith(("s", "x", "z"))
                and len(curr) > 2
                and curr_low not in INVARIABLES
                and "-" not in curr  # compound words (lance-flammes)
                and lexique is not None
            ):
                _is_sing_with_s_r11 = False
                _curr_infos_r11 = lexique.info(curr_low)
                # Guard NOM PROPRE: pure proper nouns (nantes, reims)
                if _curr_infos_r11 and all(
                    "PROPRE" in (e.get("cgram") or "")
                    for e in _curr_infos_r11
                ):
                    _is_sing_with_s_r11 = True
                # Guard OOV: word not in lexique → likely proper noun
                # (mariannes, etc.)
                if not _curr_infos_r11:
                    _is_sing_with_s_r11 = True
                # Guard NOM PROPRE + low NOM freq: word primarily used
                # as proper noun (mans=Le Mans, fars=Le Fars, limoges)
                if (
                    _curr_infos_r11
                    and any(
                        "PROPRE" in (e.get("cgram") or "")
                        for e in _curr_infos_r11
                    )
                    and max(
                        (float(e.get("freq", 0) or 0)
                         for e in _curr_infos_r11
                         if e.get("cgram") in ("NOM", "ADJ")),
                        default=0,
                    ) < 3.0
                ):
                    _is_sing_with_s_r11 = True
                if _curr_infos_r11 and any(
                    e.get("cgram") in ("NOM", "ADJ")
                    and e.get("nombre") in ("singulier", "s")
                    for e in _curr_infos_r11
                ):
                    _is_sing_with_s_r11 = True
                if not _is_sing_with_s_r11:
                    for candidate in generer_candidats_singulier_nom(curr):
                        if lexique.existe(candidate):
                            ancien = result[i]
                            result[i] = candidate
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige=candidate,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="accord.nombre_nom",
                                explication=f"Depluralization apres DET singulier '{prev_low}'",
                            ))
                            break

        # Regle 11b : "ces" + NOM singulier → "cette" (fem) / "ce" (masc)
        # "ces disposition" → "cette disposition", "ces enfant" �� "cet enfant"
        if curr_low == "ces" and pos in ("ADJ:dem", "DET:dem", "DET"):
            if i + 1 < n and lexique is not None:
                _next_11b = result[i + 1].lower()
                _next_pos_11b = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                # Guard: verbe pluriel dans les 3 positions suivantes
                _plur_verb_11b = False
                for _k11b in range(i + 2, min(n, i + 4)):
                    _kw11b = result[_k11b].lower()
                    _kp11b = pos_tags[_k11b] if _k11b < len(pos_tags) else ""
                    if _kp11b in ("VER", "AUX") and _kw11b.endswith(("ent", "ont")):
                        _plur_verb_11b = True
                        break
                if (
                    _next_pos_11b == "NOM"
                    and not _next_11b.endswith(("s", "x", "z"))
                    and len(_next_11b) > 2
                    and not _plur_verb_11b
                ):
                    _infos_11b = lexique.info(result[i + 1])
                    _nom_genres = [
                        e.get("genre") for e in _infos_11b
                        if e.get("cgram") == "NOM"
                    ]
                    if _nom_genres:
                        if all(g == "f" for g in _nom_genres):
                            _repl = "cette"
                        elif all(g == "m" for g in _nom_genres):
                            _repl = "ce"
                        else:
                            _repl = None  # ambigu
                        if _repl:
                            ancien = result[i]
                            result[i] = transferer_casse(curr, _repl)
                            corrections.append(Correction(
                                index=i,
                                original=ancien,
                                corrige=result[i],
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="accord.genre_det",
                                explication=f"'ces' -> '{_repl}' (NOM singulier)",
                            ))

        # Regle 12 : "pars" / "surs" en contexte prepositionnel → "par" / "sur"
        # L'apprenant ajoute un -s superflu a une preposition.
        # Guard: "tu pars" = VER (correct), "pars le mur" = PRE (erreur).
        if curr_low == "pars" and i + 1 < n:
            next_pos_12 = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
            next_low_12 = result[i + 1].lower()
            _pars_is_pre = (
                next_pos_12 in (
                    "ART:def", "ART:ind", "ART", "DET", "DET:dem",
                    "NOM", "NOM PROPRE", "ADJ:pos",
                )
                or next_low_12 in (
                    "le", "la", "les", "l'", "un", "une", "des",
                    "ce", "cette", "ces", "son", "sa", "ses",
                )
                # OOV (unknown POS) — likely proper noun after preposition
                or next_pos_12 == ""
            )
            # Guard: "je/tu pars" — sujet 1sg/2sg avant
            if _pars_is_pre and i > 0:
                prev_low_12 = result[i - 1].lower()
                if prev_low_12 in ("tu", "je", "j'"):
                    _pars_is_pre = False
            if _pars_is_pre:
                ancien = result[i]
                result[i] = transferer_casse(curr, "par")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.preposition",
                    explication="'pars' -> 'par' (preposition)",
                ))

        if curr_low in ("surs", "sûrs", "sûr") and i + 1 < n:
            next_pos_12 = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
            next_low_12 = result[i + 1].lower()
            _surs_is_pre = (
                next_pos_12 in (
                    "ART:def", "ART:ind", "ART", "DET", "DET:dem",
                    "NOM", "NOM PROPRE", "ADJ:pos",
                )
                or next_low_12 in (
                    "le", "la", "les", "l'", "un", "une", "des",
                    "ce", "cette", "ces", "son", "sa", "ses",
                )
            )
            if _surs_is_pre:
                ancien = result[i]
                result[i] = transferer_casse(curr, "sur")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.preposition",
                    explication="'surs' -> 'sur' (preposition)",
                ))

        # Regle 13 : "a/ont étés" → "a/ont été" (PP invariable apres avoir)
        if curr_low == "étés" and i > 0:
            # Chercher un auxiliaire avoir avant (en sautant ADV)
            _has_avoir = False
            for _k in range(i - 1, max(-1, i - 4), -1):
                _kw = result[_k].lower()
                _kp = pos_tags[_k] if _k < len(pos_tags) else ""
                if _kw in ("a", "ai", "as", "avons", "avez", "ont",
                           "avait", "avais", "avaient", "avions", "aviez",
                           "aura", "auras", "auront", "aurons", "aurez",
                           "aurait", "aurais", "auraient"):
                    _has_avoir = True
                    break
                if _kp == "ADV":
                    continue
                break
            if _has_avoir:
                ancien = result[i]
                result[i] = transferer_casse(curr, "été")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.nombre_nom",
                    explication="'étés' -> 'été' (PP invariable apres avoir)",
                ))

        # Regle 14 : DET singulier + VER en -ent → NOM
        # "une listent" → "une liste", "le groupent" → "le groupe"
        # Guard: mot doit etre tague VER et finir par -ent
        # Guard: ne pas toucher les mots courts (<= 4 chars)
        # Guard: si le mot-ent existe comme NOM (sergent, vent, dent)
        _skip_r14_nom = False
        if (
            i > 0
            and pos in ("VER", "AUX")
            and curr_low.endswith("ent")
            and len(curr_low) > 4
            and lexique is not None
        ):
            # Guard: word itself is already a NOM in lexique
            _r14_infos = lexique.info(curr_low)
            if _r14_infos and any(
                e.get("cgram") == "NOM" for e in _r14_infos
            ):
                _skip_r14_nom = True
            prev_low_14 = result[i - 1].lower()
            if prev_low_14 in SING_DET and prev_low_14 not in PLUR_DET and not _skip_r14_nom:
                # Essayer -ent → -e (1er groupe: listent→liste)
                _nom_candidates = []
                _base_ent = curr_low[:-3]
                if _base_ent:
                    _nom_candidates.append(_base_ent + "e")
                    _nom_candidates.append(_base_ent)
                for _nom_cand in _nom_candidates:
                    _cand_infos = lexique.info(_nom_cand)
                    _is_nom = any(
                        e.get("cgram") == "NOM"
                        for e in (_cand_infos if _cand_infos else [])
                    )
                    if _is_nom:
                        ancien = result[i]
                        result[i] = transferer_casse(curr, _nom_cand)
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=result[i],
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="accord.det_nom_ver",
                            explication=f"DET sing + VER -ent → NOM '{_nom_cand}'",
                        ))
                        break

        # Regle 15 : "pendants" en contexte prepositionnel → "pendant"
        if curr_low == "pendants" and i + 1 < n:
            next_pos_15 = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
            _pend_is_pre = (
                next_pos_15 in (
                    "ART:def", "ART:ind", "ART", "DET", "DET:dem",
                    "NOM", "NOM PROPRE", "ADJ:pos", "ADJ:num",
                )
                or result[i + 1].lower() in (
                    "le", "la", "les", "l'", "un", "une", "des",
                    "ce", "cette", "ces", "son", "sa", "ses",
                    "plusieurs", "quelques", "la",
                )
            )
            if _pend_is_pre:
                ancien = result[i]
                result[i] = transferer_casse(curr, "pendant")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.preposition",
                    explication="'pendants' -> 'pendant' (preposition)",
                ))

        # Regle 16 : "toutes" + DET singulier → "toute"
        # "ils rassemblent toutes la population" → "toute la population"
        # Guard: "toutes les" = correct (pluriel)
        if curr_low == "toutes" and i + 1 < n:
            next_low_16 = result[i + 1].lower()
            if next_low_16 in SING_DET and next_low_16 not in PLUR_DET:
                ancien = result[i]
                result[i] = transferer_casse(curr, "toute")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.nombre_adj",
                    explication="'toutes' -> 'toute' (devant DET singulier)",
                ))

        # Regle 17 : PRO sujet P3s + NOM en -s → VER P3s
        # "elle comptes" → "elle compte", "il combats" → "il combat"
        # L'apprenant ajoute un -s a un verbe conjugue P3s.
        _P3S_PRONOMS = frozenset({"il", "elle", "on"})
        _r17_cond = (
            i > 0
            and pos == "NOM"
            and curr_low.endswith("s")
            and not curr_low.endswith("ss")
            and len(curr) > 3
            and lexique is not None
        )
        if _r17_cond:
            # Chercher un pronom sujet P3s en sautant COD/negation
            _pro_p3s = False
            for _k in range(i - 1, max(-1, i - 4), -1):
                _kw = result[_k].lower()
                if _kw in _P3S_PRONOMS:
                    _pro_p3s = True
                    break
                if _kw in ("y", "en", "ne", "n'", "se", "s'", "le", "la",
                            "l'", "me", "m'", "te", "t'", "lui"):
                    continue
                break
            if _pro_p3s:
                _ver_cand = curr_low[:-1]  # remove trailing -s
                _ver_infos = lexique.info(_ver_cand)
                _is_p3s = any(
                    e.get("cgram") in ("VER", "AUX")
                    and str(e.get("personne")) == "3"
                    and e.get("nombre") in ("singulier", "s")
                    for e in _ver_infos
                )
                if _is_p3s:
                    ancien = result[i]
                    result[i] = transferer_casse(curr, _ver_cand)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="accord.sujet_verbe",
                        explication="PRO P3s + NOM -s → VER P3s",
                    ))

        # Regle 18 : "en" + NOM pluriel → singulier (expressions figees)
        # "en outres" → "en outre", "en passants" → "en passant"
        if (
            i > 0
            and curr_low.endswith(("s", "x"))
            and len(curr) > 3
            and result[i - 1].lower() == "en"
            and lexique is not None
        ):
            _sing_18 = curr_low[:-1] if curr_low.endswith("s") else curr_low[:-1]
            # Essayer candidats singulier
            for _cand_18 in generer_candidats_singulier_nom(curr):
                if not lexique.existe(_cand_18):
                    continue
                _cand_low = _cand_18.lower()
                # Le singulier doit etre une expression connue apres "en"
                if _cand_low in (
                    "outre", "passant", "semaine", "sprint",
                    "ensemble", "union", "oeuvre", "œuvre",
                    "partie", "revanche", "pratique", "principe",
                    "general", "général", "particulier", "moyenne",
                    "effet", "fait", "tout", "vain", "commun",
                ):
                    ancien = result[i]
                    result[i] = _cand_18
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=_cand_18,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="accord.expression",
                        explication=f"'en {ancien}' -> 'en {_cand_18}' (expression)",
                    ))
                    break

        # Regle 19 : Expressions figees — formes invariables
        # 19a: "en fin de comptes" → "en fin de compte"
        if curr_low == "comptes" and i >= 3:
            _ctx19a = (result[i-3].lower(), result[i-2].lower(), result[i-1].lower())
            if _ctx19a == ("en", "fin", "de"):
                ancien = result[i]
                result[i] = transferer_casse(curr, "compte")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.expression",
                    explication="'en fin de comptes' -> 'en fin de compte' (expression)",
                ))

        # 19b: "avoir lieus" → "avoir lieu" (lieu invariable dans "avoir lieu")
        if curr_low == "lieus" and i > 0:
            _avoir_19b = False
            for _k19 in range(i - 1, max(-1, i - 4), -1):
                _kw19 = result[_k19].lower()
                if _kw19 in ("a", "ai", "as", "avons", "avez", "ont",
                             "avait", "avais", "avaient", "avions", "aviez",
                             "aura", "auras", "auront", "aurons", "aurez",
                             "aurait", "aurais", "auraient", "aurai",
                             "eu", "ayant"):
                    _avoir_19b = True
                    break
                _kp19 = pos_tags[_k19] if _k19 < len(pos_tags) else ""
                if _kp19 == "ADV" or _kw19 in ("y", "en", "ne", "n'"):
                    continue
                break
            if _avoir_19b:
                ancien = result[i]
                result[i] = transferer_casse(curr, "lieu")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.expression",
                    explication="'avoir lieus' -> 'avoir lieu' (expression)",
                ))

        # 19c: "sur glaces" → "sur glace" (expression)
        if curr_low == "glaces" and i > 0 and result[i - 1].lower() == "sur":
            ancien = result[i]
            result[i] = transferer_casse(curr, "glace")
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige=result[i],
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="accord.expression",
                explication="'sur glaces' -> 'sur glace' (expression)",
            ))

        # 19d: "faire parties de" → "faire partie de"
        _FAIRE_FORMES = frozenset({
            "fait", "fais", "faisons", "faites", "font",
            "faisait", "faisais", "faisaient", "faisions", "faisiez",
            "fera", "feras", "feront", "ferons", "ferez",
            "ferait", "ferais", "feraient",
            "fasse", "fasses", "fassent",
            "faisant",
        })
        if curr_low == "parties" and i > 0 and i + 1 < n:
            _prev19d = result[i - 1].lower()
            if _prev19d in _FAIRE_FORMES:
                _next19d = result[i + 1].lower()
                if _next19d in ("de", "du", "d'", "des", "d"):
                    ancien = result[i]
                    result[i] = transferer_casse(curr, "partie")
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="accord.expression",
                        explication="'faire parties' -> 'faire partie' (expression)",
                    ))

        # 19e: "contres" en contexte prepositionnel → "contre"
        if curr_low == "contres" and i + 1 < n:
            next_pos_19e = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
            next_low_19e = result[i + 1].lower()
            _contres_is_pre = (
                next_pos_19e in (
                    "ART:def", "ART:ind", "ART", "DET", "DET:dem",
                    "NOM", "NOM PROPRE", "ADJ:pos", "ADJ:num",
                )
                or next_low_19e in (
                    "le", "la", "les", "l'", "un", "une", "des",
                    "ce", "cette", "ces", "son", "sa", "ses",
                )
            )
            if _contres_is_pre:
                ancien = result[i]
                result[i] = transferer_casse(curr, "contre")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.preposition",
                    explication="'contres' -> 'contre' (preposition)",
                ))

        # 19f: "fautes de" → "faute de" (expression, quand pas de DET avant)
        if curr_low == "fautes" and i + 1 < n:
            _next19f = result[i + 1].lower()
            if _next19f in ("de", "du", "d'", "d"):
                # Guard: si DET avant, c'est un NOM pluriel valide
                # ("les fautes de grammaire" = correct)
                _has_det_19f = False
                if i > 0:
                    _prev19f = result[i - 1].lower()
                    _prev_pos_19f = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                    if (
                        _prev19f in PLUR_DET
                        or _prev19f in SING_DET
                        or _prev_pos_19f.startswith(("ART", "DET"))
                        or _prev19f in ("ses", "nos", "vos", "leurs",
                                        "mes", "tes", "ces", "quelques")
                    ):
                        _has_det_19f = True
                if not _has_det_19f:
                    ancien = result[i]
                    result[i] = transferer_casse(curr, "faute")
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="accord.expression",
                        explication="'fautes de' -> 'faute de' (expression)",
                    ))

    # Regle 20 : "des" + ADJ + NOM pluriel → "de" + ADJ + NOM
    # French grammar: "des" becomes "de" before plural ADJ + NOM
    # "des nouvelles chaussures" → "de nouvelles chaussures"
    # "des bons gâteaux" → "de bons gâteaux"
    for i in range(n - 2):
        if result[i].lower() != "des":
            continue
        _pos_i20 = pos_tags[i] if i < len(pos_tags) else ""
        if not _pos_i20.startswith(("ART", "DET")):
            continue
        _pos_next20 = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
        _pos_next2_20 = pos_tags[i + 2] if i + 2 < len(pos_tags) else ""
        # Accept ADJ tag or NOM tag with ADJ entries in lexique
        _is_adj_r20 = _pos_next20 == "ADJ"
        if not _is_adj_r20 and _pos_next20 == "NOM" and lexique is not None:
            _r20_infos = lexique.info(result[i + 1].lower())
            if any(
                (e.get("cgram") or "").startswith("ADJ")
                for e in _r20_infos
            ):
                _is_adj_r20 = True
        if _is_adj_r20 and _pos_next2_20 == "NOM":
            # Guard: NOM must be plural (ends in s/x/z)
            _nom_low20 = result[i + 2].lower()
            if _nom_low20.endswith(("s", "x", "z")) and len(_nom_low20) > 2:
                ancien = result[i]
                result[i] = transferer_casse(ancien, "de")
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="accord.des_de",
                    explication="'des' -> 'de' devant ADJ + NOM pluriel",
                ))

    return result, corrections


_PRONOMS_COD = frozenset({
    "le", "la", "l'", "les", "me", "m'", "te", "t'",
    "se", "s'", "lui", "leur", "nous", "vous", "en", "y",
    "ne", "n'",
})


def _trouver_sujet_pluriel(
    mots: list[str], pos_tags: list[str], idx_verbe: int,
) -> bool:
    """Cherche un determinant pluriel avant le verbe en sautant les complements.

    Saute les groupes prepositionnels PRE (+DET) (+NOM/ADJ).
    En balayant vers la gauche, l'ordre est NOM/ADJ ← DET ← PRE, donc
    quand on rencontre un DET singulier, on verifie si une preposition suit.
    """
    j = idx_verbe - 1
    # Sauter les pronoms COD/COI et negation entre sujet et verbe
    while j >= 0 and mots[j].lower() in _PRONOMS_COD:
        j -= 1
    while j >= 0:
        pos_j = pos_tags[j] if j < len(pos_tags) else ""
        mot_j = mots[j].lower()
        if pos_j in ("NOM", "ADJ"):
            # Demonstratives/possessives tagged ADJ act as singular DET
            # boundaries — they start a new GN (e.g. "Cette notion")
            if pos_j == "ADJ" and mot_j in SING_DET:
                return False
            j -= 1
            continue
        # Contractions prepositionnelles : sauter du/au/aux
        if mot_j in ("du", "au", "aux"):
            j -= 1
            continue
        if mot_j in PLUR_DET:
            # Verifier que ce n'est pas un DET dans un PP
            # (ex: "pres des maisons" — "des" est DET pluriel dans PP)
            if j > 0:
                prev_pos = pos_tags[j - 1] if j - 1 < len(pos_tags) else ""
                prev_mot = mots[j - 1].lower()
                if prev_pos == "PRE" or prev_mot in PREPOSITIONS or prev_mot == "des":
                    # "des" before a numeral/DET = "de+les" (PP marker)
                    j -= 2  # sauter DET + PRE
                    continue
                # ADJ/quantifier between PRE and DET:
                # "de toutes les figures" → j-1=toutes(ADJ), j-2=de(PRE)
                if prev_pos in ("ADJ", "ADJ:pos") and j > 1:
                    _pp2_pos_sp = pos_tags[j - 2] if j - 2 < len(pos_tags) else ""
                    _pp2_mot_sp = mots[j - 2].lower()
                    if _pp2_pos_sp == "PRE" or _pp2_mot_sp in PREPOSITIONS or _pp2_mot_sp == "des":
                        j -= 3  # sauter DET + ADJ + PRE
                        continue
                # "des" apres un NOM/ADJ = contraction "de+les" (PP)
                # Ex: "le directeur des ecoles", "l'aspect actuel des cactus"
                if mot_j == "des" and prev_pos in ("NOM", "ADJ", "NOM PROPRE"):
                    j -= 1  # sauter "des" (la PRE est incorporee)
                    continue
            # Guard: "un/une des NOM" = singulier (un des professeurs avait)
            if mot_j == "des" and j > 0 and mots[j - 1].lower() in ("un", "une", "l'un", "l'une"):
                return False
            # Guard: DET pluriel mais NOM singulier → desaccord DET/NOM
            # "ces espèce est" → ne pas pluraliser le verbe
            _noms_entre = [
                mots[k].lower()
                for k in range(j + 1, idx_verbe)
                if (pos_tags[k] if k < len(pos_tags) else "") in ("NOM", "ADJ")
            ]
            if _noms_entre and not any(w.endswith(("s", "x")) for w in _noms_entre):
                return False
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
