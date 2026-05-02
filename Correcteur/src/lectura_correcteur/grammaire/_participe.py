"""Regles de participe passe : auxiliaire + infinitif -> participe passe.

Inclut aussi l'accord du PP avec le sujet quand l'auxiliaire est etre.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    ALLER_FORMES,
    AUXILIAIRES,
    ETRE_FORMES,
    MODAUX_FORMES,
    generer_candidats_participe,
    generer_candidats_pp_accorde,
    trouver_sujet_genre_nombre,
)

# Terminaisons de PP (pour heuristique morpho sans POS)
_PP_SUFFIXES = ("é", "és", "ée", "ées", "i", "is", "ie", "ies",
                "u", "us", "ue", "ues", "it", "ite", "ites",
                "ert", "erte", "ertes", "erts", "oint", "ointe",
                "eint", "einte", "aint", "ainte", "ort", "orte")


def verifier_participes_passes(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Corrige les confusions infinitif/PP et present/PP.

    Patterns :
    - auxiliaire + infinitif -> participe passe ("j'ai manger" -> "j'ai mange")
    - auxiliaire + present 1er groupe -> PP ("a sonne" -> "a sonne")
    - modal + PP -> infinitif ("faut ecoute" -> "faut ecouter")
    """
    if not mots:
        return mots, []

    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    _TRANSPARENTS = frozenset({
        "ne", "n'", "pas", "plus", "jamais", "rien",
        "point", "y", "en",
        # Pronoms reflexifs (va se présenter, s'est fait)
        "se", "s'", "me", "m'", "te", "t'",
        # Adverbes courants entre auxiliaire et PP
        "bien", "déjà", "donc", "alors", "toujours",
        "aussi", "souvent", "encore", "trop", "tout",
        "même", "ensuite", "récemment", "largement",
        "longtemps", "vraiment", "seulement", "beaucoup",
        # Adverbes supplementaires (bloquaient le scan AUX)
        "très", "fortement", "également", "presque",
        "autrefois", "officiellement", "entièrement",
        "complètement", "spécialement", "généralement",
        "particulièrement", "principalement", "habituellement",
        "actuellement", "initialement", "finalement",
        "assez", "peu", "aujourd", "hui",
        # PP de etre (compound aux "a été")
        "été",
    })

    for i in range(n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]
        curr_low = curr.lower()

        # Chercher un auxiliaire ou modal avant le mot (en sautant ne/pas/y/en)
        aux_found = False
        aux_word = ""
        aux_idx = -1
        modal_found = False
        aller_found = False
        prep_inf_found = False
        ver_found = False
        _PREP_INF = frozenset({
            "pour", "sans", "afin", "à", "d'", "d\u2019",
        })
        for j in range(i - 1, max(-1, i - 6), -1):
            w = result[j].lower()
            if w in AUXILIAIRES:
                aux_found = True
                aux_word = w
                aux_idx = j
                break
            if w in MODAUX_FORMES:
                modal_found = True
                break
            if w in ALLER_FORMES:
                aller_found = True
                break
            if w in _PREP_INF:
                prep_inf_found = True
                break
            if w not in _TRANSPARENTS:
                # Verbe conjugue (non-aux/modal/aller) : le suivant
                # doit etre a l'infinitif, jamais au PP
                _w_pos_j = pos_tags[j] if j < len(pos_tags) else ""
                if _w_pos_j in ("VER", "AUX"):
                    # Guard: VER preceded by DET/possessive = likely NOM
                    # (e.g. "ses œuvres" tagged VER but is NOM plural)
                    _ver_after_det = False
                    if j > 0:
                        _pj_pos = pos_tags[j - 1] if j - 1 < len(pos_tags) else ""
                        if _pj_pos.startswith(("ART", "DET", "ADJ:pos")):
                            _ver_after_det = True
                    if not _ver_after_det:
                        ver_found = True
                break

        # --- Regle 1 : AUX + infinitif -> PP ---
        if aux_found and curr_low.endswith(("er", "ir", "re")):
            # Guard R1-prep-a: "a" en debut de phrase = preposition "à"
            # "A noter que" = "À noter que" (correct), pas "a noté que"
            # "A partir de" = "À partir de" (correct), pas "a parti de"
            _skip_r1_prep_a = False
            if aux_word.lower() == "a" and aux_idx == 0:
                _skip_r1_prep_a = True
            # Guard R1: si la forme -er est principalement NOM/ADJ dans
            # le lexique, c'est probablement un nom/adj, pas un infinitif
            # ("a été conseiller culturel" = NOM, pas "a été conseillé")
            _skip_r1_nom = False
            if lexique is not None and curr_low.endswith("er"):
                _r1_infos = lexique.info(curr_low)
                if _r1_infos:
                    _r1_best = max(_r1_infos, key=lambda e: float(e.get("freq") or 0))
                    if _r1_best.get("cgram") in ("NOM", "ADJ"):
                        # Override: strong etre forms (fut, sera, soit...)
                        # strongly signal PP context even for NOM-dominant words
                        # "fut mater" = was tamed, not "fut the mate"
                        _STRONG_ETRE = frozenset({
                            "fut", "fût", "sera", "serait", "soit",
                            "soient", "furent", "seront",
                        })
                        if aux_word in _STRONG_ETRE:
                            _skip_r1_nom = False
                        else:
                            _skip_r1_nom = True
            if not _skip_r1_nom and not _skip_r1_prep_a:
                candidats = generer_candidats_participe(curr)
                # Check if the -er form is a known verb (allows low-freq PP)
                _er_is_verb = False
                if lexique is not None and curr_low.endswith("er"):
                    _er_info_r1 = lexique.info(curr_low)
                    _er_is_verb = any(
                        e.get("cgram") in ("VER", "AUX")
                        for e in _er_info_r1
                    ) if _er_info_r1 else False
                for candidate in candidats:
                    if lexique is None or lexique.existe(candidate):
                        # Guard: PP candidate with zero frequency
                        # (leadé, carté etc. are not real PPs)
                        # Exception: if -er form is a known verb, allow
                        if (
                            lexique is not None
                            and hasattr(lexique, "frequence")
                            and lexique.frequence(candidate) < 0.05
                            and not _er_is_verb
                        ):
                            continue
                        ancien = result[i]
                        result[i] = transferer_casse(curr, candidate)
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=result[i],
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="participe.infinitif_vers_pp",
                            explication="Infinitif apres auxiliaire -> participe passe",
                        ))
                        break
                if result[i] != curr:
                    continue

        # --- Regle 2 : AUX + present 1er groupe (-e) -> PP (-e) ---
        # "a sonne" -> "a sonne" (present → PP pour 1er groupe)
        if aux_found and (pos == "VER" or curr_low.endswith(_PP_SUFFIXES)):
            if (
                curr_low.endswith("e")
                and not curr_low.endswith(("ee", "er", "re", "le", "ne", "se", "te", "ée", "ue", "ie"))
                and len(curr_low) > 2
            ):
                candidate = curr_low[:-1] + "é"
                if lexique is None or lexique.existe(candidate):
                    ancien = result[i]
                    result[i] = transferer_casse(curr, candidate)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="participe.infinitif_vers_pp",
                        explication="Present apres auxiliaire -> participe passe",
                    ))
                    continue

        # --- Regle 3 : Modal + PP (-e/-i/-u) -> infinitif ---
        # "faut ecoute" -> "faut ecouter", "doit fini" -> "doit finir"
        # "peut ajoutés" -> "peut ajouter", "doit restée" -> "doit rester"
        if modal_found and (pos == "VER" or curr_low.endswith(_PP_SUFFIXES)):
            candidate = None
            if curr_low.endswith("ées") and not curr_low.endswith("er"):
                candidate = curr_low[:-3] + "er"   # ajoutées → ajouter
            elif curr_low.endswith("és") and not curr_low.endswith("er"):
                candidate = curr_low[:-2] + "er"   # ajoutés → ajouter
            elif curr_low.endswith("ée") and not curr_low.endswith("er"):
                candidate = curr_low[:-2] + "er"   # restée → rester
            elif curr_low.endswith("é") and not curr_low.endswith("er"):
                candidate = curr_low[:-1] + "er"
            elif curr_low.endswith("i") and not curr_low.endswith("ir"):
                candidate = curr_low + "r"
            elif curr_low.endswith("u") and not curr_low.endswith(("re", "ur")):
                candidate = curr_low + "re"  # rendu -> rendre (not exact)
            if candidate and (lexique is None or lexique.existe(candidate)):
                ancien = result[i]
                result[i] = transferer_casse(curr, candidate)
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="participe.pp_vers_infinitif",
                    explication="PP apres modal -> infinitif",
                ))
                continue

        # --- Regle 4 : Aller + PP (-e) -> infinitif ---
        # (extension de la regle existante dans _homophones.py, ici plus large)
        # "va mangé" → "va manger", "va restée" → "va rester"
        if aller_found and pos == "VER" and not curr_low.endswith("er"):
            candidate = None
            if curr_low.endswith("ées"):
                candidate = curr_low[:-3] + "er"
            elif curr_low.endswith("és"):
                candidate = curr_low[:-2] + "er"
            elif curr_low.endswith("ée"):
                candidate = curr_low[:-2] + "er"
            elif curr_low.endswith("é"):
                candidate = curr_low[:-1] + "er"
            if candidate and (lexique is None or lexique.existe(candidate)):
                ancien = result[i]
                result[i] = transferer_casse(curr, candidate)
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="participe.pp_vers_infinitif",
                    explication="PP -> infinitif apres aller",
                ))
                continue

        # --- Regle 4c : Preposition + PP -> infinitif ---
        # "pour participé" → "pour participer", "sans hésité" → "sans hésiter"
        # "destinée à restée" → "destinée à rester"
        if prep_inf_found and not curr_low.endswith("er"):
            candidate = None
            if curr_low.endswith("ées"):
                candidate = curr_low[:-3] + "er"
            elif curr_low.endswith("és"):
                candidate = curr_low[:-2] + "er"
            elif curr_low.endswith("ée"):
                candidate = curr_low[:-2] + "er"
            elif curr_low.endswith("é"):
                candidate = curr_low[:-1] + "er"
            if candidate and (lexique is None or lexique.existe(candidate)):
                ancien = result[i]
                result[i] = transferer_casse(curr, candidate)
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="participe.pp_vers_infinitif",
                    explication="PP -> infinitif apres preposition",
                ))
                continue

        # --- Regle 4d : VER (non-aux) + PP → infinitif ---
        # "il aime mangé" → "il aime manger"
        # Quand deux verbes se suivent (le premier n'etant pas un auxiliaire),
        # le second est toujours a l'infinitif.
        # Guard POS: le mot courant doit etre tague VER (pas ADJ)
        # pour eviter les attributs du sujet ("il rentre fatigué")
        if ver_found and pos in ("VER", "AUX") and not curr_low.endswith(("er", "ir", "re", "oir")):
            # Guard: PP + "par" = passive voice (e.g. "créé par")
            _next_is_par = (
                i + 1 < n
                and result[i + 1].lower() == "par"
            )
            # Guard: word preceded by preposition = NOM in PP
            # "tonnent en plongée" → "plongée" is NOM, not PP
            _after_prep_4d = (
                i > 0
                and (pos_tags[i - 1] if i - 1 < len(pos_tags) else "")
                == "PRE"
            )
            candidate = None
            if not _next_is_par and not _after_prep_4d:
                if curr_low.endswith("ées"):
                    candidate = curr_low[:-3] + "er"
                elif curr_low.endswith("és"):
                    candidate = curr_low[:-2] + "er"
                elif curr_low.endswith("ée"):
                    candidate = curr_low[:-2] + "er"
                elif curr_low.endswith("é"):
                    candidate = curr_low[:-1] + "er"
            if candidate and (lexique is None or lexique.existe(candidate)):
                ancien = result[i]
                result[i] = transferer_casse(curr, candidate)
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="participe.pp_vers_infinitif",
                    explication="PP -> infinitif apres verbe conjugue",
                ))
                continue

        # --- Regle 4b : AUX avoir + PP feminin → PP masculin singulier ---
        # "il a reçue" → "reçu", "avoir accueillie" → "accueilli"
        # Le PP apres avoir ne s'accorde pas (sauf COD avant, non gere ici)
        # Guard: si "été" est entre l'aux et le mot, c'est un passif
        # "a été conçue" → ne PAS strip (PP accorde avec sujet via être)
        _ete_between = False
        if aux_found and aux_word not in ETRE_FORMES:
            for _k4b in range(i - 1, max(-1, i - 4), -1):
                _w4b = result[_k4b].lower()
                if _w4b == "\xe9t\xe9":  # "été"
                    _ete_between = True
                    break
                if _w4b not in _TRANSPARENTS:
                    break
        _NOT_PP_4B = frozenset({
            "plus", "nous", "vous", "tous", "dessus", "dessous",
            "refus", "abus", "jus", "pus", "sous", "inclus",
            "motus", "couscous", "campus", "bonus", "cactus",
        })
        if aux_found and aux_word not in ETRE_FORMES and not _ete_between and lexique is not None and curr_low not in _NOT_PP_4B:
            _accorde_pp = False
            _pp_base = ""
            # Feminine forms → masculine singular
            if curr_low.endswith("\xe9e") and not curr_low.endswith("er"):
                _pp_base = curr_low[:-1]  # -ée → -é
                _accorde_pp = True
            elif curr_low.endswith("ie") and not curr_low.endswith(("rie", "lie", "nie", "sie", "tie")):
                _pp_base = curr_low[:-1]  # -ie → -i
                _accorde_pp = True
            elif curr_low.endswith("ue") and not curr_low.endswith(("que", "gue")):
                _pp_base = curr_low[:-1]  # -ue → -u
                _accorde_pp = True
            elif curr_low.endswith("\xe9es"):
                _pp_base = curr_low[:-2]  # -ées → -é
                _accorde_pp = True
            elif curr_low.endswith("ies") and not curr_low.endswith("ries"):
                _pp_base = curr_low[:-2]  # -ies → -i
                _accorde_pp = True
            # Masculine plural forms → masculine singular
            elif curr_low.endswith("\xe9s") and len(curr_low) > 3:
                _pp_base = curr_low[:-1]  # -és → -é
                _accorde_pp = True
            elif curr_low.endswith("us") and not curr_low.endswith(("ous", "nus")):
                _pp_base = curr_low[:-1]  # -us → -u (eus→eu, reçus→reçu)
                _accorde_pp = True
            elif curr_low.endswith("is") and len(curr_low) > 3 and not curr_low.endswith(("ois", "ais")):
                _pp_base = curr_low[:-1]  # -is → -i (partis→parti)
                _accorde_pp = True
            if _accorde_pp and _pp_base:
                _base_infos = lexique.info(_pp_base)
                if _base_infos and any(
                    e.get("cgram") in ("VER", "AUX") for e in _base_infos
                ):
                    ancien = result[i]
                    result[i] = transferer_casse(curr, _pp_base)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="participe.base_avoir",
                        explication="PP accorde -> base apres avoir",
                    ))
                    continue

        # === Regles 5-7 : INF → PP sans auxiliaire visible ===
        # Ciblent les cas ou un infinitif en -er devrait etre un PP
        # (apposition, adjectif verbal, nom derive).
        if (
            not aux_found and not modal_found and not aller_found
            and curr_low.endswith("er") and len(curr_low) > 3
            and lexique is not None
        ):
            # Guard global: -er doit exister dans le lexique
            _er_infos = lexique.info(curr_low)
            _skip_r567 = not _er_infos
            # Guard global: si forme -er est principalement NOM ou ADJ,
            # ne pas convertir (conseiller=NOM, fier=ADJ)
            if not _skip_r567:
                _er_best = max(
                    _er_infos, key=lambda e: float(e.get("freq") or 0),
                )
                if _er_best.get("cgram") in ("NOM", "ADJ", "NOM PROPRE"):
                    _skip_r567 = True
            # Guard: -er with significant NOM frequency (leader, poster)
            # even if VER is slightly higher-freq
            if not _skip_r567:
                _er_nom_freq = max(
                    (float(e.get("freq") or 0)
                     for e in _er_infos
                     if e.get("cgram") == "NOM"),
                    default=0.0,
                )
                if _er_nom_freq > 5.0:
                    _skip_r567 = True
            # Guard: NOM PROPRE entry with non-trivial freq → proper name
            # (carter, etc.). Skip entries with freq=0 (phantom PROPRE tags
            # on common verbs like "publier", "allier")
            if not _skip_r567 and any(
                "PROPRE" in (e.get("cgram") or "")
                and float(e.get("freq") or 0) > 0.5
                for e in _er_infos
            ):
                _skip_r567 = True
            # Guard: next word is all-caps SIGLE → foreign/technical
            # "le user ID", "le master CPU" → not French -er verb
            if not _skip_r567 and i + 1 < n:
                _next_r567 = result[i + 1]
                if len(_next_r567) >= 2 and _next_r567.isupper():
                    _skip_r567 = True

            if not _skip_r567:
                prev_pos = pos_tags[i - 1] if 0 < i <= len(pos_tags) else ""
                prev_low = result[i - 1].lower() if i > 0 else ""

                # Guard: prev est fragment d'elision (d, l, n, s)
                # → PRE/PRO context, INF est correct
                _prev_is_elision = (
                    len(prev_low) == 1 and prev_low in "dlns"
                )

                _r5_fire = False

                if not _prev_is_elision:
                    # --- Regle 5 : NOM/ADJ + -er → PP (apposition) ---
                    # "" ou "?" = mot inconnu du lexique, souvent NOM PROPRE
                    # Guard: abbreviations courtes (rdc, sc) ne sont pas des NOM
                    _prev_is_nominal_r5 = prev_pos in (
                        "NOM", "ADJ", "?",
                    ) or (prev_pos == "" and len(prev_low) > 3)
                    if _prev_is_nominal_r5:
                        _percep_before = False
                        # Guard: prev itself is a causative/perception verb
                        # tagged NOM (le faire, un laisser-faire)
                        _PERCEP_VERBS = frozenset({
                            "fait", "faire", "fais", "font", "faisant",
                            "laisse", "laisser", "laissé",
                            "voit", "voir", "vu",
                            "entend", "entendre", "entendu",
                        })
                        if prev_low in _PERCEP_VERBS:
                            _percep_before = True
                        if i > 1:
                            _p2_pos = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                            _p2_low = result[i - 2].lower()
                            if _p2_pos in ("VER", "AUX") and _p2_low in _PERCEP_VERBS:
                                _percep_before = True
                        # Guard R5: si mot APRES -er est DET/PRO →
                        # c'est probablement un objet de l'INF
                        _has_object = False
                        if i + 1 < n:
                            _next_pos_r5 = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                            _next_low_r5 = result[i + 1].lower()
                            if (
                                _next_pos_r5.startswith((
                                    "ART", "DET", "ADJ:pos", "ADJ:dem",
                                ))
                                or _next_pos_r5 == "PRO:per"
                                or _next_low_r5 in (
                                    "le", "la", "l", "les", "un", "une",
                                    "des", "du", "son", "sa", "ses",
                                    "leur", "leurs", "ce", "cet", "cette", "ces",
                                )
                            ):
                                _has_object = True
                        if not _percep_before and not _has_object:
                            _r5_fire = True

                    # --- Regle 6 : DET + -er → nom en -é/-ée ---
                    if prev_pos.startswith(("ART", "DET")) or prev_low in (
                        "le", "la", "l", "un", "une", "les", "des", "du",
                        "ce", "cet", "cette", "ces", "son", "sa", "ses",
                        "mon", "ma", "mes", "ton", "ta", "tes",
                    ):
                        # Guard R6: -er a aussi des entrees NOM
                        # → c'est un nom (le toucher, un carter)
                        _er_has_nom = any(
                            e.get("cgram") == "NOM" for e in _er_infos
                        )
                        # Guard R6: PRE introduisant un INF avant le DET
                        # → "de le blesser" = PRO context, INF correct
                        # Seulement les PRE introduisant un INF (de/à/pour/sans)
                        # pas les PRE locatifs (sur/dans/en/entre/vers)
                        _pre_before_det = False
                        if i > 1:
                            _p2_low_r6 = result[i - 2].lower()
                            if _p2_low_r6 in (
                                "de", "d", "\xe0", "pour", "sans",
                            ):
                                _pre_before_det = True
                        # Guard R6: subject PRO or inversion before DET
                        # → DET is pronoun object, not article
                        # "faut-il le rappeler" = inversion, "le" is PRO
                        # Note: VER + "la" + -er can be VER + ART + NOM
                        # ("permet la traverser" = "permet la traversée")
                        # so we only block for subject pronouns/inversions
                        _pronoun_det = False
                        if i > 1 and prev_low in ("le", "la", "l", "les"):
                            _p2_low_r6b = result[i - 2].lower()
                            _p2_pos_r6b = (
                                pos_tags[i - 2]
                                if i - 2 < len(pos_tags) else ""
                            )
                            # Hyphenated inversion: "faut-il", "peut-on"
                            if "-" in _p2_low_r6b and _p2_low_r6b.endswith(
                                ("-il", "-elle", "-on", "-je", "-tu",
                                 "-ils", "-elles", "-nous", "-vous"),
                            ):
                                _pronoun_det = True
                            # Subject pronoun at i-2 → le/la/les is COD
                            elif _p2_low_r6b in (
                                "il", "elle", "on", "je", "j",
                                "tu", "ils", "elles",
                            ):
                                _pronoun_det = True
                            # VER/AUX at i-2 + "les": COD pronoun
                            # "tuer est les mutiler", "donc les remplacer"
                            # Note: only "les" — "la"/"le" can be DET+NOM
                            # ("permet la traverser" = DET + NOM -ée)
                            elif (
                                prev_low == "les"
                                and _p2_pos_r6b in ("VER", "AUX")
                            ):
                                _pronoun_det = True
                            elif (
                                prev_low == "les"
                                and _p2_pos_r6b == "ADV"
                                and i > 2
                            ):
                                _p3_pos_r6b = (
                                    pos_tags[i - 3]
                                    if i - 3 < len(pos_tags) else ""
                                )
                                if _p3_pos_r6b in ("VER", "AUX"):
                                    _pronoun_det = True
                            # Reflexive at i-2 → le/la is object
                            # "se le masser" = reflexive context
                            elif _p2_low_r6b in (
                                "se", "s'", "s\u2019",
                                "me", "m'", "te", "t'",
                            ):
                                _pronoun_det = True
                            # aller at i-2 → le/la is object
                            # "aller le récupérer" = aller + COD + INF
                            elif (
                                _p2_low_r6b in ALLER_FORMES
                                or _p2_low_r6b == "aller"
                            ):
                                _pronoun_det = True
                        if not _er_has_nom and not _pre_before_det and not _pronoun_det:
                            _r5_fire = True

                    # --- Regle 7 : PP + "et"/"puis"/"ni" + -er → PP coord ---
                    if prev_low in ("et", "puis", "ni") and i > 1:
                        _p2_low = result[i - 2].lower()
                        _p2_pos = pos_tags[i - 2] if i - 2 < len(pos_tags) else ""
                        if (
                            _p2_pos in ("VER", "ADJ", "NOM")
                            and _p2_low.endswith(_PP_SUFFIXES)
                        ):
                            _r5_fire = True

                    # --- Regle 5b : ADV de degre + -er → PP ---
                    # "très limiter" → "très limité", "largement transformer"
                    # Guard: "ne...plus" = negation, not degree ADV
                    if prev_pos == "ADV" and prev_low in (
                        "très", "largement", "entièrement", "tant",
                        "déjà", "davantage", "autrefois",
                        "fortement", "récemment", "officiellement",
                        "usuellement", "rarement", "localement",
                    ):
                        _r5_fire = True
                    # "le plus huppé", "la plus élevée" = superlatif
                    if prev_low == "plus" and i > 1:
                        _p2_low_5b = result[i - 2].lower()
                        if _p2_low_5b in (
                            "le", "la", "les", "l",
                            "au", "du", "des",
                        ):
                            _r5_fire = True

                    # --- Regle 9 : ET/NI + -er + PREP → PP ---
                    # "et transférer à" → "et transférée à"
                    # Guard: prev2 ne doit pas etre VER/AUX (coord d'INF)
                    _PREP_R9 = frozenset({
                        "à", "au", "aux", "dans", "en", "sur",
                        "comme", "depuis", "vers", "entre",
                        "sous", "contre", "chez",
                    })
                    if (
                        prev_low in ("et", "ou", "ni", "puis")
                        and i + 1 < n
                        and result[i + 1].lower() in _PREP_R9
                    ):
                        _block_r9 = False
                        if i > 1:
                            _p2_pos_r9 = (
                                pos_tags[i - 2]
                                if i - 2 < len(pos_tags)
                                else ""
                            )
                            if _p2_pos_r9 in ("VER", "AUX"):
                                _block_r9 = True
                        if not _block_r9:
                            _r5_fire = True

                    # --- Regle 8 : -er + "par" → PP (passif) ---
                    # "assuré par", "marqué par", "supplanté par"
                    # Guard: skip causative "fait/faire" + -er + par
                    # Guard: skip common infinitive expressions (commencer par)
                    # Guard: skip if preceded by PRE (à/de/pour/sans)
                    _VERBES_PAR = frozenset({
                        "commencer", "finir", "passer", "entrer",
                    })
                    # Guard: "par exemple" is not passive
                    _par_exemple = (
                        i + 2 < n
                        and result[i + 1].lower() == "par"
                        and result[i + 2].lower() == "exemple"
                    )
                    # Guard: scan back past CON for causative (faire/laisser)
                    # "faire réformer ou annuler par" → causative context
                    _causative_back = False
                    if prev_low in ("et", "ou", "ni", "puis"):
                        for _kb in range(i - 2, max(-1, i - 8), -1):
                            _kb_low = result[_kb].lower()
                            _kb_pos = (
                                pos_tags[_kb]
                                if _kb < len(pos_tags) else ""
                            )
                            if _kb_low in (
                                "fait", "faire", "fais", "font",
                                "faisant", "laisse", "laisser",
                            ):
                                _causative_back = True
                                break
                            if _kb_pos not in (
                                "VER", "AUX", "CON", "ADV",
                            ):
                                break
                    if (
                        i + 1 < n
                        and result[i + 1].lower() == "par"
                        and curr_low not in _VERBES_PAR
                        and not _par_exemple
                        and not _causative_back
                        and prev_low not in (
                            "à", "de", "d", "pour", "sans",
                            "fait", "faire", "fais", "font",
                            "laisse", "laisser", "y",
                        )
                        and prev_pos != "PRE"
                    ):
                        _r5_fire = True

                if _r5_fire:
                    _pp_cands = [
                        curr_low[:-2] + "\xe9",
                        curr_low[:-2] + "\xe9e",
                        curr_low[:-2] + "\xe9s",
                        curr_low[:-2] + "\xe9es",
                    ]
                    # Determiner genre/nombre cible via contexte
                    _tgt_g, _tgt_n = "", ""
                    # Regle 7/9 coord: matcher le PP coordonne
                    if prev_low in ("et", "puis", "ni") and i > 1:
                        _p2_low_r7 = result[i - 2].lower()
                        if _p2_low_r7.endswith("\xe9es"):
                            _tgt_g, _tgt_n = "f", "p"
                        elif _p2_low_r7.endswith("\xe9e"):
                            _tgt_g, _tgt_n = "f", "s"
                        elif _p2_low_r7.endswith("\xe9s"):
                            _tgt_g, _tgt_n = "m", "p"
                    # Regle 5: NOM/ADJ → genre/nombre du lexique
                    if not _tgt_g and prev_pos in (
                        "NOM", "ADJ", "NOM PROPRE", "?", "",
                    ):
                        _prev_infos_r5 = lexique.info(prev_low)
                        if _prev_infos_r5:
                            _pb = max(
                                _prev_infos_r5,
                                key=lambda e: float(e.get("freq") or 0),
                            )
                            _tgt_g = _pb.get("genre", "")
                            _tgt_n = _pb.get("nombre", "")
                    # Regle 6: DET → genre/nombre du determinant
                    if not _tgt_g and (
                        prev_pos.startswith(("ART", "DET"))
                        or prev_low in (
                            "le", "la", "l", "un", "une", "les",
                            "des", "du", "ce", "cette", "ces",
                        )
                    ):
                        _det_infos_r6 = lexique.info(prev_low)
                        if _det_infos_r6:
                            _db = max(
                                _det_infos_r6,
                                key=lambda e: float(e.get("freq") or 0),
                            )
                            _tgt_g = _db.get("genre", "")
                            _tgt_n = _db.get("nombre", "")
                    # Candidat existant avec freq max (calcule d'abord)
                    _freq_best_pp = None
                    _freq_best_val = -1.0
                    for candidate in _pp_cands:
                        _c_infos = lexique.info(candidate)
                        if _c_infos:
                            _c_freq = max(
                                float(e.get("freq") or 0)
                                for e in _c_infos
                            )
                            if _c_freq > _freq_best_val:
                                _freq_best_pp = candidate
                                _freq_best_val = _c_freq
                    # Selectionner le PP accorde si cible connue
                    _best_pp = None
                    if _tgt_g or _tgt_n:
                        if _tgt_g == "f" and _tgt_n == "p":
                            _tgt_suf = "\xe9es"
                        elif _tgt_g == "f":
                            _tgt_suf = "\xe9e"
                        elif _tgt_n == "p":
                            _tgt_suf = "\xe9s"
                        else:
                            _tgt_suf = "\xe9"
                        _tgt_cand = curr_low[:-2] + _tgt_suf
                        _tgt_infos = lexique.info(_tgt_cand)
                        if _tgt_infos:
                            _tgt_freq = max(
                                float(e.get("freq") or 0)
                                for e in _tgt_infos
                            )
                            # Guard: si le candidat genre a une freq tres
                            # basse vs le meilleur candidat par freq, preferer
                            # le freq (ex: "musé" 0.05 vs "musée" 44.24)
                            if (
                                _tgt_freq >= 1.0
                                or _freq_best_val < 10.0
                            ):
                                _best_pp = _tgt_cand
                    # Fallback: candidat avec freq max
                    if _best_pp is None:
                        _best_pp = _freq_best_pp
                    if _best_pp is not None:
                        ancien = result[i]
                        result[i] = transferer_casse(curr, _best_pp)
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=result[i],
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="participe.infinitif_vers_pp",
                            explication="Infinitif -> PP (apposition/nom)",
                        ))
                    if result[i] != curr:
                        continue

    return result, corrections


def verifier_pp_accord_etre(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Accorde le participe passe avec le sujet quand l'auxiliaire est etre.

    Pattern : sujet + etre_conjugue + PP (VER qui n'est pas un infinitif)
    Ex: "elle est allé" -> "elle est allée"
    """
    if not mots:
        return mots, []

    result = list(mots)
    corrections: list[Correction] = []
    n = len(result)

    _TRANSPARENTS = frozenset({
        "ne", "n'", "pas", "plus", "jamais", "rien",
        "point", "y", "en",
        # Pronoms reflexifs (synchro avec verifier_participes_passes)
        "se", "s'", "me", "m'", "te", "t'",
        "bien", "déjà", "donc", "alors", "toujours",
        "aussi", "souvent", "encore", "trop", "tout",
        "même", "ensuite", "récemment", "largement",
        "longtemps", "vraiment", "seulement", "beaucoup",
        # Adverbes supplementaires (synchro avec verifier_participes_passes)
        "très", "fortement", "également", "presque",
        "autrefois", "officiellement", "entièrement",
        "généralement",
    })

    # Mots dont la terminaison ressemble a un PP mais qui n'en sont pas
    _NOT_PP = frozenset({
        "plus", "nous", "vous", "tous", "dessus", "dessous",
        "refus", "abus", "jus", "pus", "sous",
        "ensuite", "suite", "fuite", "uite",
    })

    for i in range(1, n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]
        curr_low = curr.lower()

        # Accepter VER ou mots ressemblant a un PP (heuristique morpho)
        _is_pp_like = pos == "VER" or curr_low.endswith(_PP_SUFFIXES)
        if not _is_pp_like:
            continue

        # Exclure les mots qui ne sont jamais des PP
        if curr_low in _NOT_PP:
            continue

        # Exclure les adverbes (endwith _PP_SUFFIXES mais pas des PP)
        if pos == "ADV":
            continue

        # Exclure les infinitifs (-er, -ir, -re) : ce n'est pas un PP
        if curr_low.endswith(("er", "ir", "re")):
            continue

        # Chercher une forme d'etre avant le mot (en sautant ne/pas/y/en)
        etre_found = False
        etre_idx = -1
        _subject_anchor = -1
        for j in range(i - 1, max(-1, i - 5), -1):
            w = result[j].lower()
            if w in ETRE_FORMES:
                etre_found = True
                etre_idx = j
                _subject_anchor = j
                break
            # Compose "a été" : "été" n'est pas dans ETRE_FORMES
            # mais forme le passif compose avec avoir.
            # Guard: seulement pour PP en forme de base (-é, -i, -u)
            # — ne pas corriger un PP deja accorde (-ée, -és, -ées)
            # car la detection de sujet est moins fiable a cette distance.
            if w == "\xe9t\xe9" and not curr_low.endswith(
                ("\xe9e", "\xe9s", "\xe9es", "ie", "ies", "ue", "ues"),
            ):
                # Chercher avoir avant "été"
                for _k_ete in range(j - 1, max(-1, j - 3), -1):
                    _wk = result[_k_ete].lower()
                    if _wk in AUXILIAIRES:
                        etre_found = True
                        etre_idx = j
                        _subject_anchor = _k_ete
                        break
                    if _wk not in _TRANSPARENTS:
                        break
                if etre_found:
                    break
            if w not in _TRANSPARENTS:
                break
        if not etre_found:
            continue

        # Guard: "en" (PRE) directement avant le mot → complement, pas PP
        # "est en crépi" = "is in plaster", not "is plastered"
        if i > 0 and result[i - 1].lower() == "en":
            continue

        # Guard: causatif "fait + infinitif" — PP invariable
        # "elles se sont fait entrer" → "fait" reste invariable
        if curr_low in ("fait", "faite", "faits", "faites") and i + 1 < n:
            _next_caus = result[i + 1].lower()
            if _next_caus.endswith(("er", "ir", "re", "oir")) and lexique is not None:
                if lexique.existe(_next_caus):
                    continue

        # Trouver le genre/nombre du sujet
        # Inclure ADJ car des noms substantives (commune, adversaire)
        # sont souvent tagues ADJ par le lexique
        gn = trouver_sujet_genre_nombre(
            result, pos_tags, morpho, _subject_anchor, lexique,
            pos_nominaux=("NOM", "NOM PROPRE", "ADJ", "ADJ:pos"),
        )

        # Guard: override nombre avec la forme d'etre (plus fiable)
        # "sont" = pluriel, "est/fut/sera" = singulier
        _etre_word = result[etre_idx].lower()
        _ETRE_PLUR = frozenset({
            "sont", "seront", "soient", "furent",
            "seraient", "serions", "seriez", "serez",
            "étaient", "fûmes", "fûtes",
        })
        _ETRE_SING = frozenset({
            "est", "suis", "es", "fut", "sera", "serai", "seras",
            "soit", "serait", "serais",
            "était", "étais",
        })

        if gn is None:
            # Fallback : si sujet inconnu mais etre pluriel, accord en nombre
            # ("sont inhumé" → "sont inhumés" sans connaitre le genre)
            # Guards : seulement si (a) PP est forme de base (pas deja genré)
            # et (b) etre original etait deja pluriel (pas un "est" corrigé en
            # "sont" par erreur en amont).
            _pp_base = (
                (curr_low.endswith("\xe9") and not curr_low.endswith("\xe9e"))
                or (curr_low.endswith("i") and not curr_low.endswith(("ie", "is")))
                or (curr_low.endswith("u") and not curr_low.endswith(("ue", "us")))
            )
            _orig_etre = (
                originaux[etre_idx].lower()
                if originaux and etre_idx < len(originaux)
                else _etre_word
            )
            if _etre_word in _ETRE_PLUR and _pp_base and _orig_etre in _ETRE_PLUR:
                gn = ("Masc", "Plur")
            else:
                continue
        genre, nombre = gn

        if _etre_word in _ETRE_PLUR:
            nombre = "Plur"
        elif _etre_word in _ETRE_SING:
            nombre = "Sing"

        # Determiner si le sujet est un pronom (detection fiable)
        # ou un NOM precede d'un article genrant (detection assez fiable)
        _subject_is_pronoun = False
        _subject_has_article = False
        _ARTICLES = frozenset({
            "la", "le", "l", "les", "une", "un", "l'",
            "sa", "son", "ses", "ma", "mon", "mes",
            "cette", "ce", "ces",
        })
        for _k_subj in range(_subject_anchor - 1, max(-1, _subject_anchor - 8), -1):
            _w_subj = result[_k_subj].lower()
            if _w_subj in _TRANSPARENTS:
                continue
            if _w_subj in (
                "il", "elle", "ils", "elles", "on",
                "je", "j'", "tu", "nous", "vous",
                "ce", "c'",
            ):
                _subject_is_pronoun = True
                break
            # Article/DET → sujet precede d'un article genrant
            if _w_subj in _ARTICLES:
                _art_in_pp = False
                if _k_subj > 0:
                    _pre_w = result[_k_subj - 1].lower()
                    if _pre_w in (
                        "de", "d", "du", "des", "à", "au",
                        "aux", "pour", "dans", "en", "par",
                        "sur", "sous", "vers", "entre",
                        "avec", "sans", "contre",
                    ):
                        _art_in_pp = True
                if not _art_in_pp:
                    _subject_has_article = True
                break
            # NOM/ADJ in the subject group → continue scanning
            # to find pronoun or article before the group
            _subj_pos = pos_tags[_k_subj] if _k_subj < len(pos_tags) else ""
            if _subj_pos in ("NOM", "NOM PROPRE", "ADJ", "ADJ:pos"):
                continue  # keep scanning backward through the NOM group
            break  # other POS → stop scanning

        # Generer la forme accordee
        _gender_reliable = _subject_is_pronoun or _subject_has_article
        candidats = generer_candidats_pp_accorde(curr_low, genre, nombre)
        for candidate in candidats:
            if lexique is None or lexique.existe(candidate):
                if candidate != curr_low:
                    # Guard: si la correction change le genre (pas juste
                    # le nombre), exiger une detection fiable du sujet
                    # (pronom ou ART+NOM).
                    _only_number = (
                        candidate == curr_low + "s"
                        or curr_low == candidate + "s"
                    )
                    if not _only_number and not _gender_reliable:
                        break
                    ancien = result[i]
                    result[i] = transferer_casse(curr, candidate)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="participe.accord_etre",
                        explication=f"Accord PP avec sujet ({genre} {nombre}) apres etre",
                    ))
                break

    return result, corrections
