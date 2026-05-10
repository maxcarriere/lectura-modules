"""Regles de conjugaison : pronom sujet + verbe.

Simplifie par rapport au POC : pas de lookup par phone (pas de IPA).
Correction par suffixe direct.
"""

from __future__ import annotations

from lectura_correcteur._types import Correction, TypeCorrection
from lectura_correcteur._utils import transferer_casse
from lectura_correcteur.grammaire._donnees import (
    ALLER_FORMES,
    AUXILIAIRES,
    ETRE_FORMES,
    IRREGULIERS_FORMES_FAUSSES,
    MODAUX_FORMES,
    PLUR_DET,
    PREPOSITIONS,
    PRONOM_PERSONNE,
    SING_DET,
    SUJETS_3PL,
    generer_candidats_1pl,
    generer_candidats_2pl,
    generer_candidats_3pl,
    generer_candidats_singulier,
)

_TRANSPARENTS_AUX = frozenset({
    "ne", "n'", "pas", "plus", "jamais", "rien", "point", "y", "en",
})

# Mots transparents entre sujet nominal et verbe (se, reflexifs, adverbes)
_TRANSPARENTS_SUJET = frozenset({
    "se", "s'", "ne", "n'", "pas", "plus", "jamais", "rien",
    "y", "en", "me", "m'", "te", "t'", "le", "la", "l'", "les",
    "lui", "nous", "vous", "leur",
})

# Demonstratives that can be tagged ADJ but act as DET boundaries.
# Narrower than SING_DET/PLUR_DET to exclude adjective-like quantifiers
# (différents, nombreux, etc.) which are not real determiners.
_DEM_SING = frozenset({"ce", "cet", "cette"})
_DEM_PLUR = frozenset({"ces"})


def _est_infinitif(mot: str, lexique) -> bool:
    """Detecte si un mot est un infinitif (par suffixe et/ou lexique)."""
    low = mot.lower()
    # Suffixes infinitifs classiques
    if low.endswith(("er", "ir", "re", "oir")):
        # Verifier dans le lexique que c'est bien un infinitif
        if lexique is not None and hasattr(lexique, "info"):
            infos = lexique.info(mot)
            if infos:
                return any(
                    e.get("cgram") in ("VER", "AUX")
                    and e.get("mode") in ("infinitif", "Inf", "inf")
                    for e in infos
                )
            # Mot inconnu terminant en -er/-ir/-re → probablement infinitif
            return True
        return True
    return False


def _conjuguer_via_lexique(
    lemme: str, personne: str, nombre: str, lexique,
) -> str | None:
    """Conjugue un verbe au present indicatif via le lexique."""
    if lexique is None or not hasattr(lexique, "formes_de"):
        return None
    nb_key = "singulier" if nombre != "p" else "pluriel"
    nb_key_s = "s" if nombre != "p" else "p"
    for f in lexique.formes_de(lemme):
        if (
            str(f.get("personne")) == personne
            and f.get("nombre") in (nb_key, nb_key_s)
            and f.get("temps") == "present"
            and f.get("mode") == "indicatif"
        ):
            return f.get("ortho", "")
    return None


def _nombre_sujet_nominal(
    mots: list[str],
    pos_tags: list[str],
    origs: list[str],
    idx_verbe: int,
) -> str | None:
    """Detecte le nombre du sujet nominal avant le verbe.

    Retourne 'sing', 'plur', ou None si pas de sujet nominal detecte.

    Scanne en arriere en sautant les mots transparents (se, ne, ...),
    puis les groupes prepositionnels (PRE + DET + NOM/ADJ) pour trouver
    le vrai sujet et non un complement de nom.

    Ex: "le chat de mes voisins dort" → sujet = "le chat" (sing), pas "voisins".
    """
    j = idx_verbe - 1
    # Sauter les mots transparents
    while j >= 0 and mots[j].lower() in _TRANSPARENTS_SUJET:
        j -= 1
    if j < 0:
        return None

    # Memoriser le premier NOM rencontre (pour le fallback)
    _first_nom_j = -1
    # Track if we've crossed a PP boundary (du/au/des/PRE)
    _crossed_pp = False

    # Scan arriere : sauter NOM/ADJ, detecter DET, sauter PP
    while j >= 0:
        pos_j = pos_tags[j] if j < len(pos_tags) else ""
        mot_j = mots[j].lower()

        if pos_j in ("NOM", "NOM PROPRE", "ADJ"):
            # Demonstratives tagged ADJ act as DET boundaries
            # (e.g. "Cette" tagged ADJ instead of DET:dem)
            if pos_j == "ADJ" and mot_j in _DEM_SING:
                return "sing"
            if pos_j == "ADJ" and mot_j in _DEM_PLUR:
                return "plur"
            if _first_nom_j < 0:
                _first_nom_j = j
            j -= 1
            continue

        # VER/AUX tagged words ending in plural marks (données, prises)
        # may be NOM in subject position. Treat like NOM for scanning.
        # Only apply before any PP boundary to avoid scanning through
        # complements ("les plaques du marbre blanc bia a" → "plaques"
        # is beyond PP "du", should not be treated as NOM here).
        if (
            not _crossed_pp
            and pos_j in ("VER", "AUX")
            and mot_j.endswith(("s", "x", "z"))
            and len(mot_j) > 3
        ):
            if _first_nom_j < 0:
                _first_nom_j = j
            j -= 1
            continue

        # ADV : transparent (les lycées les plus proches est)
        if pos_j == "ADV":
            j -= 1
            continue

        # OOV → traiter comme NOM PROPRE (noms propres souvent
        # minuscules dans le corpus : "mont kamui dominez")
        if pos_j in ("?", "") and len(mot_j) > 2:
            if _first_nom_j < 0:
                _first_nom_j = j
            j -= 1
            continue

        # Contractions prepositionnelles : "du", "au", "aux" = PRE+DET
        # Tout NOM vu apres (plus pres du verbe) est dans un PP
        if mot_j in ("du", "au", "aux"):
            _first_nom_j = -1
            _crossed_pp = True
            j -= 1
            continue

        if pos_j in ("ART:def", "ART:ind", "ART", "DET", "DET:dem", "ADJ:pos", "ADJ:dem"):
            # Verifier si ce DET est dans un complement prepositionnel
            if j > 0:
                prev_pos = pos_tags[j - 1] if j - 1 < len(pos_tags) else ""
                prev_mot = mots[j - 1].lower()
                if prev_pos == "PRE" or prev_mot in PREPOSITIONS or prev_mot == "des":
                    # C'est un PP → sauter DET + PRE et continuer
                    # Tout NOM vu apres (plus pres du verbe) est dans ce PP
                    # "des" before a numeral/DET = "de+les" (PP marker)
                    _first_nom_j = -1
                    _crossed_pp = True
                    j -= 2
                    continue
                # ADJ/quantifier between PRE and DET:
                # "de toutes les figures" → j-1=toutes(ADJ), j-2=de(PRE)
                if prev_pos in ("ADJ", "ADJ:pos") and j > 1:
                    _pp2_pos_sn = pos_tags[j - 2] if j - 2 < len(pos_tags) else ""
                    _pp2_mot_sn = mots[j - 2].lower()
                    if _pp2_pos_sn == "PRE" or _pp2_mot_sn in PREPOSITIONS or _pp2_mot_sn == "des":
                        _first_nom_j = -1
                        _crossed_pp = True
                        j -= 3  # sauter DET + ADJ + PRE
                        continue
                # "des" apres un NOM/ADJ/PRO = contraction "de+les" (PP)
                # Ex: "le directeur des ecoles" → "des" introduit un PP
                # Ex: "les jeux méditerranéens cijm a" → "des" apres ADJ
                # Ex: "celui des habitants" → "des" apres PRO:dem = genitif
                if mot_j == "des" and prev_pos in (
                    "NOM", "ADJ", "NOM PROPRE",
                    "PRO:dem", "PRO:rel", "PRO:ind",
                ):
                    _first_nom_j = -1
                    _crossed_pp = True
                    j -= 1  # sauter "des" (la PRE est incorporee)
                    continue
            # Pas de preposition devant → c'est le DET du sujet
            if mot_j in PLUR_DET:
                # Guard: "un/une des NOM" = singulier (un des professeurs avait)
                if mot_j == "des" and j > 0 and mots[j - 1].lower() in ("un", "une", "l'un", "l'une"):
                    return "sing"
                # Guard: superlatif "les plus/moins ADJ" → pas un DET sujet
                if mot_j == "les" and j + 1 < len(mots):
                    _next_sup = mots[j + 1].lower()
                    if _next_sup in ("plus", "moins"):
                        _first_nom_j = -1
                        j -= 1
                        continue
                return "plur"
            if mot_j in SING_DET:
                # Guard: quantifiers taking plural agreement
                # "la plupart vendent", "une trentaine trouvent"
                _QUANTIFIERS_PLUR = frozenset({
                    "plupart", "majorité", "totalité", "moitié",
                    "trentaine", "vingtaine", "quarantaine",
                    "cinquantaine", "soixantaine", "centaine",
                    "dizaine", "douzaine", "millier",
                })
                if _first_nom_j >= 0:
                    _nom_q = mots[_first_nom_j].lower()
                    if _nom_q in _QUANTIFIERS_PLUR:
                        return "plur"
                return "sing"
            return None

        if pos_j == "PRE" or mot_j in PREPOSITIONS:
            # Tout NOM vu apres (plus pres du verbe) est dans un PP
            _first_nom_j = -1
            _crossed_pp = True
            j -= 1
            continue

        break

    # Coordination : si on a casse sur "et" et qu'un NOM/NOM PROPRE/OOV
    # a ete trouve apres "et", avec un autre NOM PROPRE/OOV ou DET avant
    # "et", les sujets coordonnes forment un pluriel.
    # Ex: "delville et ginchy subissent" → plur
    # Guard: ne pas detecter les appositifs ("journaliste et romancier a")
    # ni les frontières de clause ("X est ... et Y est")
    # Note: POS check removed — "et" in French is always CON. POS may
    # show AUX when homophones corrected "est" → "et" without POS update.
    if (
        j >= 0
        and _first_nom_j >= 0
        and mots[j].lower() == "et"
    ):
        _nom_pos_coord = pos_tags[_first_nom_j] if _first_nom_j < len(pos_tags) else ""
        # NOM PROPRE, OOV, or NOM can indicate coordination.
        # NOM is safe here because the before-et check below still
        # requires NOM PROPRE/OOV/DET, preventing appositives
        # ("journaliste et romancier" won't fire: "journaliste" is NOM,
        # not NOM PROPRE/OOV/DET).
        if _nom_pos_coord in ("NOM PROPRE", "NOM") or (
            _nom_pos_coord in ("?", "") and len(mots[_first_nom_j]) > 2
        ):
            for _k_coord in range(j - 1, max(-1, j - 4), -1):
                _pk_coord = pos_tags[_k_coord] if _k_coord < len(pos_tags) else ""
                if _pk_coord in ("NOM PROPRE",):
                    return "plur"
                if _pk_coord in ("?", "") and len(mots[_k_coord]) > 2:
                    return "plur"
                if _pk_coord.startswith(("ART", "DET")):
                    return "plur"
                break

    # Fallback : pas de DET trouve, mais un NOM existe avant le verbe.
    # Si le NOM ne se termine pas par -s/-x/-z, il est morphologiquement
    # singulier. Utile pour les noms propres (carita, singapour) et les
    # sujets sans article.
    # Guard: ignorer les tokens trop courts (1-2 chars) souvent mal tagues
    if _first_nom_j >= 0:
        _nom_low = mots[_first_nom_j].lower()
        if not _nom_low.endswith(("s", "x", "z")) and len(_nom_low) > 2:
            return "sing"

    return None


def _est_sujet_nominal_pluriel(
    mots: list[str],
    pos_tags: list[str],
    origs: list[str],
    idx_verbe: int,
) -> bool:
    """Detecte un sujet nominal pluriel avant le verbe."""
    return _nombre_sujet_nominal(mots, pos_tags, origs, idx_verbe) == "plur"

# Terminaisons attendues par personne (indicatif present, 1er groupe)
_SUFFIXES_ATTENDUS: dict[str, list[str]] = {
    "1": ["e", "s"],      # je mange, je finis
    "2": ["es", "s"],     # tu manges, tu finis
    "3": ["e", "t", "d"],  # il mange, il finit, il prend
    "1p": ["ons"],        # nous mangeons
    "2p": ["ez"],         # vous mangez
    "3p": ["ent"],        # ils mangent
}


def verifier_conjugaisons(
    mots: list[str],
    pos_tags: list[str],
    morpho: dict[str, list[str]],
    lexique,
    originaux: list[str] | None = None,
) -> tuple[list[str], list[Correction]]:
    """Applique les regles de conjugaison sur la phrase.

    Regles :
    3. ils/elles + VER en -e -> -ent
    5. Pronom sujet + VER -> forcer conjugaison correcte (par suffixe)
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

        # Regle 2b : "qui" + P1p/P2p etre/avoir → P3p
        # En relative, P1p/P2p sans pronom correspondant est une erreur
        # "qui sommes executees" → "qui sont executees"
        # "qui avez fui" → "qui ont fui"
        # Guard: "nous qui sommes" / "vous qui etes" = correct
        _P1P2_TO_3PL = {
            "sommes": "sont", "êtes": "sont",
            "avons": "ont", "avez": "ont",
            "es": "sont", "suis": "sont",
            "ai": "ont", "as": "ont",
        }
        _MATCHING_PRONOUN = {
            "sommes": "nous", "avons": "nous",
            "êtes": "vous", "avez": "vous",
            "es": "tu", "as": "tu",
            "suis": "je", "ai": "je",
        }
        curr_low = curr.lower()
        if (
            curr_low in _P1P2_TO_3PL
            and i > 0
            and (pos in ("VER", "AUX", "?", "")
                 or curr_low in ETRE_FORMES
                 or curr_low in AUXILIAIRES)
        ):
            _pp_r2b = result[i - 1].lower()
            if _pp_r2b in ("qui", "ne", "n'", "n\u2019"):
                _match_pron = _MATCHING_PRONOUN.get(curr_low, "")
                _has_match = False
                for _k_r2b in range(i - 2, max(-1, i - 5), -1):
                    _w_r2b = result[_k_r2b].lower()
                    if _w_r2b == _match_pron:
                        _has_match = True
                        break
                    # Stop at clause boundary
                    _pk_r2b = pos_tags[_k_r2b] if _k_r2b < len(pos_tags) else ""
                    if _pk_r2b in ("VER", "AUX", "CON"):
                        break
                if not _has_match:
                    _repl_r2b = _P1P2_TO_3PL[curr_low]
                    ancien = result[i]
                    result[i] = transferer_casse(curr, _repl_r2b)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="conjugaison.relatif",
                        explication=f"'{curr_low}' -> '{_repl_r2b}' (P1p/P2p -> P3p en relative/negation)",
                    ))
                    continue

        # Regle 3a : Formes fausses (allent->vont, etc.) — AVANT le check POS
        # car le tagger peut mal etiqueter ces formes (NOM au lieu de VER)
        # Verifie aussi le mot original (avant correction orthographique)
        if i > 0:
            prev_is_3pl = (
                result[i - 1].lower() in SUJETS_3PL
                or (i - 1 < len(origs) and origs[i - 1].lower() in SUJETS_3PL)
                or _est_sujet_nominal_pluriel(result, pos_tags, origs, i)
            )
            if prev_is_3pl:
                faux_candidate = IRREGULIERS_FORMES_FAUSSES.get(curr.lower())
                # Fallback : verifier le mot original (avant correction ortho)
                if faux_candidate is None and i < len(origs):
                    faux_candidate = IRREGULIERS_FORMES_FAUSSES.get(origs[i].lower())
                if faux_candidate is not None and (lexique is None or lexique.existe(faux_candidate)):
                    ancien = result[i]
                    result[i] = transferer_casse(curr, faux_candidate)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="conjugaison.sujet_pluriel",
                        explication="ils/elles + forme fausse -> 3pl",
                    ))
                    continue

        # Regle 3 : ils/elles + VER -> 3e pluriel
        # Guard: VER directement apres PRE = probable nom propre (à Vienne)
        # Guard: VER directement apres DET/ART = probable NOM (ses œuvres)
        if i > 0 and pos in ("VER", "AUX"):
            _imm_prev_pos = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
            _imm_prev_low = result[i - 1].lower()
            _after_prep = _imm_prev_pos == "PRE" or _imm_prev_low in PREPOSITIONS
            # Exception: quantifiers tagged ART:ind can be pronoun subjects
            # ("plusieurs est" = "plusieurs sont", not DET + NOM)
            _QUANT_PRONOUNS = frozenset({
                "plusieurs", "certains", "certaines",
                "quelques-uns", "quelques-unes",
                "tous", "toutes",
            })
            _after_det_r3 = (
                _imm_prev_pos.startswith(("ART", "DET", "ADJ:pos", "ADJ:dem", "ADJ:ind"))
                and _imm_prev_low not in _QUANT_PRONOUNS
            )
            _curr_prev_low = result[i - 1].lower()
            _subject_is_pronoun_r3 = (
                _curr_prev_low in SUJETS_3PL
                or (
                    i - 1 < len(origs)
                    and origs[i - 1].lower() in SUJETS_3PL
                    # Respecter elles→elle / ils→il (homophones)
                    and _curr_prev_low not in ("il", "elle", "on")
                )
            )
            prev_is_3pl = not _after_prep and not _after_det_r3 and (
                _subject_is_pronoun_r3
                or _est_sujet_nominal_pluriel(result, pos_tags, origs, i)
            )

            # Guard est→sont : si le contexte suggere une coordination
            # (NOM/ADJ + est + NOM/ART/DET/PRE), laisser homophones decider
            # Exception: pronoun subjects (elles/ils) are unambiguous → skip guard
            _skip_coord = False
            if prev_is_3pl and not _subject_is_pronoun_r3 and curr.lower() == "est" and i + 1 < n:
                _next_pos_c = pos_tags[i + 1] if i + 1 < len(pos_tags) else ""
                _next_low_c = result[i + 1].lower()
                if _next_pos_c in (
                    "NOM", "NOM PROPRE", "ART", "ART:def", "ART:ind",
                    "DET", "?", "",
                ) or _next_low_c.endswith(("'", "\u2019")):
                    _skip_coord = True
                # Guard: single-letter fragment (orphan elision "l", "d")
                elif len(_next_low_c) == 1 and _next_low_c.isalpha():
                    _skip_coord = True
                # Guard: VER/AUX not a PP form → conjugated verb = coordination
                # ("tensions et récupère", "alsace et détruisent")
                elif _next_pos_c in ("VER", "AUX") and not _next_low_c.endswith((
                    "\u00e9", "\u00e9s", "\u00e9e", "\u00e9es",
                    "i", "is", "ie", "ies",
                    "u", "us", "ue", "ues",
                    "it", "ite", "ites",
                    "ert", "erte", "ertes", "erts",
                )):
                    _skip_coord = True
                # Guard: est + singular ADJ = copula (nominal subject only)
                # "les indicateurs est complexe" → copula, not *sont
                # But "ils est grand" → keep correction (pronoun = sure 3pl)
                elif (
                    _next_pos_c in ("ADJ", "ADJ:pos")
                    and not _next_low_c.endswith(("s", "x", "z"))
                    and _curr_prev_low not in SUJETS_3PL
                ):
                    _skip_coord = True
                # Guard: est + singular PP (VER ending in -é/-ée/-i/-ie/-u/-ue/-it/-ert)
                # = passive voice ("est située", "est construit"), not coordination
                # Plural PPs (-és/-ées/-is/-ies) suggest 3pl, allow correction.
                elif (
                    _next_pos_c in ("VER", "AUX")
                    and not _next_low_c.endswith(("s", "x", "z"))
                    and _next_low_c.endswith((
                        "\u00e9", "\u00e9e",  # é, ée
                        "i", "ie", "u", "ue",
                        "it", "ite", "ert", "erte",
                    ))
                    and _curr_prev_low not in SUJETS_3PL
                ):
                    _skip_coord = True

            # Guard: causatif "fait/faire + infinitif" — ne pas pluraliser
            _skip_causatif = False
            if prev_is_3pl and curr.lower() in ("fait", "fais") and i + 1 < n:
                _next_caus_c = result[i + 1].lower()
                if _next_caus_c.endswith(("er", "ir", "re", "oir")):
                    if lexique is None or lexique.existe(_next_caus_c):
                        _skip_causatif = True

            # Guard: NOM/ADJ homograph — si la forme singuliere du mot
            # est principalement NOM/ADJ, c'est probablement un nom/adj
            # au pluriel et non un verbe a conjuguer.
            # Ex: "arts graphiques" → "graphique" = NOM/ADJ, pas VER
            _skip_nom_adj_r3 = False
            if (
                prev_is_3pl
                and not _skip_coord
                and not _skip_causatif
                and lexique is not None
                and hasattr(lexique, "info")
            ):
                _curr_low_r3 = curr.lower()
                # Verifier la forme singuliere (sans -s)
                if _curr_low_r3.endswith("s") and len(_curr_low_r3) > 3:
                    _sing_r3 = _curr_low_r3[:-1]
                    _sing_infos_r3 = lexique.info(_sing_r3)
                    if _sing_infos_r3:
                        _best_sing_r3 = max(
                            _sing_infos_r3,
                            key=lambda e: float(e.get("freq") or 0),
                        )
                        if (_best_sing_r3.get("cgram") or "") in ("NOM", "ADJ"):
                            # Guard: require significant NOM/ADJ freq
                            # (devon freq=0.35 is a proper noun, not a real NOM;
                            # graphique freq=4.34 is a real NOM/ADJ)
                            _nom_adj_freq_r3 = float(
                                _best_sing_r3.get("freq") or 0,
                            )
                            if _nom_adj_freq_r3 > 2.0:
                                _skip_nom_adj_r3 = True
                    # Also check the current (plural) form directly
                    # (frites: NOM freq=3.56, VER freq=10.67 — singular "frite"
                    # has VER as best, but "frites" as NOM is significant)
                    if not _skip_nom_adj_r3:
                        _curr_infos_r3 = lexique.info(_curr_low_r3)
                        if _curr_infos_r3 and any(
                            (e.get("cgram") or "") in ("NOM", "ADJ")
                            and float(e.get("freq") or 0) > 2.0
                            for e in _curr_infos_r3
                        ):
                            _skip_nom_adj_r3 = True

            if prev_is_3pl and not _skip_coord and not _skip_causatif and not _skip_nom_adj_r3 and not curr.lower().endswith(("ent", "nt")):
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
                            regle="conjugaison.sujet_pluriel",
                            explication="ils/elles + verbe -> 3pl",
                        ))
                        break
                else:
                    candidate = None
                if candidate is not None and result[i] != curr:
                    continue

        # Regle 5 : Pronom sujet + VER -> correction conjugaison
        # Ne pas appliquer si un auxiliaire precede (laisser la regle PP)
        if i > 0 and pos in ("VER", "AUX"):
            _skip_aux = False
            for _j in range(i - 1, max(-1, i - 4), -1):
                _w = result[_j].lower()
                if _w in AUXILIAIRES or _w in MODAUX_FORMES:
                    _skip_aux = True
                    break
                if _w not in _TRANSPARENTS_AUX:
                    break
            # Guard: mot apres un DET est probablement un NOM, pas un VER
            # "il pilote les avions" → "avions" apres "les" = NOM avion
            _after_det_r5 = False
            if not _skip_aux and i > 0:
                _prev_pos_r5 = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                if _prev_pos_r5.startswith(("ART", "DET")):
                    _after_det_r5 = True
            if _skip_aux or _after_det_r5:
                pass  # Laisser la regle des participes gerer / homographe NOM
            elif (pronom_info := _trouver_pronom_sujet(result, origs, i, pos_tags)) is not None:
                personne, nombre = pronom_info
                # Essayer d'abord par lexique (imparfait/futur)
                temps = _detecter_temps_from_suffixe(curr)
                correction = None
                if temps is not None:
                    correction = _corriger_par_lexique(
                        curr, personne, nombre, temps, lexique,
                    )
                # Fallback: deriver directement quand lemmatisation echoue
                if correction is None and temps is not None:
                    correction = _deriver_forme_nombre(
                        curr, personne, nombre, temps, lexique,
                    )
                # Sinon fallback sur suffixe (present)
                if correction is None:
                    correction = _corriger_par_suffixe(
                        curr, personne, nombre, lexique,
                    )
                if correction and correction.lower() != curr.lower():
                    # Guard: if current form is already valid for
                    # this person+number, don't override tense
                    # (avoids changing passé simple to futur etc.)
                    _already_ok = False
                    if lexique is not None and hasattr(lexique, "info"):
                        _pers_infos = lexique.info(curr.lower())
                        _tgt_nb = (
                            "pluriel" if nombre == "p" else "singulier"
                        )
                        if _pers_infos and any(
                            e.get("cgram") in ("VER", "AUX")
                            and str(e.get("personne")) == personne
                            and e.get("nombre") in (_tgt_nb, "s" if nombre != "p" else "p")
                            for e in _pers_infos
                        ):
                            _already_ok = True
                    if not _already_ok:
                        ancien = result[i]
                        result[i] = transferer_casse(curr, correction)
                        corrections.append(Correction(
                            index=i,
                            original=ancien,
                            corrige=result[i],
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="conjugaison.personne",
                            explication=f"Conjugaison P{personne}",
                        ))
                        continue

        # Regle 5b : Sujet nominal pluriel + imparfait/futur
        # "les gens se promenais" -> "promenaient"
        # Guard: word after DET is likely NOM, not VER
        # ("des avions" = NOM avion, not AUX avoir)
        if i > 0 and pos in ("VER", "AUX"):
            _after_det_5b = False
            if i > 0:
                _prev_pos_5b = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                _prev_low_5b = result[i - 1].lower()
                if (
                    _prev_pos_5b.startswith(("ART", "DET"))
                    or _prev_low_5b in PLUR_DET
                    or _prev_low_5b in SING_DET
                ):
                    _after_det_5b = True
            temps = _detecter_temps_from_suffixe(curr)
            if temps is not None and not _after_det_5b and _est_sujet_nominal_pluriel(
                result, pos_tags, origs, i,
            ):
                correction = _deriver_forme_nombre(
                    curr, "3", "p", temps, lexique,
                )
                if correction is None:
                    correction = _corriger_par_lexique(
                        curr, "3", "p", temps, lexique,
                    )
                if correction and correction.lower() != curr.lower():
                    ancien = result[i]
                    result[i] = transferer_casse(curr, correction)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="conjugaison.sujet_pluriel",
                        explication="Sujet nominal pluriel + imp/fut -> 3pl",
                    ))
                    continue

        # Regle 5b-sing : Sujet nominal singulier + imp/fut mauvaise personne
        # "cet arbre avais" -> "avait", "l'actrice étais" -> "était"
        # Guard: mot apres un DET/ART est probablement un NOM, pas un VER
        # ("les avions" = NOM avion, pas VER avoir)
        if i > 0 and pos in ("VER", "AUX") and result[i] == curr:
            _prev_low_5bs = result[i - 1].lower()
            _after_det_5bs = (
                _prev_low_5bs in PLUR_DET
                or _prev_low_5bs in SING_DET
                or (i - 1 < len(pos_tags)
                    and pos_tags[i - 1].startswith(("ART", "DET")))
            )
            temps_5bs = _detecter_temps_from_suffixe(curr)
            if temps_5bs is not None and not _after_det_5bs and _nombre_sujet_nominal(
                result, pos_tags, origs, i,
            ) == "sing":
                correction_5bs = _deriver_forme_nombre(
                    curr, "3", "s", temps_5bs, lexique,
                )
                if correction_5bs is None:
                    correction_5bs = _corriger_par_lexique(
                        curr, "3", "s", temps_5bs, lexique,
                    )
                if correction_5bs and correction_5bs.lower() != curr.lower():
                    ancien = result[i]
                    result[i] = transferer_casse(curr, correction_5bs)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="conjugaison.sujet_singulier",
                        explication="Sujet nominal singulier + imp/fut -> P3s",
                    ))
                    continue

        # Regle 5c : Sujet nominal singulier + VER en -ent/-ez/-ons -> P3s
        # "le club remportent" -> "remporte", "la ville comprenez" -> "comprend"
        if i > 0 and pos in ("VER", "AUX"):
            curr_low_5c = curr.lower()
            _is_wrong_number = (
                (curr_low_5c.endswith("ent") and len(curr_low_5c) > 3)
                or (curr_low_5c.endswith("ez") and len(curr_low_5c) > 3)
                or (curr_low_5c.endswith("ons") and len(curr_low_5c) > 4)
                or (curr_low_5c.endswith("eux") and len(curr_low_5c) > 3)
            )
            # Guard: mots en -ient (P3s de -enir) ne sont pas P3p
            # revient, devient, obtient → deja singulier
            if curr_low_5c.endswith("ient") and not curr_low_5c.endswith("iennent"):
                _is_wrong_number = False
            # Guard: mots en -oint ne sont pas P3p (rejoint, point)
            if curr_low_5c.endswith("oint"):
                _is_wrong_number = False
            # Guard: homographes NOM/VER (incident, continent, president)
            # Si le mot existe comme NOM dans le lexique, c'est probablement
            # un nom et pas un verbe conjugue au pluriel
            if _is_wrong_number and lexique is not None and hasattr(lexique, "info"):
                _infos_5c = lexique.info(curr)
                if _infos_5c:
                    # Skip si NOM, NOM PROPRE, ou SIGLE
                    if any(
                        (e.get("cgram") or "") in ("NOM", "NOM PROPRE", "SIGLE")
                        for e in _infos_5c
                    ):
                        _is_wrong_number = False
                    # Skip si VER tres basse freq + capitalise (NOM PROPRE non-recense)
                    elif (
                        curr[0].isupper()
                        and all(
                            float(e.get("freq") or 0) < 1.0
                            for e in _infos_5c
                            if e.get("cgram") in ("VER", "AUX")
                        )
                    ):
                        _is_wrong_number = False
            # Guard cascade: si un NOM/ADJ entre DET et VER a ete
            # depluralize par une regle precedente ("soldats"→"soldat"),
            # la detection "sing" est suspecte → skip
            # Seuls les changements de nombre comptent (pas les accents)
            if _is_wrong_number and origs:
                for _kc in range(i - 1, max(-1, i - 6), -1):
                    if _kc < len(origs):
                        _kc_orig = origs[_kc].lower()
                        _kc_curr = result[_kc].lower()
                        # Depluralization: orig ends in s/x, result doesn't
                        _was_deplural = (
                            _kc_orig != _kc_curr
                            and _kc_orig.endswith(("s", "x"))
                            and not _kc_curr.endswith(("s", "x"))
                        )
                        if _was_deplural:
                            _kc_pos = pos_tags[_kc] if _kc < len(pos_tags) else ""
                            if _kc_pos in ("NOM", "NOM PROPRE", "ADJ", "ADJ:pos"):
                                _is_wrong_number = False
                                break
                    _kc_pos = pos_tags[_kc] if _kc < len(pos_tags) else ""
                    if _kc_pos.startswith(("ART", "DET")) or _kc_pos in ("PRE", "CON"):
                        break
            if _is_wrong_number and _nombre_sujet_nominal(
                result, pos_tags, origs, i,
            ) == "sing":
                correction_5c = _corriger_par_suffixe(
                    curr, "3", "s", lexique,
                )
                # Fallback: use lexique lemme → P3s present lookup
                # Guard: skip passé simple 3pl (-èrent, -irent, -urent)
                _is_passe_simple = curr_low_5c.endswith((
                    "èrent", "irent", "urent",
                ))
                if (
                    correction_5c is None
                    and not _is_passe_simple
                    and lexique is not None
                    and hasattr(lexique, "formes_de")
                ):
                    _5c_infos = lexique.info(curr)
                    _5c_lemmes = set(
                        e.get("lemme", "")
                        for e in _5c_infos
                        if e.get("cgram") in ("VER", "AUX")
                    )
                    for _5c_lemme in _5c_lemmes:
                        if not _5c_lemme:
                            continue
                        for _5c_f in lexique.formes_de(_5c_lemme):
                            if (
                                str(_5c_f.get("personne")) == "3"
                                and _5c_f.get("nombre") in ("singulier", "s")
                                and _5c_f.get("temps") == "present"
                                and _5c_f.get("mode") == "indicatif"
                            ):
                                _5c_cand = _5c_f.get("ortho", "")
                                if _5c_cand and _5c_cand.lower() != curr_low_5c:
                                    correction_5c = _5c_cand
                                    break
                        if correction_5c:
                            break
                # Guard: candidat doit etre VER/AUX (pas un PRO/NOM/ADJ)
                # "ment" → "me" serait PRO, pas VER
                if (
                    correction_5c
                    and correction_5c.lower() != curr_low_5c
                    and lexique is not None
                    and hasattr(lexique, "info")
                ):
                    _5c_cand_infos = lexique.info(correction_5c)
                    if _5c_cand_infos and not any(
                        e.get("cgram") in ("VER", "AUX")
                        for e in _5c_cand_infos
                    ):
                        correction_5c = None
                if correction_5c and correction_5c.lower() != curr_low_5c:
                    ancien = result[i]
                    result[i] = transferer_casse(curr, correction_5c)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="conjugaison.sujet_singulier",
                        explication="Sujet nominal singulier + verbe plur -> P3s",
                    ))

        # Regle 5d : Sujet nominal singulier + VER en -es (P2s) -> P3s
        # "le match se déroules" -> "déroule", "se situes" -> "situe"
        # Conservative: candidat -s doit etre VER P3s confirme dans lexique
        if i > 0 and pos in ("VER", "AUX"):
            curr_low_5d = curr.lower()
            if (
                curr_low_5d.endswith("es")
                and not curr_low_5d.endswith(("tes", "des"))
                and len(curr_low_5d) > 3
                and lexique is not None
            ):
                cand_5d = curr_low_5d[:-1]  # remove trailing -s
                # Guard: NOM PROPRE — Castres, Sèvres ne sont pas des VER
                _infos_orig_5d = lexique.info(curr)
                if _infos_orig_5d and any(
                    "PROPRE" in (e.get("cgram") or "")
                    for e in _infos_orig_5d
                ):
                    cand_5d = None  # skip
                # Guard: capitalise + VER basse freq = probable NOM PROPRE
                if (
                    cand_5d is not None
                    and curr[0].isupper()
                    and _infos_orig_5d
                    and all(
                        float(e.get("freq") or 0) < 1.0
                        for e in _infos_orig_5d
                        if e.get("cgram") in ("VER", "AUX")
                    )
                ):
                    cand_5d = None  # skip
                # Guard: candidat doit etre VER P3s dans le lexique
                _cand_is_p3s = False
                _cand_infos_5d = None
                if cand_5d is not None:
                    _cand_infos_5d = lexique.info(cand_5d)
                if _cand_infos_5d and any(
                    e.get("cgram") in ("VER", "AUX")
                    and str(e.get("personne")) == "3"
                    and e.get("nombre") in ("singulier", "s")
                    for e in _cand_infos_5d
                ):
                    _cand_is_p3s = True
                # Guard: si le candidat (forme sans -s) a NOM ou ADJ
                # comme cgram dominante → pas un verbe P3s cible
                # (ex: plastiques→plastique=NOM, étanches→étanche=ADJ)
                if _cand_is_p3s and _cand_infos_5d:
                    _best_cand_5d = max(
                        _cand_infos_5d,
                        key=lambda e: float(e.get("freq") or 0),
                    )
                    if (_best_cand_5d.get("cgram") or "") in ("NOM", "ADJ"):
                        _cand_is_p3s = False
                # Guard: PLUR_DET immediately before → NOM pluriel, pas VER
                _prev_is_plur_det_5d = result[i - 1].lower() in PLUR_DET
                if _cand_is_p3s and not _prev_is_plur_det_5d and _nombre_sujet_nominal(
                    result, pos_tags, origs, i,
                ) == "sing":
                    ancien = result[i]
                    result[i] = transferer_casse(curr, cand_5d)
                    corrections.append(Correction(
                        index=i,
                        original=ancien,
                        corrige=result[i],
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="conjugaison.sujet_singulier",
                        explication="Sujet nominal singulier + P2s -> P3s",
                    ))

        # Regle 5e : "qui" + VER mauvaise personne → P3s
        # "qui regroupes" → "qui regroupe", "qui exercez" → "qui exerce"
        # Guard: ne pas toucher -ent/-nt (P3p peut etre correct avec
        # antecedent pluriel : "les gens qui mangent")
        if i > 0 and pos in ("VER", "AUX") and result[i] == curr:
            _prev_is_qui = result[i - 1].lower() == "qui"
            if not _prev_is_qui and i > 1:
                if result[i - 1].lower() in _TRANSPARENTS_SUJET:
                    _prev_is_qui = result[i - 2].lower() == "qui"
            if _prev_is_qui:
                curr_low_5e = curr.lower()
                # Skip P3p (-ent/-nt) : antecedent peut etre pluriel
                if not (curr_low_5e.endswith(("ent", "nt")) and len(curr_low_5e) > 3):
                    # Skip etre/avoir irreguliers (sommes→somme serait faux)
                    if curr_low_5e not in ("sommes", "êtes", "avons", "avez",
                                           "suis", "serons", "serez"):
                        correction_5e = _corriger_par_suffixe(
                            curr, "3", "s", lexique,
                        )
                        if correction_5e and correction_5e.lower() != curr_low_5e:
                            # Guard: candidat doit etre VER dans le lexique
                            # et NOM/ADJ ne doit pas etre la cgram dominante
                            # (gênes=Gênes → gêne=NOM serait faux)
                            _is_ver_p3s_5e = False
                            if lexique is not None:
                                _infos_5e = lexique.info(correction_5e)
                                _is_ver_p3s_5e = bool(_infos_5e) and any(
                                    e.get("cgram") in ("VER", "AUX")
                                    for e in _infos_5e
                                )
                                if _is_ver_p3s_5e and _infos_5e:
                                    _best_5e = max(
                                        _infos_5e,
                                        key=lambda e: float(
                                            e.get("freq") or 0,
                                        ),
                                    )
                                    if (_best_5e.get("cgram") or "") in (
                                        "NOM", "ADJ",
                                    ):
                                        _is_ver_p3s_5e = False
                            if _is_ver_p3s_5e:
                                ancien = result[i]
                                result[i] = transferer_casse(
                                    curr, correction_5e,
                                )
                                corrections.append(Correction(
                                    index=i,
                                    original=ancien,
                                    corrige=result[i],
                                    type_correction=TypeCorrection.GRAMMAIRE,
                                    regle="conjugaison.relatif",
                                    explication="'qui' + mauvaise personne -> P3s",
                                ))

    # Regle 6 : PRO sujet + infinitif -> conjuguer au present
    # "je revoir ce film" → "je revois ce film"
    # Guards: modal/auxiliaire/aller avant → futur proche / modal OK
    for i in range(n):
        # Skip si deja corrige par une regle precedente
        if result[i] != mots[i] or (
            any(c.index == i for c in corrections)
        ):
            continue
        if not _est_infinitif(result[i], lexique):
            continue
        # Chercher un pronom sujet avant (en sautant les transparents)
        pronom_info = _trouver_pronom_sujet(result, origs, i, pos_tags)
        if pronom_info is None:
            continue
        personne, nombre = pronom_info
        # Guard: modal/auxiliaire/aller precede → ne pas conjuguer
        _skip_modal = False
        for _j6 in range(i - 1, max(-1, i - 4), -1):
            _w6 = result[_j6].lower()
            if _w6 in MODAUX_FORMES or _w6 in AUXILIAIRES or _w6 in ALLER_FORMES:
                _skip_modal = True
                break
            if _w6 not in _TRANSPARENTS_AUX and _w6 not in PRONOM_PERSONNE:
                break
        if _skip_modal:
            continue
        # Guard: preposition avant le pronom → complement, pas sujet
        # "pour revoir" (mais ici le pronom serait absent)
        # Guard: "a" / "de" / preposition immediatement avant
        _skip_prep = False
        for _j6p in range(i - 1, max(-1, i - 3), -1):
            _w6p = result[_j6p].lower()
            if _w6p in PREPOSITIONS:
                _skip_prep = True
                break
            if _w6p in PRONOM_PERSONNE or _w6p in _TRANSPARENTS_AUX:
                continue
            break
        if _skip_prep:
            continue
        # Conjuguer via lexique
        lemme_inf = result[i].lower()
        forme = _conjuguer_via_lexique(lemme_inf, personne, nombre or "s", lexique)
        if forme and forme.lower() != result[i].lower():
            ancien = result[i]
            result[i] = transferer_casse(mots[i], forme)
            corrections.append(Correction(
                index=i,
                original=ancien,
                corrige=result[i],
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="conjugaison.infinitif_apres_pronom",
                explication=f"PRO sujet + infinitif -> conjugaison P{personne}",
            ))

    # Regle 7 : Verification systematique personne/nombre via morpho + lexique
    # Catch-all pour les cas que les regles par suffixe manquent (irreguliers)
    # Ex: "il vais" (P1s→P3s), "tu prend" (P3s→P2s avec -s)
    if lexique is not None and hasattr(lexique, "info") and hasattr(lexique, "formes_de"):
        _pers_morpho = morpho.get("personne", [])
        _nb_morpho = morpho.get("nombre", [])
        _mode_morpho = morpho.get("mode", [])
        _temps_morpho = morpho.get("temps", [])
        for i in range(n):
            # Skip si deja corrige
            if result[i] != mots[i] or any(c.index == i for c in corrections):
                continue
            pos_i = pos_tags[i] if i < len(pos_tags) else ""
            if pos_i not in ("VER", "AUX"):
                continue
            # Skip infinitifs, participes, gerondifs
            _mode_i = _mode_morpho[i] if i < len(_mode_morpho) else "_"
            if _mode_i in ("infinitif", "Inf", "inf", "participe", "Par",
                           "par", "gerondif", "Ger", "ger"):
                continue
            # Guard: mot apres DET/possessif/demonstratif est probablement
            # un NOM homographe (ex: "ses œuvres" tague VER mais = NOM)
            if i > 0:
                _prev_pos_r7 = pos_tags[i - 1] if i - 1 < len(pos_tags) else ""
                if _prev_pos_r7.startswith(("ART", "DET", "ADJ:pos", "ADJ:dem", "ADJ:ind")):
                    continue
            # Guard: NOM PROPRE — ne pas conjuguer un nom propre
            # (Castres, Sèvres ont des entrees NOM PROPRE + VER)
            # (Auvergne n'a que VER "auvergner" mais freq tres basse)
            _curr_r7_infos = lexique.info(result[i])
            if _curr_r7_infos:
                # Skip si le mot a une entree NOM PROPRE
                _has_np_r7 = any(
                    "PROPRE" in (e.get("cgram") or "")
                    for e in _curr_r7_infos
                )
                if _has_np_r7:
                    continue
                # Skip si TOUTES les entrees VER sont tres basses en freq
                # et le mot est capitalise (probablement NOM PROPRE non-recense)
                _ver_entries_r7 = [
                    e for e in _curr_r7_infos
                    if e.get("cgram") in ("VER", "AUX")
                ]
                if (
                    _ver_entries_r7
                    and result[i][0].isupper()
                    and all(float(e.get("freq") or 0) < 1.0 for e in _ver_entries_r7)
                ):
                    continue
            # Guard: PP forms — le lexique dit que c'est un participe passe
            # (émis, mis, pris, vu, dit, fait, etc.)
            if _curr_r7_infos and any(
                e.get("cgram") in ("VER", "AUX")
                and e.get("mode") in ("participe", "par", "Par")
                and e.get("temps") in ("passé", "pas", "past", "passe_simple",
                                       "passé composé")
                for e in _curr_r7_infos
            ):
                continue
            # Detecter le sujet attendu
            _pronom_r7 = _trouver_pronom_sujet(result, origs, i, pos_tags)
            if _pronom_r7 is not None:
                _pers_att, _nb_att = _pronom_r7
                _nb_att = _nb_att or "s"
            else:
                _nb_sujet_r7 = _nombre_sujet_nominal(result, pos_tags, origs, i)
                if _nb_sujet_r7 is None:
                    continue
                _pers_att = "3"
                _nb_att = "s" if _nb_sujet_r7 == "sing" else "p"
            # Trouver le lemme du verbe actuel
            _infos_r7 = lexique.info(result[i])
            if not _infos_r7:
                continue
            _lemmes_r7 = set(
                e.get("lemme", "") for e in _infos_r7
                if e.get("cgram") in ("VER", "AUX") and e.get("lemme")
            )
            if not _lemmes_r7:
                continue
            # Verifier si la forme actuelle est deja correcte
            # (en cherchant une entree qui correspond au sujet attendu)
            _deja_ok = False
            for e in _infos_r7:
                if e.get("cgram") not in ("VER", "AUX"):
                    continue
                _ep = str(e.get("personne", "_"))
                _en = e.get("nombre", "_")
                _en_norm = "s" if _en in ("s", "singulier", "Sing") else (
                    "p" if _en in ("p", "pluriel", "Plur") else "_"
                )
                if _ep == _pers_att and _en_norm == _nb_att:
                    _deja_ok = True
                    break
            if _deja_ok:
                continue
            # La forme ne correspond pas au sujet → chercher la bonne forme
            # Determiner le temps cible (present par defaut, ou detecte)
            _temps_i = _temps_morpho[i] if i < len(_temps_morpho) else "_"
            _temps_cible = "present"
            if _temps_i in ("imparfait", "Imp", "imp"):
                _temps_cible = "imparfait"
            elif _temps_i in ("futur", "Fut", "fut"):
                _temps_cible = "futur"
            _forme_r7 = None
            for _lem_r7 in _lemmes_r7:
                for f in lexique.formes_de(_lem_r7):
                    if (
                        str(f.get("personne")) == _pers_att
                        and f.get("nombre") in (
                            "singulier" if _nb_att == "s" else "pluriel",
                            _nb_att,
                        )
                        and f.get("temps") == _temps_cible
                        and f.get("mode") == "indicatif"
                    ):
                        _cand_r7 = f.get("ortho", "")
                        if _cand_r7 and _cand_r7.lower() != result[i].lower():
                            _forme_r7 = _cand_r7
                            break
                if _forme_r7:
                    break
            if _forme_r7:
                # Guard: la forme candidate doit etre VER/AUX en majorite
                _cand_infos_r7 = lexique.info(_forme_r7)
                if _cand_infos_r7:
                    _best_r7 = max(
                        _cand_infos_r7,
                        key=lambda e: float(e.get("freq") or 0),
                    )
                    if (_best_r7.get("cgram") or "") not in ("VER", "AUX"):
                        continue
                ancien = result[i]
                result[i] = transferer_casse(mots[i], _forme_r7)
                corrections.append(Correction(
                    index=i,
                    original=ancien,
                    corrige=result[i],
                    type_correction=TypeCorrection.GRAMMAIRE,
                    regle="conjugaison.accord_morpho",
                    explication=(
                        f"Accord sujet-verbe P{_pers_att}"
                        f"{'s' if _nb_att == 's' else 'p'}"
                    ),
                ))

    return result, corrections


def _trouver_pronom_sujet(
    mots: list[str], origs: list[str], idx_verbe: int,
    pos_tags: list[str] | None = None,
) -> tuple[str, str] | None:
    """Cherche le pronom sujet le plus proche avant le verbe.

    Guard: si un verbe conjugue se trouve entre le pronom et le mot
    courant, le pronom est probablement sujet de ce verbe intermediaire,
    pas du mot courant. Ex: "il pilote les avions" → "il" est sujet de
    "pilote", pas de "avions".
    """
    for j in range(idx_verbe - 1, max(-1, idx_verbe - 4), -1):
        mot = mots[j].lower()
        orig = origs[j].lower() if j < len(origs) else ""
        # Guard: si on rencontre un VER/AUX entre le pronom et le mot
        # courant, c'est le sujet de ce verbe, pas du notre
        if pos_tags is not None and j < len(pos_tags):
            if pos_tags[j] in ("VER", "AUX") and mot not in _TRANSPARENTS_AUX:
                return None  # verbe intermediaire
        for candidate in (mot, orig):
            if candidate in PRONOM_PERSONNE:
                # Guard PRE: pronom apres PRE non-temporelle = complement
                # Ex: "parmi elles sœur marie" → "elles" est complement
                # Exclut "en", "à", "de" (trop courants en temporel)
                _COMPLEMENT_PREPS = frozenset({
                    "parmi", "avec", "sans", "contre", "entre",
                    "selon", "envers", "malgré", "hormis",
                    "chez", "sous", "vers", "devant", "derrière",
                })
                if j > 0 and pos_tags is not None:
                    _pre_pro_mot = mots[j - 1].lower()
                    if _pre_pro_mot in _COMPLEMENT_PREPS:
                        continue  # complement, pas sujet
                # Guard COD: si un NOM/ADJ precede nous/vous,
                # c'est probablement un COD, pas un sujet
                # Ex: "les orphelins vous maudissent" → "vous" est COD
                if candidate in ("nous", "vous") and j > 0:
                    if pos_tags is not None and j - 1 < len(pos_tags):
                        _prev_j_pos = pos_tags[j - 1]
                        if _prev_j_pos in ("NOM", "ADJ"):
                            continue
                return PRONOM_PERSONNE[candidate]
    return None


def _detecter_temps_from_suffixe(mot: str) -> str | None:
    """Detecte le temps d'une forme verbale par son suffixe.

    Retourne "Imp" (imparfait) ou "Fut" (futur) ou None.
    """
    low = mot.lower()
    # Imparfait : -aient (avant -ais/-ait pour eviter collision)
    if low.endswith("aient"):
        return "Imp"
    if low.endswith(("ais", "ait")) and len(low) > 3:
        return "Imp"
    if low.endswith(("ions", "iez")) and len(low) > 4:
        return "Imp"
    # Futur : -ront, -rons, -rez (avant -ra/-rai/-ras)
    if low.endswith(("ront", "rons", "rez")) and len(low) > 4:
        return "Fut"
    if low.endswith(("rai", "ras")) and len(low) > 4:
        return "Fut"
    if low.endswith("ra") and len(low) > 3:
        return "Fut"
    return None


def _lemmatiser_verbe(mot: str, temps: str) -> str | None:
    """Retrouve l'infinitif a partir d'une forme conjuguee.

    Heuristique par suffixe pour imparfait et futur.
    """
    low = mot.lower()
    if temps == "Imp":
        # Ordre : suffixes longs d'abord
        for suf in ("aient", "ions", "iez", "ais", "ait"):
            if low.endswith(suf) and len(low) > len(suf):
                radical = low[:-len(suf)]
                # 2e groupe : finissait → finiss → finir
                if radical.endswith("iss"):
                    return radical[:-3] + "ir"
                # 3e groupe -oir : avait → av → avoir (NOT avir)
                # Try -oir before -ir for radicals ending in v
                if radical.endswith("v"):
                    return radical + "oir"
                # 3e groupe : dormait → dorm → dormir
                if radical.endswith(("m", "t", "n")) and not radical.endswith("e"):
                    return radical + "ir"
                # 1er groupe : mangeait → mange → manger
                # radical se termine deja par 'e' (mange) → juste ajouter 'r'
                if radical.endswith("e"):
                    return radical + "r"
                return radical + "er"
    if temps == "Fut":
        for suf in ("ront", "rons", "rez", "rai", "ras", "ra"):
            if low.endswith(suf) and len(low) > len(suf) + 1:
                radical = low[:-len(suf)]
                # Le radical du futur = infinitif pour les reguliers
                # manger(a) → radical=manger, finir(a) → radical=finir
                if radical.endswith("er") or radical.endswith("ir"):
                    return radical
                # 2e groupe : finira → radical=fini → finir
                if radical.endswith("i"):
                    return radical + "r"
                # 1er groupe : mangera → radical=mange → manger
                if radical.endswith("e"):
                    return radical + "r"
                return radical + "er"
    return None


def _corriger_par_lexique(
    mot: str, personne: str, nombre: str, temps: str, lexique,
) -> str | None:
    """Corrige par lookup dans lexique.conjuguer().

    Cherche l'infinitif, puis la bonne forme conjuguee.
    Gere deux formats de cles :
    - MockLexique : "1s", "2s", "3s", "1p", "2p", "3p"
    - Lexique reel : "1", "2", "3" (generalement une seule forme par personne)
    """
    if lexique is None:
        return None

    infinitif = _lemmatiser_verbe(mot, temps)
    if infinitif is None:
        return None

    conj = lexique.conjuguer(infinitif)
    if not conj:
        return None

    temps_key = "imparfait" if temps == "Imp" else "futur"
    indicatif = conj.get("indicatif", {})
    table = indicatif.get(temps_key, {})
    if not table:
        return None

    # Essayer les cles du format MockLexique ("1s", "3p", etc.)
    key_sn = personne + ("s" if nombre in ("", "s") else "p")
    forme = table.get(key_sn)
    if forme and forme.lower() != mot.lower():
        if lexique.existe(forme):
            return forme

    # Essayer la cle simple (format reel : "1", "2", "3")
    forme_base = table.get(personne)
    if forme_base:
        # La forme dans la table peut etre sing ou plur.
        # On doit deriver la bonne forme selon le nombre.
        candidat = _deriver_forme_nombre(forme_base, personne, nombre, temps, lexique)
        if candidat and candidat.lower() != mot.lower():
            return candidat

    return None


def _deriver_forme_nombre(
    forme: str, personne: str, nombre: str, temps: str, lexique,
) -> str | None:
    """Derive la forme sing/plur a partir d'une forme de base du lexique.

    Le lexique reel stocke souvent une seule forme par personne.
    Par ex. imparfait P3 = "mangeaient" (3pl). Si on veut 3s, on derive "mangeait".
    """
    low = forme.lower()

    # Determiner si la forme de base est sing ou plur
    is_plur = low.endswith(("ent", "ons", "ez", "ont"))
    want_plur = nombre == "p"

    if is_plur == want_plur:
        # Verifier que la personne correspond aussi (plur->plur)
        _person_mismatch = False
        if is_plur and temps == "Fut":
            if low.endswith("rons") and personne != "1":
                _person_mismatch = True
            elif low.endswith("ront") and personne != "3":
                _person_mismatch = True
            elif low.endswith("rez") and personne != "2":
                _person_mismatch = True
        if is_plur and temps == "Imp":
            if low.endswith("ions") and personne != "1":
                _person_mismatch = True
            elif low.endswith("iez") and personne != "2":
                _person_mismatch = True
            elif low.endswith("aient") and personne != "3":
                _person_mismatch = True
        # Singulier: verifier personne (etais P1/2 vs etait P3)
        if not is_plur and temps == "Imp":
            if low.endswith("ais") and personne == "3":
                _person_mismatch = True
            elif low.endswith("ait") and personne in ("1", "2"):
                _person_mismatch = True
        # Singulier futur: rai=P1, ras=P2, ra=P3
        if not is_plur and temps == "Fut":
            if low.endswith("ras") and personne != "2":
                _person_mismatch = True
            elif low.endswith("rai") and personne != "1":
                _person_mismatch = True
        if not _person_mismatch:
            # La forme correspond deja au nombre et personne voulus
            if lexique is None or lexique.existe(forme):
                return forme

    if temps == "Imp":
        if is_plur and not want_plur:
            # 3pl -> sing : mangeaient -> mangeait (P3s), mangeais (P1/P2s)
            if low.endswith("aient"):
                if personne in ("1", "2"):
                    cand = low[:-4] + "is"
                else:
                    cand = low[:-3] + "t"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ions"):
                if personne == "3":
                    cand = low[:-4] + "ait"
                else:
                    cand = low[:-4] + "ais"
                if lexique is None or lexique.existe(cand):
                    return cand
                # c→ç before a : dénoncions→dénonçait
                radical_ions = low[:-4]
                if radical_ions.endswith("c"):
                    suffix_ions = "ait" if personne == "3" else "ais"
                    cand2 = radical_ions[:-1] + "ç" + suffix_ions
                    if lexique is None or lexique.existe(cand2):
                        return cand2
            if low.endswith("iez"):
                if personne == "3":
                    cand = low[:-3] + "ait"
                else:
                    cand = low[:-3] + "ais"
                if lexique is None or lexique.existe(cand):
                    return cand
                # c→ç before a : dénonciez→dénonçait, prononciez→prononçait
                radical_iez = low[:-3]
                if radical_iez.endswith("c"):
                    suffix_iez = "ait" if personne == "3" else "ais"
                    cand2 = radical_iez[:-1] + "ç" + suffix_iez
                    if lexique is None or lexique.existe(cand2):
                        return cand2
        elif is_plur and want_plur:
            # plur -> plur (changement de personne)
            # iez (P2p) -> aient (P3p), ions (P1p) -> aient (P3p)
            if low.endswith("iez") and personne == "3":
                cand = low[:-3] + "aient"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ions") and personne == "3":
                cand = low[:-4] + "aient"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("aient") and personne == "1":
                cand = low[:-5] + "ions"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("aient") and personne == "2":
                cand = low[:-5] + "iez"
                if lexique is None or lexique.existe(cand):
                    return cand
        elif not is_plur and not want_plur:
            # sing -> sing (changement de personne) : étais (P1/2) -> était (P3)
            if low.endswith("ais") and personne == "3":
                cand = low[:-1] + "t"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ait") and personne in ("1", "2"):
                cand = low[:-1] + "s"
                if lexique is None or lexique.existe(cand):
                    return cand
        elif not is_plur and want_plur:
            # sing -> plur
            if low.endswith("ait"):
                if personne == "1":
                    cand = low[:-3] + "ions"
                elif personne == "2":
                    cand = low[:-3] + "iez"
                else:
                    cand = low[:-1] + "ent"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ais"):
                radical = low[:-3]
                if personne == "1":
                    suffixes = ["ions"]
                elif personne == "2":
                    suffixes = ["iez"]
                else:
                    suffixes = ["aient"]
                for suf in suffixes:
                    cand = radical + suf
                    if lexique is None or lexique.existe(cand):
                        return cand
                    # 1er groupe : mangeais → mang + ions (drop 'e')
                    if radical.endswith("e"):
                        cand2 = radical[:-1] + suf
                        if lexique is None or lexique.existe(cand2):
                            return cand2

    if temps == "Fut":
        if is_plur and not want_plur:
            # ront -> ra (P3s), rai (P1s), ras (P2s)
            if low.endswith("ront"):
                if personne == "1":
                    cand = low[:-4] + "rai"
                elif personne == "2":
                    cand = low[:-4] + "ras"
                else:
                    cand = low[:-4] + "ra"
                if lexique is None or lexique.existe(cand):
                    return cand
        elif is_plur and want_plur:
            # plur -> plur (changement de personne) : rons -> ront, ront -> rons
            if low.endswith("rons") and personne == "3":
                cand = low[:-4] + "ront"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("rons") and personne == "2":
                cand = low[:-4] + "rez"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ront") and personne == "1":
                cand = low[:-4] + "rons"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ront") and personne == "2":
                cand = low[:-4] + "rez"
                if lexique is None or lexique.existe(cand):
                    return cand
        elif not is_plur and want_plur:
            # sing -> plur
            if low.endswith("ra") and not low.endswith("ira"):
                if personne == "1":
                    cand = low[:-1] + "ons"
                elif personne == "2":
                    cand = low[:-1] + "ez"
                else:
                    cand = low[:-2] + "ront"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ra"):
                # For 2e groupe : finira -> finiront/finirons/finirez
                if personne == "1":
                    cand = low[:-1] + "ons"
                elif personne == "2":
                    cand = low[:-1] + "ez"
                else:
                    cand = low[:-2] + "ront"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("rai"):
                if personne == "1":
                    cand = low[:-2] + "ons"
                elif personne == "2":
                    cand = low[:-2] + "ez"
                else:
                    cand = low[:-2] + "ont"
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ras"):
                if personne == "1":
                    cand = low[:-3] + "rons"
                elif personne == "2":
                    cand = low[:-3] + "rez"
                else:
                    cand = low[:-3] + "ront"
                if lexique is None or lexique.existe(cand):
                    return cand
        elif not is_plur and not want_plur:
            # sing -> sing (changement de personne)
            # ras (P2s) -> ra (P3s), rai (P1s) -> ra (P3s)
            if low.endswith("ras") and personne == "3":
                cand = low[:-1]  # deviendras → deviendra
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("rai") and personne == "3":
                cand = low[:-1]  # deviendrai → deviendra
                if lexique is None or lexique.existe(cand):
                    return cand
            if low.endswith("ra") and personne in ("1", "2"):
                if personne == "1":
                    cand = low + "i"  # deviendra → deviendrai
                else:
                    cand = low + "s"  # deviendra → deviendras
                if lexique is None or lexique.existe(cand):
                    return cand

    return None


def _corriger_par_suffixe(
    mot: str, personne: str, nombre: str, lexique,
) -> str | None:
    """Corrige la conjugaison par ajustement de suffixe.

    Pour la 2e personne : "mange" -> "manges"
    Pour la 3e pluriel : "mange" -> "mangent"
    """
    low = mot.lower()
    key = personne + nombre  # ex: "3p", "2", "1"

    # Tu + verbe en -e sans -s final -> ajouter -s
    if key == "2" and low.endswith("e") and not low.endswith("es"):
        candidate = mot + "s"
        if lexique is None or lexique.existe(candidate):
            return candidate

    # Tu + verbe en -ent (3pl) -> singulariser + s
    if key == "2" and low.endswith("ent") and len(low) > 3:
        for candidate in generer_candidats_singulier(mot, "2"):
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Je (P1) : -ent -> singulariser, -es -> retirer s
    if key == "1":
        for candidate in generer_candidats_singulier(mot, "1"):
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Il/elle (P3s) : -ent -> singulariser
    # Guard: -ient est deja P3s (revient, obtient) sauf -iennent (P3p)
    if key in ("3", "3s") and low.endswith("ent") and len(low) > 3:
        if not (low.endswith("ient") and not low.endswith("iennent")):
            if not low.endswith("oint"):
                for candidate in generer_candidats_singulier(mot, "3"):
                    if lexique is None or lexique.existe(candidate):
                        return candidate

    # Il/elle (P3s) : -es (P2s) -> retirer -s
    # "il effectues" -> "il effectue", "il manges" -> "il mange"
    if key in ("3", "3s") and low.endswith("es") and len(low) > 3:
        candidate = mot[:-1]
        if lexique is None or lexique.existe(candidate):
            return candidate

    # Il/elle (P3s) : -ois/-ois (P1/P2) -> -oit
    # "il vois" -> "il voit", "il crois" -> "il croit"
    if key in ("3", "3s") and low.endswith("ois") and len(low) > 3:
        cand = low[:-1] + "t"  # vois → voit
        if lexique is None or lexique.existe(cand):
            return cand

    # Il/elle (P3s) : -is (P1/P2) -> -it
    # "on refroidis" -> "refroidit", "il finis" -> "finit"
    if key in ("3", "3s") and low.endswith("is") and not low.endswith("ois") and len(low) > 3:
        cand = low[:-1] + "t"  # refroidis → refroidit
        if lexique is None or lexique.existe(cand):
            return cand

    # Il/elle (P3s) : -eux (P1/P2) -> -eut
    # "il veux" -> "il veut", "il peux" -> "il peut"
    if key in ("3", "3s") and low.endswith("eux") and len(low) > 3:
        cand = low[:-1] + "t"  # veux → veut
        if lexique is None or lexique.existe(cand):
            return cand

    # Il/elle (P3s) : -onds (P1/P2) -> -ond
    # "il réponds" -> "il répond"
    if key in ("3", "3s") and low.endswith("onds") and len(low) > 4:
        cand = low[:-1]  # réponds → répond
        if lexique is None or lexique.existe(cand):
            return cand

    # Il/elle/on (P3s) ou Je (P1) : -ez (P2p) -> trouver forme P3s/P1s
    # "il travaillez" -> "il travaille"
    if key in ("3", "3s", "1") and low.endswith("ez") and len(low) > 3:
        radical = low[:-2]
        # Stem-changing verbs d'abord (plus specifiques, evite faux amis)
        # -enir: devenez → devient (en → ien+t)
        if radical.endswith("en") and len(radical) > 2:
            cand = radical[:-2] + "ient"
            if lexique is not None and lexique.existe(cand):
                return cand
        # mourir: mourez → meurt (our → eur+t)
        if radical.endswith("our") and len(radical) > 3:
            cand = radical[:-3] + "eurt"
            if lexique is not None and lexique.existe(cand):
                return cand
        # -cevoir: recevez → reçoit (cev → çoi+t)
        if radical.endswith("cev"):
            cand = radical[:-3] + "çoit"
            if lexique is not None and lexique.existe(cand):
                return cand
        # 1er groupe : travaillez → travaille
        # Guard: eviter les formes subjonctif-only
        for cand in (radical + "e", radical + "t", radical + "d"):
            if lexique is None or lexique.existe(cand):
                if lexique is not None:
                    _cinf = lexique.info(cand)
                    _ver_entries = [
                        e for e in _cinf
                        if e.get("cgram") in ("VER", "AUX")
                    ]
                    if _ver_entries and all(
                        e.get("mode") == "subjonctif"
                        for e in _ver_entries
                    ):
                        continue  # Skip subjonctif-only
                return cand
        # 3e groupe : maintenez → maintient (lookup by lemma)
        for cand in (radical + "ient", radical + "it"):
            if lexique is not None and lexique.existe(cand):
                return cand
        # Accent-changing verbs: e→è (achetez→achète)
        _last_e_ez = radical.rfind("e")
        if _last_e_ez >= 0:
            _rad_acc_ez = radical[:_last_e_ez] + "è" + radical[_last_e_ez + 1:]
            for cand in (_rad_acc_ez + "e", _rad_acc_ez + "t"):
                if lexique is not None and lexique.existe(cand):
                    return cand
        # Accent-changing verbs: é→è (possédez→possède, répétez→répète)
        _last_eacute_ez = radical.rfind("\u00e9")
        if _last_eacute_ez >= 0:
            _rad_acc_ez2 = radical[:_last_eacute_ez] + "\u00e8" + radical[_last_eacute_ez + 1:]
            for cand in (_rad_acc_ez2 + "e", _rad_acc_ez2 + "t"):
                if lexique is not None and lexique.existe(cand):
                    return cand
        # Lexique fallback: find lemme → P3s indicatif present
        # Handles irregular verbs: pouvez→peut, devez→doit
        if lexique is not None and hasattr(lexique, "formes_de"):
            _ez_infos = lexique.info(mot)
            _ez_lemmes = set(
                e.get("lemme", "")
                for e in _ez_infos
                if e.get("cgram") in ("VER", "AUX")
            )
            for _ez_lem in _ez_lemmes:
                if not _ez_lem:
                    continue
                for _ez_f in lexique.formes_de(_ez_lem):
                    if (
                        str(_ez_f.get("personne")) == "3"
                        and _ez_f.get("nombre") in ("singulier", "s")
                        and _ez_f.get("temps") == "present"
                        and _ez_f.get("mode") == "indicatif"
                    ):
                        _ez_cand = _ez_f.get("ortho", "")
                        if _ez_cand and _ez_cand.lower() != low:
                            return _ez_cand

    # Il/elle/on (P3s) : -ons (P1p) -> trouver forme P3s
    # "il contenons" -> "il contient"
    if key in ("3", "3s") and low.endswith("ons") and len(low) > 4:
        radical = low[:-3]
        # Stem-changing verbs d'abord
        if radical.endswith("en") and len(radical) > 2:
            cand = radical[:-2] + "ient"
            if lexique is not None and lexique.existe(cand):
                return cand
        if radical.endswith("our") and len(radical) > 3:
            cand = radical[:-3] + "eurt"
            if lexique is not None and lexique.existe(cand):
                return cand
        if radical.endswith("cev"):
            cand = radical[:-3] + "çoit"
            if lexique is not None and lexique.existe(cand):
                return cand
        # Radical seul pour -eons (déménageons→déménage)
        # Guard: le radical doit etre VER (pas NOM PROPRE : achet)
        if lexique is not None and radical.endswith("e"):
            _rad_infos = lexique.info(radical)
            if _rad_infos and any(
                e.get("cgram") in ("VER", "AUX") for e in _rad_infos
            ):
                return radical
        # Generiques — guard: eviter les formes subjonctif-only
        # (poursuivons→poursuive est subjonctif, pas indicatif)
        for cand in (radical + "e", radical + "t", radical + "d",
                     radical + "ient", radical + "it"):
            if lexique is None or lexique.existe(cand):
                if lexique is not None:
                    _cinf = lexique.info(cand)
                    _ver_entries = [
                        e for e in _cinf
                        if e.get("cgram") in ("VER", "AUX")
                    ]
                    if _ver_entries and all(
                        e.get("mode") == "subjonctif"
                        for e in _ver_entries
                    ):
                        continue  # Skip subjonctif-only
                return cand
        # Accent-changing verbs: e→è before silent -e (acheter→achète)
        # Try replacing the last 'e' in radical with 'è'
        _last_e = radical.rfind("e")
        if _last_e >= 0:
            _rad_accent = radical[:_last_e] + "è" + radical[_last_e + 1:]
            for cand in (_rad_accent + "e", _rad_accent + "t"):
                if lexique is not None and lexique.existe(cand):
                    return cand
        # Also é→è (possédons→possède, répétons→répète)
        _last_eacute = radical.rfind("\u00e9")
        if _last_eacute >= 0:
            _rad_accent2 = radical[:_last_eacute] + "\u00e8" + radical[_last_eacute + 1:]
            for cand in (_rad_accent2 + "e", _rad_accent2 + "t"):
                if lexique is not None and lexique.existe(cand):
                    return cand
        # Lexique fallback: find lemme → P3s indicatif present
        # Handles irregular verbs: pouvons→peut, devons→doit
        if lexique is not None and hasattr(lexique, "formes_de"):
            _ons_infos = lexique.info(mot)
            _ons_lemmes = set(
                e.get("lemme", "")
                for e in _ons_infos
                if e.get("cgram") in ("VER", "AUX")
            )
            for _ons_lem in _ons_lemmes:
                if not _ons_lem:
                    continue
                for _ons_f in lexique.formes_de(_ons_lem):
                    if (
                        str(_ons_f.get("personne")) == "3"
                        and _ons_f.get("nombre") in ("singulier", "s")
                        and _ons_f.get("temps") == "present"
                        and _ons_f.get("mode") == "indicatif"
                    ):
                        _ons_cand = _ons_f.get("ortho", "")
                        if _ons_cand and _ons_cand.lower() != low:
                            return _ons_cand

    # Il/elle/on (P3s) : -iens/-iens (P1/P2) -> P3s -ient
    # "il deviens" -> "il devient"
    if key in ("3", "3s") and low.endswith("iens") and len(low) > 4:
        cand = low[:-1] + "t"  # deviens → devient
        if lexique is None or lexique.existe(cand):
            return cand

    # 3e pluriel : generer candidats (1er, 2e, 3e groupe)
    if key == "3p" and not low.endswith(("ent", "nt")):
        candidats = generer_candidats_3pl(mot)
        for candidate in candidats:
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Nous (P1p) : imparfait tronque "dansion" -> "dansions" (manque -s)
    if key == "1p" and low.endswith("ion") and not low.endswith("ions"):
        candidate = mot + "s"
        if lexique is None or lexique.existe(candidate):
            return candidate

    # Nous (P1p) : futur tronque "partiron" -> "partirons" (manque -s)
    if key == "1p" and low.endswith("ron") and not low.endswith("rons"):
        candidate = mot + "s"
        if lexique is None or lexique.existe(candidate):
            return candidate

    # Nous (P1p) : generer candidats 1re pluriel
    if key == "1p" and not low.endswith("ons"):
        for candidate in generer_candidats_1pl(mot):
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Vous (P2p) : imparfait tronque "dansie" -> "dansiez" (manque -z)
    if key == "2p" and low.endswith("ie") and not low.endswith("iez"):
        candidate = mot + "z"
        if lexique is None or lexique.existe(candidate):
            return candidate

    # Vous (P2p) : generer candidats 2e pluriel
    if key == "2p" and not low.endswith("ez"):
        for candidate in generer_candidats_2pl(mot):
            if lexique is None or lexique.existe(candidate):
                return candidate

    # Fallback P3s : strip -s generique (bats→bat, mets→met, prends→prend)
    # Dernier recours apres toutes les regles specifiques
    if key in ("3", "3s") and low.endswith("s") and len(low) > 3:
        cand = low[:-1]
        if lexique is not None:
            _cand_infos = lexique.info(cand)
            if _cand_infos and any(
                e.get("cgram") in ("VER", "AUX") for e in _cand_infos
            ):
                return cand

    return None
