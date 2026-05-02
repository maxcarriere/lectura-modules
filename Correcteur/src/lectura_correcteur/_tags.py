"""Tags du BiLSTM edit tagger : definitions, application et detection.

Definit les 51 tags de l'edit tagger (alignes sur le checkpoint),
ainsi que ``appliquer_tag`` et ``detecter_tag`` pour transformer un mot
selon un tag ou identifier le tag correspondant a une paire (original,
corrige).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lectura_lexique import Lexique

# =====================================================================
# Tag constants  (valeurs = strings du checkpoint tag_list)
# =====================================================================

PAD = "<PAD>"
KEEP = "KEEP"

PLUR = "PLUR"
SING = "SING"
FEM = "FEM"
MASC = "MASC"

CONJ_1S = "CONJ_1S"
CONJ_2S = "CONJ_2S"
CONJ_3S = "CONJ_3S"
CONJ_1P = "CONJ_1P"
CONJ_2P = "CONJ_2P"
CONJ_3P = "CONJ_3P"
CONJ_INF = "CONJ_INF"

PP_E = "PP_E"
PP_ER = "PP_ER"
PP_EE = "PP_EE"
PP_ES = "PP_ES"
PP_EES = "PP_EES"

HOMO_A = "HOMO_a"
HOMO_A_ACC = "HOMO_\u00e0"
HOMO_EST = "HOMO_est"
HOMO_ET = "HOMO_et"
HOMO_SON = "HOMO_son"
HOMO_SONT = "HOMO_sont"
HOMO_ON = "HOMO_on"
HOMO_ONT = "HOMO_ont"
HOMO_OU = "HOMO_ou"
HOMO_OU_ACC = "HOMO_o\u00f9"
HOMO_LA = "HOMO_la"
HOMO_LA_ACC = "HOMO_l\u00e0"
HOMO_CE = "HOMO_ce"
HOMO_SE = "HOMO_se"
HOMO_CES = "HOMO_ces"
HOMO_SES = "HOMO_ses"
HOMO_LEUR = "HOMO_leur"
HOMO_LEURS = "HOMO_leurs"
HOMO_SA = "HOMO_sa"
HOMO_CA = "HOMO_\u00e7a"
HOMO_PEU = "HOMO_peu"
HOMO_PEUT = "HOMO_peut"
HOMO_PEUX = "HOMO_peux"
HOMO_DANS = "HOMO_dans"
HOMO_DEN = "HOMO_d'en"
HOMO_SANS = "HOMO_sans"
HOMO_SEN = "HOMO_s'en"
HOMO_MAIS = "HOMO_mais"
HOMO_MES = "HOMO_mes"
HOMO_PRES = "HOMO_pr\u00e8s"
HOMO_PRET = "HOMO_pr\u00eat"
HOMO_SI = "HOMO_si"
HOMO_SY = "HOMO_s'y"

# =====================================================================
# Tag list (ordre = indices du checkpoint)
# =====================================================================

TAGS: list[str] = [
    PAD, KEEP,
    PLUR, SING, FEM, MASC,
    CONJ_1S, CONJ_2S, CONJ_3S, CONJ_1P, CONJ_2P, CONJ_3P, CONJ_INF,
    PP_E, PP_ER, PP_EE, PP_ES, PP_EES,
    HOMO_A, HOMO_A_ACC,
    HOMO_EST, HOMO_ET,
    HOMO_SON, HOMO_SONT,
    HOMO_ON, HOMO_ONT,
    HOMO_OU, HOMO_OU_ACC,
    HOMO_LA, HOMO_LA_ACC,
    HOMO_CE, HOMO_SE,
    HOMO_CES, HOMO_SES,
    HOMO_LEUR, HOMO_LEURS,
    HOMO_SA, HOMO_CA,
    HOMO_PEU, HOMO_PEUT, HOMO_PEUX,
    HOMO_DANS, HOMO_DEN,
    HOMO_SANS, HOMO_SEN,
    HOMO_MAIS, HOMO_MES,
    HOMO_PRES, HOMO_PRET,
    HOMO_SI, HOMO_SY,
]

N_TAGS: int = len(TAGS)  # 51

TAG2IDX: dict[str, int] = {tag: i for i, tag in enumerate(TAGS)}

# =====================================================================
# Internal mappings
# =====================================================================

# Homophone tag -> target word
_TAG_TO_HOMO: dict[str, str] = {
    HOMO_A: "a", HOMO_A_ACC: "\u00e0",
    HOMO_EST: "est", HOMO_ET: "et",
    HOMO_SON: "son", HOMO_SONT: "sont",
    HOMO_ON: "on", HOMO_ONT: "ont",
    HOMO_OU: "ou", HOMO_OU_ACC: "o\u00f9",
    HOMO_LA: "la", HOMO_LA_ACC: "l\u00e0",
    HOMO_CE: "ce", HOMO_SE: "se",
    HOMO_CES: "ces", HOMO_SES: "ses",
    HOMO_LEUR: "leur", HOMO_LEURS: "leurs",
    HOMO_SA: "sa", HOMO_CA: "\u00e7a",
    HOMO_PEU: "peu", HOMO_PEUT: "peut", HOMO_PEUX: "peux",
    HOMO_DANS: "dans", HOMO_DEN: "d'en",
    HOMO_SANS: "sans", HOMO_SEN: "s'en",
    HOMO_MAIS: "mais", HOMO_MES: "mes",
    HOMO_PRES: "pr\u00e8s", HOMO_PRET: "pr\u00eat",
    HOMO_SI: "si", HOMO_SY: "s'y",
}

# Target word -> tag (reverse)
_HOMO_TO_TAG: dict[str, str] = {v: k for k, v in _TAG_TO_HOMO.items()}

# Conjugation tag <-> person key used by lexique.conjuguer()
_CONJ_TAG_KEY: dict[str, str] = {
    CONJ_1S: "1s", CONJ_2S: "2s", CONJ_3S: "3s",
    CONJ_1P: "1p", CONJ_2P: "2p", CONJ_3P: "3p",
}
_CONJ_KEY_TAG: dict[str, str] = {v: k for k, v in _CONJ_TAG_KEY.items()}

# PP tag -> suffix
_PP_TAG_SUFFIX: dict[str, str] = {
    PP_E: "\u00e9", PP_ER: "er",
    PP_EE: "\u00e9e", PP_ES: "\u00e9s", PP_EES: "\u00e9es",
}

# PP suffixes for detection (longest first)
_PP_SUFFIXES_DETECT = ("\u00e9es", "\u00e9e", "\u00e9s", "er", "\u00e9")
_PP_TAGS_DETECT = (PP_EES, PP_EE, PP_ES, PP_ER, PP_E)


# =====================================================================
# Helpers
# =====================================================================

def _preserve_case(original: str, target: str) -> str:
    if not original or not target:
        return target
    if original[0].isupper():
        return target[0].upper() + target[1:]
    return target


def _pp_stem(mot: str) -> str:
    low = mot.lower()
    for sfx in _PP_SUFFIXES_DETECT:
        if low.endswith(sfx):
            return mot[:len(mot) - len(sfx)]
    return mot


def _get_verb_entry(entries: list[dict]) -> dict | None:
    for e in entries:
        if e.get("cgram") in ("VER", "AUX"):
            return e
    return None


def _genre_matches(g1: str, g2: str) -> bool:
    """Compare genres qui peuvent etre 'm'/'masculin' ou 'f'/'feminin'."""
    if not g1 or not g2:
        return True
    return g1[0] == g2[0]


# =====================================================================
# appliquer_tag
# =====================================================================

def appliquer_tag(mot: str, tag: str, lexique: Lexique) -> str:
    """Applique un tag a un mot pour produire la forme corrigee."""
    if tag in (KEEP, PAD):
        return mot

    # -- Homophones ------------------------------------------------
    target = _TAG_TO_HOMO.get(tag)
    if target is not None:
        return _preserve_case(mot, target)

    # -- Nombre ----------------------------------------------------
    if tag == PLUR:
        return _appliquer_nombre(mot, "pluriel", lexique)
    if tag == SING:
        return _appliquer_nombre(mot, "singulier", lexique)

    # -- Genre -----------------------------------------------------
    if tag == FEM:
        return _appliquer_genre(mot, "f", lexique)
    if tag == MASC:
        return _appliquer_genre(mot, "m", lexique)

    # -- Conjugaison -----------------------------------------------
    if tag == CONJ_INF:
        entries = lexique.info(mot)
        ve = _get_verb_entry(entries)
        if ve:
            lemme = ve.get("lemme", "")
            if lemme:
                return _preserve_case(mot, lemme)
        return mot

    person_key = _CONJ_TAG_KEY.get(tag)
    if person_key is not None:
        return _appliquer_conj(mot, person_key, lexique)

    # -- Participe passe -------------------------------------------
    suffix = _PP_TAG_SUFFIX.get(tag)
    if suffix is not None:
        return _pp_stem(mot) + suffix

    return mot


def _appliquer_nombre(mot: str, nombre_cible: str, lexique: Lexique) -> str:
    """Produit la forme singulier/pluriel via le lexique, sinon par suffixe."""
    entries = lexique.info(mot)
    # Try each entry (ADJ, NOM, etc.) until we find the target form
    for entry in entries:
        lemme = entry.get("lemme", mot)
        cgram = entry.get("cgram", "")
        if not cgram or cgram == "NOM PROPRE":
            continue
        genre = entry.get("genre", "")
        formes = lexique.formes_de(lemme, cgram)
        for f in formes:
            if (f.get("nombre") == nombre_cible
                    and f["ortho"].lower() != mot.lower()
                    and _genre_matches(genre, f.get("genre", ""))):
                return _preserve_case(mot, f["ortho"])
    # Relax genre constraint
    for entry in entries:
        lemme = entry.get("lemme", mot)
        cgram = entry.get("cgram", "")
        if not cgram or cgram == "NOM PROPRE":
            continue
        formes = lexique.formes_de(lemme, cgram)
        for f in formes:
            if (f.get("nombre") == nombre_cible
                    and f["ortho"].lower() != mot.lower()):
                return _preserve_case(mot, f["ortho"])

    # Fallback suffix rules
    if nombre_cible == "pluriel":
        low = mot.lower()
        if low.endswith(("s", "x", "z")):
            return mot
        if low.endswith(("au", "eu")):
            return mot + "x"
        if low.endswith("al"):
            return mot[:-2] + "aux"
        return mot + "s"

    # singulier
    low = mot.lower()
    if low.endswith("aux") and len(low) > 3:
        return mot[:-3] + "al"
    if low.endswith("s") and len(low) > 1:
        return mot[:-1]
    return mot


def _appliquer_genre(mot: str, genre_cible: str, lexique: Lexique) -> str:
    """Produit la forme feminine/masculine via le lexique."""
    entries = lexique.info(mot)
    for entry in entries:
        lemme = entry.get("lemme", mot)
        cgram = entry.get("cgram", "")
        if not cgram or cgram == "NOM PROPRE":
            continue
        nombre = entry.get("nombre", "")
        formes = lexique.formes_de(lemme, cgram)
        for f in formes:
            fg = f.get("genre", "")
            if (fg and fg[0] == genre_cible
                    and f["ortho"].lower() != mot.lower()
                    and (not nombre or not f.get("nombre")
                         or f.get("nombre") == nombre)):
                return _preserve_case(mot, f["ortho"])

    # Minimal fallback
    if genre_cible == "f" and not mot.endswith("e"):
        return mot + "e"
    if genre_cible == "m" and mot.endswith("e") and len(mot) > 1:
        return mot[:-1]
    return mot


def _appliquer_conj(mot: str, person_key: str, lexique: Lexique) -> str:
    """Re-conjugue a la personne/nombre demandee, meme mode/temps."""
    entries = lexique.info(mot)
    ve = _get_verb_entry(entries)
    if not ve:
        return mot

    lemme = ve.get("lemme", "")
    mode = ve.get("mode", "")
    temps = ve.get("temps", "")
    if not lemme:
        return mot

    conj = lexique.conjuguer(lemme)
    if not conj:
        return mot

    # Same mode/temps
    if mode in conj and temps in conj[mode]:
        forme = conj[mode][temps].get(person_key)
        if forme:
            return _preserve_case(mot, forme)

    # Fallback: indicatif present
    ind = conj.get("indicatif", {})
    pres = ind.get("present", ind.get("pr\u00e9sent", {}))
    if pres:
        forme = pres.get(person_key)
        if forme:
            return _preserve_case(mot, forme)

    return mot


# =====================================================================
# detecter_tag
# =====================================================================

def detecter_tag(
    original: str,
    corrige: str,
    type_erreur: str,
    lexique: Lexique,
) -> str:
    """Identifie le tag correspondant a la transformation original -> corrige."""
    if type_erreur in ("PHON", "ACCENT", "TYPO"):
        return KEEP

    if original.lower() == corrige.lower():
        return KEEP

    if type_erreur == "HOMO":
        return _HOMO_TO_TAG.get(corrige.lower(), KEEP)

    if type_erreur == "ACC":
        return _detecter_accord(original, corrige, lexique)

    if type_erreur == "CONJ":
        return _detecter_conj(corrige, lexique)

    if type_erreur == "PP":
        return _detecter_pp(corrige)

    # Unknown type: try detection cascade
    tag = _HOMO_TO_TAG.get(corrige.lower())
    if tag:
        return tag
    tag = _detecter_pp(corrige)
    if tag != KEEP:
        return tag
    tag = _detecter_accord(original, corrige, lexique)
    if tag != KEEP:
        return tag
    return _detecter_conj(corrige, lexique)


def _detecter_accord(original: str, corrige: str, lexique: Lexique) -> str:
    """Detecte PLUR/SING/FEM/MASC entre deux formes."""
    orig_entries = lexique.info(original)
    corr_entries = lexique.info(corrige)

    if orig_entries and corr_entries:
        # Find matching lemma+cgram pairs
        for e_o in orig_entries:
            lo = e_o.get("lemme", "")
            co = e_o.get("cgram", "")
            if not lo:
                continue
            for e_c in corr_entries:
                if e_c.get("lemme") == lo and e_c.get("cgram") == co:
                    on = e_o.get("nombre", "")
                    cn = e_c.get("nombre", "")
                    if on and cn and on != cn:
                        return PLUR if cn == "pluriel" else SING
                    og = e_o.get("genre", "")
                    cg = e_c.get("genre", "")
                    if og and cg and not _genre_matches(og, cg):
                        return FEM if cg[0] == "f" else MASC

    # Suffix fallback
    if corrige.endswith("s") and not original.endswith("s"):
        return PLUR
    if not corrige.endswith("s") and original.endswith("s"):
        return SING
    if corrige.endswith("aux") and original.endswith("al"):
        return PLUR
    if corrige.endswith("al") and original.endswith("aux"):
        return SING

    return KEEP


def _detecter_conj(corrige: str, lexique: Lexique) -> str:
    """Detecte le tag de conjugaison de la forme corrigee."""
    entries = lexique.info(corrige)
    ve = _get_verb_entry(entries)
    if not ve:
        return KEEP

    mode = ve.get("mode", "")
    if mode == "infinitif":
        return CONJ_INF

    personne = ve.get("personne", "")
    nombre = ve.get("nombre", "")
    if personne and nombre:
        nombre_short = nombre[0]  # "singulier" -> "s", "pluriel" -> "p"
        return _CONJ_KEY_TAG.get(f"{personne}{nombre_short}", KEEP)

    return KEEP


def _detecter_pp(corrige: str) -> str:
    """Detecte le tag PP d'apres le suffixe de la forme corrigee."""
    low = corrige.lower()
    for sfx, tag in zip(_PP_SUFFIXES_DETECT, _PP_TAGS_DETECT):
        if low.endswith(sfx):
            return tag
    return KEEP
