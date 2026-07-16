"""Lectura G2P — Pipeline complet grapheme-phoneme du francais.

Orchestre : Tokeniseur → Formules → Phonemiseur → Groupes de lecture.
Option aligneur : + syllabification phonologique.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md

Installation :
    pip install lectura-g2p              # pipeline complet
    pip install lectura-g2p[aligneur]    # + syllabification

Exemple rapide::

    from lectura_g2p import analyser_phrase
    result = analyser_phrase("Les enfants jouent dans le jardin.")
"""

__version__ = "4.1.0"

# ── Re-exports depuis le phonemiseur (couche 1) ──────────────────────────

from lectura_phonemiseur import creer_engine, tokeniser, phrase_vers_chars
from lectura_phonemiseur import (
    appliquer_liaison,
    appliquer_regles_g2p,
    corriger_g2p,
)

# Groupes de lecture
from lectura_phonemiseur import (
    construire_groupes_lecture,
    GroupeLecture,
    OptionsGroupes,
    ajouter_schwa_final,
)

# Pipeline phrase complete (tokeniseur + formules + phonemiseur)
from lectura_phonemiseur import (
    analyser_phrase_complete,
    ResultatPhraseG2P,
    MotAnalyseG2P,
)

# Aligneur (optionnel)
try:
    from lectura_aligneur import (
        syllabifier_groupes,
        LecturaSyllabeur,
        ResultatGroupe,
        ResultatSyllabation,
        Syllabe,
    )
    _HAS_ALIGNEUR = True
except ImportError:
    _HAS_ALIGNEUR = False


# ── Facade simplifiee ────────────────────────────────────────────────────

def analyser_phrase(
    texte: str,
    *,
    engine: object | None = None,
    options_groupes: OptionsGroupes | None = None,
    syllabifier: bool = False,
    engine_kwargs: dict | None = None,
) -> ResultatPhraseG2P | object:
    """Analyse G2P complete d'une phrase en francais.

    Pipeline : tokenisation → phonemisation + formules → groupes de lecture.

    Parameters
    ----------
    texte : str
        Phrase en francais.
    engine : object | None
        Moteur G2P neural. Si None, en cree un automatiquement.
    options_groupes : OptionsGroupes | None
        Options pour les groupes de lecture (liaisons, elisions, etc.).
    syllabifier : bool
        Si True, ajoute la syllabification (necessite lectura-aligneur).
    engine_kwargs : dict | None
        Kwargs supplementaires passes a ``engine.analyser()``
        (ex: ``sep_apos=True``, ``sep_hyphen=True``).

    Returns
    -------
    ResultatPhraseG2P
        Resultat avec mots analyses et groupes de lecture.
        Si ``syllabifier=True``, retourne un ResultatSyllabation.
    """
    from lectura_tokeniseur import normalise, tokenise

    if engine is None:
        engine = creer_engine()

    tokens = tokenise(normalise(texte))
    result = analyser_phrase_complete(
        tokens, engine=engine, engine_kwargs=engine_kwargs,
    )

    # Construire les groupes de lecture
    groupes = construire_groupes_lecture(result, options_groupes)

    if syllabifier:
        if not _HAS_ALIGNEUR:
            raise RuntimeError(
                "La syllabification necessite lectura-aligneur. "
                "Installez-le : pip install lectura-g2p[aligneur]"
            )
        return LecturaSyllabeur().analyser_complet(groupes=groupes)

    # Attacher les groupes au resultat
    result._groupes = groupes
    return result


# ── Ponctuation TTS ────────────────────────────────────────────────────

_PUNCT_MAP = {
    ",": ",", ";": ",", ":": ",",
    ".": ".", "!": "!", "?": "?",
    "\u2026": "\u2026", "...": "\u2026",
    "\u2014": ",", "\u2013": ",",   # tirets cadratins/demi-cadratins → pause
    "(": ",", ")": ",",             # parentheses → pause
}

_SENTENCE_PUNCT = {".", "?", "!", "\u2026", "..."}


# ── Pipeline texte → IPA (pour TTS) ───────────────────────────────────

def groupes_vers_ipa(groupes: list[GroupeLecture]) -> str:
    """Assemble les groupes de lecture en chaine IPA avec liaisons.

    Les groupes de ponctuation sont convertis en caracteres de pause
    reconnus par les modeles TTS (,  .  ?  !  …).

    Parameters
    ----------
    groupes : list[GroupeLecture]
        Sortie de ``construire_groupes_lecture()``.

    Returns
    -------
    str
        Chaine IPA avec espaces entre groupes et ponctuation inseree.
    """
    parts: list[str] = []
    for gi, grp in enumerate(groupes):
        # Ponctuation → caractere de pause (non cumulatif)
        if len(grp.mots) == 1 and getattr(grp.mots[0], "est_ponctuation", False):
            p = _PUNCT_MAP.get(grp.mots[0].text.strip())
            if p:
                # Eviter d'empiler les pauses : garder la plus forte
                if parts and parts[-1] in (",", ".", "?", "!", "\u2026"):
                    _PUNCT_WEIGHT = {",": 0, ".": 1, "?": 1, "!": 1, "\u2026": 1}
                    if _PUNCT_WEIGHT.get(p, 0) > _PUNCT_WEIGHT.get(parts[-1], 0):
                        parts[-1] = p
                else:
                    parts.append(p)
            continue

        # Espace entre groupes (sauf avant le premier et apres ponctuation)
        if gi > 0 and parts and parts[-1] not in (",", ".", "?", "!", "\u2026"):
            parts.append(" ")

        # Assembler l'IPA du groupe avec insertions de liaison
        grp_phones = [m.phone for m in grp.mots]
        if grp.jonctions:
            grp_parts = [grp_phones[0]]
            for j, jonction in enumerate(grp.jonctions):
                if jonction.startswith("liaison_"):
                    grp_parts.append(jonction[len("liaison_"):])
                elif jonction == "elision":
                    grp_parts.append(" ")
                grp_parts.append(grp_phones[j + 1])
            parts.append("".join(grp_parts))
        else:
            parts.append(grp.phone_groupe)

    return "".join(parts)


def texte_vers_phrases_ipa(
    texte: str,
    *,
    engine: object | None = None,
    options_groupes: OptionsGroupes | None = None,
) -> list[tuple[str, int]]:
    """Convertit du texte francais en phrases IPA avec type de phrase.

    Pipeline unifie : Tokeniseur → Formules → G2P → Groupes de lecture → IPA.
    Gere les liaisons, elisions, enchainements et la ponctuation.

    Parameters
    ----------
    texte : str
        Texte francais (une ou plusieurs phrases).
    engine : object | None
        Moteur G2P neural. Si None, en cree un automatiquement.
    options_groupes : OptionsGroupes | None
        Options pour les groupes de lecture.

    Returns
    -------
    list[tuple[str, int]]
        Liste de (ipa_string, phrase_type) par phrase.
        phrase_type : 0=declaratif, 1=interrogatif, 2=exclamatif, 3=suspensif.

    Example
    -------
    >>> from lectura_g2p import texte_vers_phrases_ipa
    >>> texte_vers_phrases_ipa("Les enfants jouent.")
    [('lez‿ɑ̃fɑ̃ ʒu.', 0)]
    """
    from lectura_tokeniseur import normalise, tokenise

    if engine is None:
        engine = creer_engine()

    all_tokens = tokenise(normalise(texte))

    # Enrichir les formules
    try:
        from lectura_formules import enrichir_formules
        enrichir_formules(all_tokens)
    except ImportError:
        pass

    # Decouper en phrases (a la ponctuation terminale)
    sentences: list[list] = []
    current: list = []
    for token in all_tokens:
        current.append(token)
        if token.type.name == "PONCTUATION" and token.text.strip() in _SENTENCE_PUNCT:
            sentences.append(current)
            current = []
    if current and any(t.type.name in ("MOT", "FORMULE") for t in current):
        sentences.append(current)

    if not sentences:
        return []

    results: list[tuple[str, int]] = []

    for sent_tokens in sentences:
        # Detecter phrase_type depuis la ponctuation terminale
        phrase_type = 0
        for tok in reversed(sent_tokens):
            if tok.type.name == "PONCTUATION":
                p = tok.text.strip()
                if p == "?":
                    phrase_type = 1
                elif p == "!":
                    phrase_type = 2
                elif p in ("\u2026", "..."):
                    phrase_type = 3
                break

        # Pipeline G2P
        result = analyser_phrase_complete(sent_tokens, engine=engine)

        # Groupes de lecture (liaisons, elisions, enchainements)
        groupes = construire_groupes_lecture(result, options_groupes)

        # Assembler l'IPA
        ipa = groupes_vers_ipa(groupes)

        if ipa:
            results.append((ipa, phrase_type))

    return results


# ── Représentation alignée pour corpus ──────────────────────────────────


def aligner_pour_corpus(
    texte: str,
    *,
    engine: object | None = None,
    options_groupes: OptionsGroupes | None = None,
    syllabe_lookup: dict[str, str] | None = None,
    ajouter_schwas: bool = False,
    enchainement: bool = False,
) -> dict | None:
    """Produit la représentation alignée 3 lignes pour une phrase.

    Pipeline : tokenisation → G2P → groupes de lecture → alignement per-word
    avec labels de coupure unifiant syllabes et frontières de mots.

    Chaque mot est aligné individuellement (mode sep). Les types de frontières
    (liaison, élision, enchaînement...) sont encodés dans aligned_coupure.

    Parameters
    ----------
    texte : str
        Phrase en français.
    engine : object | None
        Moteur G2P neural. Si None, en crée un automatiquement.
    options_groupes : OptionsGroupes | None
        Options pour les groupes de lecture.
    syllabe_lookup : dict[str, str] | None
        Lookup syllabes par mot (clé=mot lower, valeur=notation syllabique).
    ajouter_schwas : bool
        Si True, ajoute les schwas finaux optionnels (e muet → ə)
        et re-syllabifie chaque mot.
    enchainement : bool
        Si True, re-syllabifie à travers les frontières de mots
        lorsqu'une consonne finale enchaîne avec une voyelle initiale.

    Returns
    -------
    dict | None
        {aligned_ortho, aligned_phone, aligned_coupure}
        ou None si l'alignement échoue.

    Requires
    --------
    lectura-aligneur (pip install lectura-g2p[aligneur])
    """
    if not _HAS_ALIGNEUR:
        raise RuntimeError(
            "aligner_pour_corpus() necessite lectura-aligneur. "
            "Installez-le : pip install lectura-g2p[aligneur]"
        )

    from lectura_tokeniseur import normalise, tokenise
    from lectura_aligneur import (
        align_for_corpus as _align_word,
        jonction_to_coupure,
        CUT_NONE, CUT_SPC, CUT_SYL, CUT_APO, CUT_TIR,
        CONT_M, CONT_C,
        iter_phonemes as _iter_ph,
    )

    if engine is None:
        engine = creer_engine()

    tokens = tokenise(normalise(texte))
    result_g2p = analyser_phrase_complete(tokens, engine=engine)
    groupes = construire_groupes_lecture(result_g2p, options_groupes)

    all_ortho: list[str] = []
    all_phone: list[str] = []
    all_coupure: list[str] = []
    is_first_word = True

    for grp in groupes:
        # Ponctuation : inclure avec _C phone et PONCT coupure
        if len(grp.mots) == 1 and getattr(grp.mots[0], "est_ponctuation", False):
            ponct_text = grp.mots[0].text
            for ch in ponct_text:
                if not is_first_word:
                    all_ortho.append("<SEP>")
                    all_phone.append("<SEP>")
                    all_coupure.append(CUT_SPC)
                    is_first_word = False
                all_ortho.append(ch)
                all_phone.append(CONT_C)
                all_coupure.append("PONCT")
                is_first_word = False
            continue

        for m_idx, mot in enumerate(grp.mots):
            phone = mot.phone
            word = mot.text.lower()

            if not phone and not getattr(mot, 'est_formule', False):
                continue

            # Déterminer le type de frontière
            if is_first_word:
                boundary = CUT_NONE
            elif m_idx == 0:
                # Premier mot du groupe = frontière espace normale
                boundary = CUT_SPC
            else:
                # Mot lié au précédent par une jonction
                jonction = grp.jonctions[m_idx - 1] if m_idx - 1 < len(grp.jonctions) else ""
                boundary = jonction_to_coupure(jonction)

            # Séparateur entre mots
            if not is_first_word:
                if boundary == CUT_APO:
                    # Élision : apostrophe + <SEP> APO (deux mots distincts)
                    if not (all_ortho and all_ortho[-1] == "'"):
                        all_ortho.append("'")
                        all_phone.append(CONT_M)
                        all_coupure.append(CUT_NONE)
                    all_ortho.append("<SEP>")
                    all_phone.append("<SEP>")
                    all_coupure.append(CUT_APO)
                elif boundary == CUT_TIR:
                    # Tiret : caractère ortho, pas de <SEP>
                    if all_ortho and all_ortho[-1] == "-":
                        all_coupure[-1] = CUT_TIR
                    else:
                        all_ortho.append("-")
                        all_phone.append(CONT_M)
                        all_coupure.append(CUT_TIR)
                else:
                    all_ortho.append("<SEP>")
                    all_phone.append("<SEP>")
                    all_coupure.append(boundary)

            # Formule : placeholder unique <FML>
            if getattr(mot, 'est_formule', False):
                all_ortho.append("<FML>")
                all_phone.append("<FML>")
                all_coupure.append("FML")
                is_first_word = False
                continue

            # Syllabes pour ce mot
            syl = None
            if syllabe_lookup:
                from lectura_aligneur import _aligned
                syl_str = syllabe_lookup.get(word, "")
                if syl_str:
                    try:
                        # Import local pour éviter dépendance V5
                        from lectura_aligneur._utilitaires import iter_phonemes
                        n_ph = len(iter_phonemes(phone))
                        # Parser la notation syllabique simple (syl.la.be → [0,1,1])
                        parts = syl_str.split(".")
                        syl_labels: list[int] = []
                        for part in parts:
                            ph_in_syl = len(iter_phonemes(part))
                            syl_labels.extend([1 if i == 0 and syl_labels else 0
                                               for i in range(ph_in_syl)])
                        if len(syl_labels) == n_ph:
                            syl = syl_labels
                    except Exception:
                        pass

            r = _align_word(
                ortho=word, phone=phone,
                syllabe_labels=syl,
                word_boundary=CUT_NONE,
            )

            if r is None:
                # Fallback
                for ch in word:
                    all_ortho.append(ch)
                    all_phone.append("<PAD>")
                    all_coupure.append(CUT_NONE)
            else:
                all_ortho.extend(r["aligned_ortho"])
                all_phone.extend(r["aligned_phone"])
                all_coupure.extend(r["aligned_coupure"])

            is_first_word = False

    if not all_ortho:
        return None

    # ── Syllabification de base + post-traitement optionnel ──
    from lectura_aligneur.lectura_aligneur import _syllabify_ipa
    from lectura_aligneur._chargeur import voyelles as _voyelles, consonnes as _consonnes
    import unicodedata

    _V = _voyelles() | {"\u0251"}  # + ɑ (base de la nasale ɑ̃)
    _C = _consonnes()

    def _iter_ph(ipa: str) -> list[str]:
        tokens: list[str] = []
        for ch in ipa:
            if unicodedata.category(ch).startswith("M"):
                if tokens:
                    tokens[-1] += ch
            else:
                tokens.append(ch)
        return tokens

    def _est_v(ph: str) -> bool:
        return ph in _V or (ph and ph[0] in _V)

    def _raw_phone(seq: list[str]) -> str:
        return "".join(c for c in seq if c not in (CONT_M, CONT_C, "_D", "<SEP>"))

    def _syl_labels(phone_brut: str) -> list[int]:
        if not phone_brut:
            return []
        sylls = _syllabify_ipa(phone_brut)
        labels: list[int] = []
        for s_idx, syll in enumerate(sylls):
            for p_idx, _ in enumerate(_iter_ph(syll)):
                labels.append(1 if s_idx > 0 and p_idx == 0 else 0)
        return labels

    def _apply_labels(phone, coup, labels, start, end):
        ph_idx = 0
        first = True
        for i in range(start, end + 1):
            tok = phone[i]
            if tok == "<SEP>":
                continue
            elif tok in (CONT_M, CONT_C):
                coup[i] = CUT_NONE
            else:
                if first:
                    first = False
                    ph_idx += 1
                elif ph_idx < len(labels):
                    coup[i] = CUT_SYL if labels[ph_idx] == 1 else CUT_NONE
                    ph_idx += 1
                else:
                    coup[i] = CUT_NONE

    # Extraire les plages de mots
    word_ranges: list[tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(all_ortho):
        if ch == "<SEP>":
            if i > start:
                word_ranges.append((start, i - 1))
            start = i + 1
    if start < len(all_ortho):
        word_ranges.append((start, len(all_ortho) - 1))

    # Syllabification de base (toujours)
    for ws, we in word_ranges:
        phone_brut = _raw_phone(all_phone[ws:we + 1])
        _apply_labels(all_phone, all_coupure, _syl_labels(phone_brut), ws, we)

    # Schwas finaux par mot (optionnel)
    if ajouter_schwas:
        for ws, we in word_ranges:
            phone_brut = _raw_phone(all_phone[ws:we + 1])
            if not phone_brut:
                continue
            ph_tokens = _iter_ph(phone_brut)
            if ph_tokens and _est_v(ph_tokens[-1]):
                continue
            # Chercher le dernier 'e'/_M depuis la fin du mot
            e_pos = None
            for i in range(we, ws - 1, -1):
                if all_ortho[i] == "e":
                    if all_phone[i] in (CONT_M, CONT_C):
                        e_pos = i
                    break
            if e_pos is None:
                continue
            # Vérifier que le 'e' est en position finale :
            # aucun phonème réel ne doit suivre dans le mot.
            has_real_after = False
            for i in range(e_pos + 1, we + 1):
                if all_phone[i] not in (CONT_M, CONT_C, "_D", "<SEP>"):
                    has_real_after = True
                    break
            if has_real_after:
                continue
            all_phone[e_pos] = "\u0259"
            new_brut = _raw_phone(all_phone[ws:we + 1])
            _apply_labels(all_phone, all_coupure, _syl_labels(new_brut), ws, we)

    # Enchaînements (optionnel)
    if enchainement and len(word_ranges) >= 2:
        groups: list[list[int]] = [[0]]
        for w_idx in range(1, len(word_ranges)):
            ps, pe = word_ranges[w_idx - 1]
            cs, ce = word_ranges[w_idx]
            sep_pos = None
            for i in range(pe + 1, cs):
                if all_ortho[i] == "<SEP>":
                    sep_pos = i
                    break
            if sep_pos is None or all_coupure[sep_pos] != CUT_SPC:
                groups.append([w_idx])
                continue
            prev_ph = _raw_phone(all_phone[ps:pe + 1])
            curr_ph = _raw_phone(all_phone[cs:ce + 1])
            if not prev_ph or not curr_ph:
                groups.append([w_idx])
                continue
            pt = _iter_ph(prev_ph)
            ct = _iter_ph(curr_ph)
            if pt and ct and pt[-1] in _C and _est_v(ct[0]):
                groups[-1].append(w_idx)
            else:
                groups.append([w_idx])

        for group in groups:
            if len(group) < 2:
                continue
            grp_phone = "".join(
                _raw_phone(all_phone[word_ranges[wi][0]:word_ranges[wi][1] + 1])
                for wi in group
            )
            labels = _syl_labels(grp_phone)
            ph_idx = 0
            first_real = True
            for wpos, wi in enumerate(group):
                ws, we = word_ranges[wi]
                # Marquer les SPC enchaînés : >SPC<
                if wpos > 0:
                    prev_we = word_ranges[group[wpos - 1]][1]
                    for i in range(prev_we + 1, ws):
                        if all_ortho[i] == "<SEP>":
                            all_coupure[i] = ">SPC<"
                for i in range(ws, we + 1):
                    tok = all_phone[i]
                    if tok == "<SEP>":
                        continue
                    elif tok in (CONT_M, CONT_C):
                        all_coupure[i] = CUT_NONE
                    else:
                        if first_real:
                            first_real = False
                            ph_idx += 1
                        elif ph_idx < len(labels):
                            all_coupure[i] = CUT_SYL if labels[ph_idx] == 1 else CUT_NONE
                            ph_idx += 1
                        else:
                            all_coupure[i] = CUT_NONE

    # Restaurer APO/TIR/PONCT sur les positions spéciales
    _ponct = frozenset(",.;:!?«»\"()…–—")
    for i, ch in enumerate(all_ortho):
        if ch in ("'", "\u2019"):
            all_coupure[i] = CUT_APO
        elif ch == "-":
            all_coupure[i] = CUT_TIR
        elif ch in _ponct:
            all_coupure[i] = "PONCT"

    return {
        "aligned_ortho": all_ortho,
        "aligned_phone": all_phone,
        "aligned_coupure": all_coupure,
    }
