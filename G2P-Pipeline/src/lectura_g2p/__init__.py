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
    result = analyser_phrase_complete(tokens, engine=engine)

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
        # Ponctuation → caractere de pause
        if len(grp.mots) == 1 and getattr(grp.mots[0], "est_ponctuation", False):
            p = _PUNCT_MAP.get(grp.mots[0].text.strip())
            if p:
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
