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

__version__ = "4.0.0"

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
    from lectura_tokeniseur import tokenise

    if engine is None:
        engine = creer_engine()

    tokens = tokenise(texte)
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
