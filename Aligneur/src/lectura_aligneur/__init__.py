"""Lectura Aligneur-Syllabeur — Aligneur grapheme-phoneme et syllabeur phonologique du francais.

Pivot central du pipeline Lectura. Realise l'alignement lettre-par-lettre
entre orthographe et phonetique, et decompose chaque syllabe en
attaque/noyau/coda avec correspondance grapheme-phoneme.

La construction des groupes de lecture (E1 : elisions, liaisons,
enchainements) est desormais dans le module G2P (lectura_phonemiseur.groupes_lecture).

Mode bi-modal :
  - Local : si l'algo et les donnees sont disponibles (installation complete)
  - API :   sinon, delegue au serveur Lectura via HTTP

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md
"""

# Types publics (toujours disponibles)
from lectura_aligneur._types import (
    Span,
    Phoneme,
    GroupePhonologique,
    Syllabe,
    ResultatAnalyse,
    EventFormule,
    LectureFormule,
    OptionsGroupes,
    GroupeLecture,
    ResultatGroupe,
    ResultatSyllabation,
)

# Utilitaires phonologiques (toujours disponibles, zero dependance)
from lectura_aligneur._utilitaires import (
    iter_phonemes,
    est_voyelle,
    est_consonne,
    est_semi_voyelle,
)

# Représentation alignée (toujours disponible, zero dependance)
from lectura_aligneur._aligned import (
    CONT_C,
    CONT_M,
    CONT_D,
    CONT_TOKENS,
    build_aligned_word,
    build_aligned_from_alignment,
    map_syllabes_to_aligned,
    build_coupure_labels,
    # Labels de coupure (tête unifiée syllabe + frontière de mot)
    COUPURE_LABELS,
    CUT_NONE, CUT_SYL, CUT_SPC, CUT_APO, CUT_TIR,
    CUT_LIZ, CUT_LIT, CUT_LIN, CUT_LIR, CUT_LIP,
    jonction_to_coupure,
    bracket_to_coupure,
)

# Detection du mode : local (algo present) ou API
try:
    from lectura_aligneur.lectura_aligneur import (
        # Protocoles
        Phonemizer,
        EspeakPhonemizer,
        # Fonctions E2
        lecture_depuis_g2p,
        syllabifier_groupes,
        _valider_spans_formule,
        # Représentation alignée
        align_for_corpus,
        align_utterance,
        align_phrase,
        # Classe principale (version locale)
        LecturaSyllabeur,
    )
    _MODE = "local"
except ImportError:
    # Algo non disponible → mode API
    from lectura_aligneur._api_client import (
        LecturaSyllabeur,
        LecturaApiError,
    )
    _MODE = "api"

    # Stubs pour les fonctions non disponibles en mode API
    Phonemizer = None  # type: ignore[assignment,misc]
    EspeakPhonemizer = None  # type: ignore[assignment,misc]

    def syllabifier_groupes(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "syllabifier_groupes() n'est pas disponible en mode API. "
            "Utilisez LecturaSyllabeur().syllabifier_groupes() a la place."
        )

    def lecture_depuis_g2p(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "lecture_depuis_g2p() n'est pas disponible en mode API."
        )

    def _valider_spans_formule(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "_valider_spans_formule() n'est pas disponible en mode API."
        )

# Re-export depuis le G2P pour retrocompatibilite
try:
    from lectura_phonemiseur.groupes_lecture import (
        construire_groupes_lecture,
        OptionsGroupes as _G2POptionsGroupes,
        ajouter_schwa_final,
    )
except ImportError:
    def construire_groupes_lecture(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "construire_groupes_lecture() requiert le module lectura_phonemiseur. "
            "Installez-le : pip install lectura-phonemiseur"
        )

    def ajouter_schwa_final(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "ajouter_schwa_final() requiert le module lectura_phonemiseur. "
            "Installez-le : pip install lectura-phonemiseur"
        )


__version__ = "4.0.0"
