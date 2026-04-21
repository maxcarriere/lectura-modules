"""Lectura Aligneur-Syllabeur — Aligneur grapheme-phoneme et syllabeur phonologique du francais.

Pivot central du pipeline Lectura. Realise l'alignement lettre-par-lettre
entre orthographe et phonetique, construit les groupes de lecture (elisions,
liaisons, enchainements), et decompose chaque syllabe en attaque/noyau/coda
avec correspondance grapheme-phoneme.

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
    MotAnalyse,
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

# Detection du mode : local (algo present) ou API
try:
    from lectura_aligneur.lectura_aligneur import (
        # Protocoles
        Phonemizer,
        EspeakPhonemizer,
        # Fonctions E1/E2
        construire_groupes,
        lecture_depuis_g2p,
        syllabifier_groupes,
        _valider_spans_formule,
        # Schwas pedagogiques
        ajouter_schwa_final,
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

    def construire_groupes(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "construire_groupes() n'est pas disponible en mode API. "
            "Utilisez LecturaSyllabeur().construire_groupes() a la place."
        )

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

    def ajouter_schwa_final(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "ajouter_schwa_final() n'est pas disponible en mode API."
        )


__version__ = "3.1.0"
