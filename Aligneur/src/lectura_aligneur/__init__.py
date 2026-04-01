"""Lectura Aligneur-Syllabeur — Aligneur grapheme-phoneme et syllabeur phonologique du francais.

Pivot central du pipeline Lectura. Realise l'alignement lettre-par-lettre
entre orthographe et phonetique, construit les groupes de lecture (elisions,
liaisons, enchainements), et decompose chaque syllabe en attaque/noyau/coda
avec correspondance grapheme-phoneme.

Module autonome, zero dependance Python.
Phonemiseur pluggable avec backend eSpeak-NG par defaut.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md
"""

from lectura_aligneur.lectura_aligneur import (
    # Fonctions utilitaires phonologiques
    iter_phonemes,
    est_voyelle,
    est_consonne,
    est_semi_voyelle,
    # Dataclasses
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
    # Protocoles
    Phonemizer,
    EspeakPhonemizer,
    # Fonctions E1/E2
    construire_groupes,
    lecture_depuis_g2p,
    syllabifier_groupes,
    _valider_spans_formule,
    # Classe principale
    LecturaSyllabeur,
    # Type alias
    Span,
)

__version__ = "2.2.0"
