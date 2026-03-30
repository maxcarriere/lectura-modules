"""Lectura Syllabeur — Analyseur syllabique du français avec groupes de lecture.

Module autonome, zéro dépendance Python.
Phonémiseur pluggable avec backend eSpeak-NG par défaut.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md
"""

from lectura_syllabeur.lectura_syllabeur import (
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

__version__ = "2.0.0"
