"""Lectura Formules — Lecture algorithmique des formules pour le français.

Module autonome, zéro dépendance externe. Fournit :
  - Tokeniseur : identification du type + display_fr via enrichir_formules()
  - G2P : transcription phonétique IPA via lire_formule()
  - Syllabeur/Aligneur : events décomposés avec groupement par composant
  - Tables externalisées (CSV + WAV) via TablesStore
  - Chiffres romains bidirectionnels via int_to_roman / roman_to_int

Licence : CC-BY-SA-4.0
"""

from lectura_formules.lecture_formules import (
    EventFormuleLecture,
    LectureFormuleResult,
    OptionsLecture,
    lire_formule,
    lire_nombre,
    lire_sigle,
    lire_date,
    lire_telephone,
    lire_ordinal,
    lire_fraction,
    lire_scientifique,
    lire_maths,
    lire_numero,
    lire_heure,
    lire_monnaie,
    lire_pourcentage,
    lire_intervalle,
    lire_gps,
    lire_page_chapitre,
    enrichir_formules,
)
from lectura_formules.tables import TablesStore, UniteDef, get_store, get_sound_path, get_sounds_dir, set_sounds_dir
from lectura_formules.romains import int_to_roman, roman_to_int

__version__ = "2.0.0"
