"""Lectura Formules — Lecture algorithmique des formules pour le francais.

Module autonome. Fournit :
  - Tokeniseur : identification du type + display_fr via enrichir_formules()
  - G2P : transcription phonetique IPA via lire_formule()
  - Aligneur : events decomposes avec groupement par composant
  - Tables externalisees (CSV + WAV) via TablesStore
  - Chiffres romains bidirectionnels via int_to_roman / roman_to_int

Mode bi-modal :
  - Local : si les donnees JSON sont presentes (installation complete)
  - API :   sinon, delegue au serveur Lectura via HTTP

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md
"""

from lectura_formules._chargeur import donnees_disponibles

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

__version__ = "3.0.1"
