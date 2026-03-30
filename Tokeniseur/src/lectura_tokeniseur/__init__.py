"""Lectura Tokeniseur — Normalisateur et tokeniseur pour le français.

Module autonome, zéro dépendance externe.
Détecte les formules (nombres, sigles, dates, téléphones, numéros,
ordinaux, fractions, notations scientifiques, expressions mathématiques).

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md
"""

from lectura_tokeniseur.lectura_tokeniseur import (
    # Enums
    TokenType,
    FormuleType,
    # Dataclasses
    Token,
    Mot,
    Ponctuation,
    Separateur,
    Formule,
    ResultatTokenisation,
    # Fonctions
    normalise,
    tokenise,
    # Classe principale
    LecturaTokeniseur,
    # Type alias
    Span,
)

__version__ = "2.0.0"
