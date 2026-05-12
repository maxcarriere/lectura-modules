"""Lectura Tokeniseur — Normalisateur et tokeniseur pour le français.

Module autonome, zéro dépendance externe.
Détecte les formules (nombres, sigles, dates, téléphones, numéros,
ordinaux, fractions, notations scientifiques, expressions mathématiques,
heures, monnaies, pourcentages, intervalles, GPS, pages/chapitres).

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md
"""

from lectura_tokeniseur.models import (
    Span,
    TokenType,
    FormuleType,
    Token,
    Mot,
    Ponctuation,
    Separateur,
    Formule,
)
from lectura_tokeniseur.normalisation import normalise
from lectura_tokeniseur.pipeline import (
    tokenise,
    ResultatTokenisation,
    LecturaTokeniseur,
)
try:
    from lectura_formules._maths import MathToken, tokenize_maths
except ImportError:
    MathToken = None  # type: ignore[assignment,misc]
    tokenize_maths = None  # type: ignore[assignment]

__version__ = "2.3.0"
