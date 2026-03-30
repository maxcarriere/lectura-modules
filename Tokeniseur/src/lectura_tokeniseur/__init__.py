"""Lectura Tokeniseur — Normalisateur et tokeniseur pour le français.

Module autonome, zéro dépendance externe.
Détecte les formules (nombres, sigles, dates, téléphones, numéros,
ordinaux, fractions, notations scientifiques, expressions mathématiques).

Licence : CC-BY-SA-4.0
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
