"""Modèles de données pour le tokeniseur."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

Span = tuple[int, int]


class TokenType(Enum):
    """Types de tokens reconnus."""
    MOT = "mot"
    PONCTUATION = "ponctuation"
    SEPARATEUR = "separateur"
    FORMULE = "formule"


class FormuleType(Enum):
    """Sous-types de formules détectées."""
    NOMBRE = "nombre"
    SIGLE = "sigle"
    DATE = "date"
    TELEPHONE = "telephone"
    NUMERO = "numero"
    ORDINAL = "ordinal"
    FRACTION = "fraction"
    SCIENTIFIQUE = "scientifique"
    MATHS = "maths"
    HEURE = "heure"
    MONNAIE = "monnaie"
    POURCENTAGE = "pourcentage"
    INTERVALLE = "intervalle"
    GPS = "gps"
    PAGE_CHAPITRE = "page_chapitre"


@dataclass
class Token:
    """Token de base avec type, texte et position dans le texte source."""
    type: TokenType
    text: str
    span: Span


@dataclass
class Mot(Token):
    """Token de type MOT (séquence de lettres).

    Attributs :
        ortho : forme orthographique en minuscules
        children : sous-tokens pour les mots composés (tiret/apostrophe)
    """
    ortho: str = ""
    children: list[Token] = field(default_factory=list)


@dataclass
class Ponctuation(Token):
    """Token de type PONCTUATION (virgule, point, etc.)."""
    pass


@dataclass
class Separateur(Token):
    """Token de type SEPARATEUR (apostrophe, trait d'union, espace).

    Attributs :
        sep_type : "apostrophe" | "hyphen" | "space"
    """
    sep_type: str | None = None


@dataclass
class Formule(Token):
    """Token de type FORMULE (nombre, sigle, date, téléphone, etc.).

    Attributs :
        formule_type : sous-type de la formule
        children : sous-tokens pour les formules complexes
        valeur : valeur normalisée (ex: "42", "SNCF", "06 12 34 56 78")
    """
    formule_type: FormuleType = FormuleType.NOMBRE
    children: list[Token] = field(default_factory=list)
    valeur: str = ""
    display_fr: str = ""
