"""Types publics du module lectura-correcteur."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TypeCorrection(Enum):
    """Type de correction appliquee a un mot."""

    AUCUNE = "aucune"
    HORS_LEXIQUE = "hors_lexique"
    GRAMMAIRE = "grammaire"
    SYNTAXE = "syntaxe"
    RESEGMENTATION = "resegmentation"


@dataclass
class MotAnalyse:
    """Details d'analyse pour un mot."""

    original: str
    corrige: str
    pos: str = ""
    morpho: dict[str, str] = field(default_factory=dict)
    dans_lexique: bool = False
    type_correction: TypeCorrection = TypeCorrection.AUCUNE


@dataclass
class Correction:
    """Une correction individuelle avec explication."""

    index: int
    original: str
    corrige: str
    type_correction: TypeCorrection
    explication: str = ""


@dataclass
class ResultatCorrection:
    """Resultat complet de la correction."""

    phrase_originale: str
    phrase_corrigee: str
    mots: list[MotAnalyse] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)

    @property
    def n_corrections(self) -> int:
        return sum(
            1 for m in self.mots
            if m.original.lower() != m.corrige.lower()
        )
