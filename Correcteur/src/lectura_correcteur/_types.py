"""Types publics du module lectura-correcteur."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


@runtime_checkable
class TaggerProtocol(Protocol):
    """Protocol duck-typing pour un tagger externe (CRF, BiLSTM unifie, etc.)."""

    def tokenize(self, text: str) -> list[tuple[str, bool]]: ...

    def tag_words(self, words: list[str]) -> list[dict]: ...


@runtime_checkable
class G2PProtocol(Protocol):
    """Protocol duck-typing pour un module G2P (grapheme-to-phoneme)."""

    def prononcer(self, mot: str) -> str | None: ...


class TokeniseurProtocol(Protocol):
    """Protocol duck-typing pour un tokeniseur externe (lectura-tokeniseur, etc.)."""

    def tokeniser(self, text: str) -> list: ...


class TypeCorrection(Enum):
    """Type de correction appliquee a un mot."""

    AUCUNE = "aucune"
    HORS_LEXIQUE = "hors_lexique"
    GRAMMAIRE = "grammaire"
    SYNTAXE = "syntaxe"
    RESEGMENTATION = "resegmentation"


@dataclass
class Candidat:
    """Un candidat de remplacement pour un mot."""

    forme: str
    source: str          # "identite", "ortho_d1", "ortho_d2", "homophone", "morpho", "g2p"
    freq: float = 0.0
    edit_dist: int = 0   # distance d'edition graphemique vs original
    pos: str = ""        # POS du candidat (cgram lexique)
    phone: str = ""
    lemme: str = ""
    genre: str = ""
    nombre: str = ""
    score: float = 0.0   # calcule par le scoring


@dataclass
class MotAnalyse:
    """Details d'analyse pour un mot."""

    original: str
    corrige: str
    pos: str = ""
    morpho: dict[str, str] = field(default_factory=dict)
    dans_lexique: bool = False
    type_correction: TypeCorrection = TypeCorrection.AUCUNE
    confiance: float = 1.0
    confiance_pos: float = 1.0
    pos_blind: str = ""
    divergence_pos: bool = False
    pos_scores: list[tuple[str, float]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    suggestions_scored: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Correction:
    """Une correction individuelle avec explication."""

    index: int
    original: str
    corrige: str
    type_correction: TypeCorrection
    regle: str = ""          # identifiant stable ("homophone.et_est", "ortho.distance", etc.)
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
