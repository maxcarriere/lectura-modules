"""Types centraux pour lectura-tts.

Protocole TTSEngine, dataclasses de résultats et enums de configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class PhonemeTiming:
    """Timing d'un phonème dans l'audio synthétisé."""

    ipa: str
    start_ms: float
    end_ms: float


@dataclass
class TTSResult:
    """Résultat d'une synthèse TTS : audio + timings phonèmes."""

    samples: np.ndarray  # float32, mono
    sample_rate: int
    phoneme_timings: list[PhonemeTiming] = field(default_factory=list)


class InputType(Enum):
    """Type d'entrée pour la synthèse."""

    ORTHOGRAPHE = "orthographe"
    PHONEMIQUE = "phonemique"


class Granularity(Enum):
    """Granularité de la lecture."""

    FLUIDE = "fluide"
    MOT_A_MOT = "mot_a_mot"
    SYLLABES = "syllabes"


@dataclass
class SyllabeTiming:
    """Timing d'une syllabe dans l'audio TTS."""

    phone: str
    ortho: str
    start_ms: float
    end_ms: float


@dataclass
class GroupeTiming:
    """Timings de toutes les syllabes d'un groupe."""

    group_index: int
    syllabe_timings: list[SyllabeTiming] = field(default_factory=list)


@runtime_checkable
class TTSEngine(Protocol):
    """Protocole pour un moteur TTS avec timestamps phonémiques."""

    def synthesize(self, text: str) -> TTSResult: ...

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult: ...
