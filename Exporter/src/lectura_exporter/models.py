"""Types d'entrée pour l'export audio.

Dataclasses autonomes sans dépendance sur les autres modules Lectura.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioSegment:
    """Segment audio à exporter — miroir simplifié de PlaybackSegment sans PySide6."""

    samples: np.ndarray
    sample_rate: int
    group_index: int
    syllabe_index: int | None = None


@dataclass
class SyllabeTiming:
    """Timing d'une syllabe dans l'export."""

    phone: str
    ortho: str
    start_ms: float
    end_ms: float


@dataclass
class GroupeTiming:
    """Timing d'un groupe de lecture."""

    group_index: int
    text: str
    phone_groupe: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    syllabe_timings: list[SyllabeTiming] = field(default_factory=list)


@dataclass
class ExportTimingData:
    """Données de timing complètes pour l'export JSON."""

    text: str
    group_timings: list[GroupeTiming]
    total_duration_ms: float
    sample_rate: int
    granularity: str = "syllabes"


@dataclass
class AudioExportOptions:
    """Options d'export audio."""

    format: str = "wav"  # wav | mp3
    mp3_bitrate: int = 192
    mp3_title: str = ""
    mp3_artist: str = "Lectura"
    mp3_comment: str = ""
    include_timing: bool = True
    normalize: bool = True


@dataclass
class VideoExportOptions:
    """Options d'export vidéo."""

    format: str            # "mp4" | "webm" | "mov"
    scale_factor: float    # 1.0, 1.5, 2.0, 3.0
    fps: int               # 30 par défaut
    background: str        # "transparent" | "dark" | "custom"
    bg_color: str          # hex si custom, sinon ignoré
    output_path: str


@dataclass
class TextExportOptions:
    """Options d'export texte."""

    include_display: bool
    include_pos: bool
    include_morpho: bool
    include_phone: bool
    format: str          # "html" | "odt" | "docx" | "pdf"
    output_path: str


@dataclass
class FrameHighlight:
    """État de surlignage pour une frame vidéo."""

    action: str  # "group" | "syllabe" | "clear"
    group_index: int = -1
    syllabe_index: int = -1
