"""lectura-exporter — Module d'export pour Lectura."""

__version__ = "0.1.0"

from .models import (
    AudioExportOptions,
    AudioSegment,
    ExportTimingData,
    FrameHighlight,
    GroupeTiming,
    SyllabeTiming,
    TextExportOptions,
    VideoExportOptions,
)
from .audio import export_audio
from .audio.wav import export_wav
from .audio.mp3 import export_mp3
from .audio.timing import export_timing_json, timing_to_dict
from ._util import concatenate_segments, normalize_audio

__all__ = [
    "AudioExportOptions",
    "AudioSegment",
    "ExportTimingData",
    "FrameHighlight",
    "GroupeTiming",
    "SyllabeTiming",
    "TextExportOptions",
    "VideoExportOptions",
    "export_audio",
    "export_wav",
    "export_mp3",
    "export_timing_json",
    "timing_to_dict",
    "concatenate_segments",
    "normalize_audio",
]
