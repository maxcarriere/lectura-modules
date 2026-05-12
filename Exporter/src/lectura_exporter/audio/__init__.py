"""Export audio — fonction de haut niveau."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import AudioExportOptions, AudioSegment, ExportTimingData
from .._util import concatenate_segments, normalize_audio
from .timing import export_timing_json
from .wav import export_wav


def export_audio(
    segments: list[AudioSegment],
    output: str | Path,
    timing_data: ExportTimingData | None = None,
    options: AudioExportOptions | None = None,
) -> dict[str, Path | None]:
    """Concatène les segments et exporte l'audio + timing JSON.

    Parameters
    ----------
    segments : list[AudioSegment]
        Segments audio à concaténer.
    output : str | Path
        Chemin du fichier audio de sortie (.wav ou .mp3).
    timing_data : ExportTimingData | None
        Si fourni et options.include_timing, écrit un JSON à côté.
    options : AudioExportOptions | None
        Options d'export.

    Returns
    -------
    dict[str, Path | None]
        ``{"audio": Path, "timing": Path | None}``
    """
    if options is None:
        options = AudioExportOptions()

    output = Path(output)
    sample_rate = segments[0].sample_rate if segments else 22050

    audio, offsets = concatenate_segments(segments, sample_rate=sample_rate)

    result: dict[str, Path | None] = {"audio": None, "timing": None}

    if options.format == "mp3":
        from .mp3 import export_mp3
        result["audio"] = export_mp3(audio, sample_rate, output, options)
    else:
        result["audio"] = export_wav(
            audio, sample_rate, output, normalize=options.normalize,
        )

    if timing_data is not None and options.include_timing:
        timing_path = output.with_suffix(".json")
        result["timing"] = export_timing_json(timing_data, timing_path)

    return result
