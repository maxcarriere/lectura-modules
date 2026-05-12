"""Export MP3 via pydub + tags ID3 via mutagen (optionnel)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from ..models import AudioExportOptions
from .._util import normalize_audio


def _require_pydub():
    try:
        from pydub import AudioSegment
        return AudioSegment
    except ImportError:
        raise ImportError(
            "L'export MP3 nécessite pydub et ffmpeg.\n"
            "Installation : pip install lectura-exporter[mp3]\n"
            "ffmpeg doit être installé sur le système."
        )


def _require_mutagen():
    try:
        from mutagen.id3 import ID3, TIT2, TPE1, COMM
        return ID3, TIT2, TPE1, COMM
    except ImportError:
        return None


def export_mp3(
    samples: np.ndarray,
    sample_rate: int,
    output: str | Path,
    options: AudioExportOptions | None = None,
) -> Path:
    """Exporte un array float32 en fichier MP3 avec tags ID3 optionnels.

    Parameters
    ----------
    samples : np.ndarray
        Audio en float32.
    sample_rate : int
        Fréquence d'échantillonnage.
    output : str | Path
        Chemin du fichier de sortie.
    options : AudioExportOptions | None
        Options (bitrate, tags ID3).

    Returns
    -------
    Path
        Chemin absolu du fichier créé.
    """
    PydubAudio = _require_pydub()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if options is None:
        options = AudioExportOptions(format="mp3")

    audio = samples.astype(np.float32)
    if options.normalize:
        audio = normalize_audio(audio)

    # float32 → int16 → WAV en mémoire → pydub
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)

    segment = PydubAudio.from_wav(buf)
    segment.export(str(output), format="mp3", bitrate=f"{options.mp3_bitrate}k")

    # Tags ID3
    mutagen_imports = _require_mutagen()
    if mutagen_imports is not None:
        ID3, TIT2, TPE1, COMM = mutagen_imports
        try:
            tags = ID3(str(output))
        except Exception:
            tags = ID3()

        if options.mp3_title:
            tags.add(TIT2(encoding=3, text=options.mp3_title))
        if options.mp3_artist:
            tags.add(TPE1(encoding=3, text=options.mp3_artist))
        if options.mp3_comment:
            tags.add(COMM(encoding=3, lang="fra", desc="", text=options.mp3_comment))

        tags.save(str(output))

    return output.resolve()
