"""Export WAV via le module wave (stdlib)."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

from .._util import normalize_audio


def export_wav(
    samples: np.ndarray,
    sample_rate: int,
    output: str | Path,
    *,
    normalize: bool = True,
) -> Path:
    """Exporte un array float32 en fichier WAV 16 bits mono.

    Parameters
    ----------
    samples : np.ndarray
        Audio en float32 (valeurs entre -1.0 et 1.0).
    sample_rate : int
        Fréquence d'échantillonnage.
    output : str | Path
        Chemin du fichier de sortie.
    normalize : bool
        Si True, normalise le volume avant conversion.

    Returns
    -------
    Path
        Chemin absolu du fichier créé.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    audio = samples.astype(np.float32)
    if normalize:
        audio = normalize_audio(audio)

    # float32 → int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    with wave.open(str(output), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return output.resolve()
