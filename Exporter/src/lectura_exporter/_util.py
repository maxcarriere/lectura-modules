"""Utilitaires internes pour l'export audio."""

from __future__ import annotations

import numpy as np

from .models import AudioSegment


def concatenate_segments(
    segments: list[AudioSegment],
    pause_ms: float = 200.0,
    sample_rate: int = 22050,
) -> tuple[np.ndarray, list[float]]:
    """Concatène les AudioSegment avec pauses.

    Returns
    -------
    audio : np.ndarray
        Audio complet (float32).
    offsets : list[float]
        Offset en ms du début de chaque segment dans l'audio final.
    """
    if not segments:
        return np.array([], dtype=np.float32), []

    pause_samples = int(pause_ms * sample_rate / 1000)
    pause = np.zeros(pause_samples, dtype=np.float32)

    parts: list[np.ndarray] = []
    offsets: list[float] = []
    current_samples = 0

    for i, seg in enumerate(segments):
        offsets.append(current_samples / sample_rate * 1000)
        samples = seg.samples.astype(np.float32)
        parts.append(samples)
        current_samples += len(samples)

        if i < len(segments) - 1:
            parts.append(pause)
            current_samples += pause_samples

    audio = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return audio, offsets


def normalize_audio(samples: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalise l'amplitude à un pic cible."""
    if samples.size == 0:
        return samples
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    return (samples * (target_peak / peak)).astype(np.float32)
