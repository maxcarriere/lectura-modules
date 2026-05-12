"""Tests pour l'export WAV."""

import wave
from pathlib import Path

import numpy as np
import pytest

from lectura_exporter.audio.wav import export_wav


def test_export_wav_creates_file(tmp_path: Path):
    """Vérifie qu'un fichier WAV valide est créé."""
    samples = np.random.uniform(-1, 1, 22050).astype(np.float32)  # 1 seconde
    output = tmp_path / "test.wav"

    result = export_wav(samples, 22050, output)

    assert result.exists()
    assert result.suffix == ".wav"


def test_export_wav_readable(tmp_path: Path):
    """Vérifie que le WAV créé est lisible avec le module wave."""
    sr = 22050
    duration_s = 0.5
    samples = np.random.uniform(-1, 1, int(sr * duration_s)).astype(np.float32)
    output = tmp_path / "test.wav"

    export_wav(samples, sr, output)

    with wave.open(str(output), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == sr
        assert wf.getnframes() == len(samples)


def test_export_wav_creates_parent_dirs(tmp_path: Path):
    """Vérifie que les répertoires parents sont créés."""
    samples = np.zeros(100, dtype=np.float32)
    output = tmp_path / "sub" / "dir" / "test.wav"

    result = export_wav(samples, 22050, output)

    assert result.exists()


def test_export_wav_no_normalize(tmp_path: Path):
    """Vérifie l'export sans normalisation."""
    samples = np.array([0.5, -0.5, 0.25], dtype=np.float32)
    output = tmp_path / "test.wav"

    result = export_wav(samples, 22050, output, normalize=False)

    assert result.exists()
    with wave.open(str(output), "rb") as wf:
        assert wf.getnframes() == 3
