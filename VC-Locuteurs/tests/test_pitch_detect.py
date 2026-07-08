"""Tests pour _pitch_detect.py --- detection F0 et auto-adaptation."""

import numpy as np
import pytest

from lectura_vc_locuteurs._pitch_detect import (
    auto_adapt,
    compute_pitch_shift,
    compute_protect,
    detect_mean_f0,
    get_speaker_f0,
    get_speaker_gender,
)


class TestDetectMeanF0:
    def test_sine_wave(self):
        """Un sinus pur a 200 Hz doit etre detecte."""
        sr = 16000
        duration = 2.0
        freq = 200.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        f0 = detect_mean_f0(audio, sr=sr)
        assert 180 < f0 < 220, f"F0 detecte: {f0}"

    def test_silence(self):
        """Silence -> F0 = 0."""
        audio = np.zeros(16000, dtype=np.float32)
        f0 = detect_mean_f0(audio, sr=16000)
        assert f0 == 0.0

    def test_noise(self):
        """Bruit blanc -> F0 faible ou 0."""
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(32000).astype(np.float32) * 0.01
        f0 = detect_mean_f0(audio, sr=16000)
        assert isinstance(f0, float)


class TestSpeakerMetadata:
    def test_known_speakers(self):
        assert get_speaker_f0("ezwa") > 0
        assert get_speaker_f0("bernard") > 0

    def test_unknown_speaker(self):
        assert get_speaker_f0("inconnu") == 0.0

    def test_gender(self):
        assert get_speaker_gender("ezwa") == "F"
        assert get_speaker_gender("bernard") == "M"
        assert get_speaker_gender("inconnu") == ""

    def test_female_higher_f0(self):
        """Les voix feminines ont une F0 plus elevee que les masculines."""
        f_f0 = get_speaker_f0("ezwa")
        m_f0 = get_speaker_f0("bernard")
        assert f_f0 > m_f0


class TestComputePitchShift:
    def test_same_f0(self):
        """Meme F0 -> shift ~ 0."""
        shift = compute_pitch_shift(200.0, "ezwa")  # ezwa ~195 Hz
        assert abs(shift) < 1.0

    def test_octave_up(self):
        """Source 100 Hz -> cible 200 Hz = +12 demi-tons."""
        shift = compute_pitch_shift(100.0, "nadine")  # nadine ~200 Hz
        assert abs(shift - 12.0) < 1.0

    def test_invalid_f0(self):
        """F0 invalide -> shift = 0."""
        assert compute_pitch_shift(0.0, "ezwa") == 0.0
        assert compute_pitch_shift(200.0, "inconnu") == 0.0


class TestComputeProtect:
    def test_small_shift(self):
        assert compute_protect(2.0) == 0.5

    def test_large_shift(self):
        assert compute_protect(6.0) == 0.33

    def test_negative_large_shift(self):
        assert compute_protect(-5.0) == 0.33


class TestAutoAdapt:
    def test_returns_tuple(self):
        """Retourne (pitch_shift, protect)."""
        sr = 16000
        freq = 200.0
        t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        pitch_shift, protect = auto_adapt(audio, sr, "bernard")
        assert pitch_shift < 0
        assert protect in (0.33, 0.5)
