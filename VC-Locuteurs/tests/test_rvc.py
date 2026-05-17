"""Tests pour _rvc.py --- RVC ONNX inference (avec mocks)."""

import numpy as np
import pytest

from lectura_vc_locuteurs._rvc import (
    SR_16K,
    SR_48K,
    SYNTH_FIXED_T,
    calculate_f0_bins,
    compute_mel,
    decode_f0,
    normalize_audio_for_hubert,
)


class TestNormalizeAudio:
    def test_zero_mean(self):
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        normalized = normalize_audio_for_hubert(audio)
        assert abs(np.mean(normalized)) < 1e-6

    def test_unit_variance(self):
        audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        normalized = normalize_audio_for_hubert(audio)
        assert abs(np.var(normalized) - 1.0) < 0.01

    def test_silent_audio(self):
        """Silence ne devrait pas planter."""
        audio = np.zeros(1000, dtype=np.float32)
        normalized = normalize_audio_for_hubert(audio)
        assert np.all(np.isfinite(normalized))


class TestComputeMel:
    def test_shape(self):
        """Le mel doit avoir la bonne forme."""
        audio = np.random.randn(SR_16K).astype(np.float32)  # 1 seconde
        mel = compute_mel(audio)
        assert mel.ndim == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == 128
        assert mel.shape[2] > 0

    def test_dtype(self):
        audio = np.random.randn(SR_16K).astype(np.float32)
        mel = compute_mel(audio)
        assert mel.dtype == np.float32


class TestDecodeF0:
    def test_silent_frames(self):
        """Frames avec salience < threshold -> F0 = 0."""
        hidden = np.zeros((10, 360), dtype=np.float32)
        f0 = decode_f0(hidden, thred=0.03)
        assert np.all(f0 == 0)

    def test_voiced_frames(self):
        """Frames avec un pic de salience -> F0 > 0."""
        hidden = np.zeros((5, 360), dtype=np.float32)
        hidden[:, 100] = 0.8
        hidden[:, 99] = 0.3
        hidden[:, 101] = 0.3
        f0 = decode_f0(hidden, thred=0.03)
        assert np.all(f0 > 0)

    def test_output_shape(self):
        hidden = np.zeros((20, 360), dtype=np.float32)
        f0 = decode_f0(hidden)
        assert f0.shape == (20,)


class TestCalculateF0Bins:
    def test_zero_f0(self):
        """F0=0 -> bin=1."""
        f0 = np.array([0.0])
        bins = calculate_f0_bins(f0)
        assert bins[0] == 1

    def test_in_range(self):
        """F0 dans la plage -> bins dans [1, 255]."""
        f0 = np.array([100.0, 200.0, 300.0, 500.0])
        bins = calculate_f0_bins(f0)
        assert np.all(bins >= 1)
        assert np.all(bins <= 255)

    def test_monotonic(self):
        """Plus F0 est haut, plus le bin est eleve."""
        f0 = np.array([100.0, 200.0, 400.0])
        bins = calculate_f0_bins(f0)
        assert bins[0] < bins[1] < bins[2]

    def test_clipping_high(self):
        """F0 tres haut -> bin = 255."""
        f0 = np.array([5000.0])
        bins = calculate_f0_bins(f0)
        assert bins[0] == 255


class TestConstants:
    def test_sample_rates(self):
        assert SR_16K == 16000
        assert SR_48K == 48000

    def test_synth_fixed_t(self):
        assert SYNTH_FIXED_T == 1000


class TestRVCConverterInit:
    def test_missing_models(self, tmp_path):
        """Init avec modeles manquants -> FileNotFoundError."""
        from lectura_vc_locuteurs._rvc import RVCConverter

        with pytest.raises(FileNotFoundError):
            RVCConverter(tmp_path, "ezwa")
