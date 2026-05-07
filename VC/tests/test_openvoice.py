"""Tests pour _openvoice.py — OpenVoice ONNX inference (avec mocks)."""

import numpy as np
import pytest

from lectura_vc._openvoice import (
    OV_SR,
    OV_N_FFT,
    OV_HOP,
    OV_N_FREQ,
    compute_spectrogram,
)


class TestComputeSpectrogram:
    def test_shape(self):
        """Le spectrogram doit avoir la bonne forme."""
        audio = np.random.randn(OV_SR).astype(np.float32)  # 1 seconde
        spec = compute_spectrogram(audio)
        assert spec.ndim == 3
        assert spec.shape[0] == 1
        assert spec.shape[1] == OV_N_FREQ  # 513
        assert spec.shape[2] > 0

    def test_dtype(self):
        audio = np.random.randn(OV_SR).astype(np.float32)
        spec = compute_spectrogram(audio)
        assert spec.dtype == np.float32

    def test_positive_values(self):
        """Magnitude + epsilon -> toujours positif."""
        audio = np.random.randn(OV_SR).astype(np.float32)
        spec = compute_spectrogram(audio)
        assert np.all(spec > 0)

    def test_silent_audio(self):
        """Silence -> spectrogram presque nul mais pas zero (epsilon)."""
        audio = np.zeros(OV_SR, dtype=np.float32)
        spec = compute_spectrogram(audio)
        # sqrt(0 + 1e-6) = 0.001
        assert np.allclose(spec, 0.001, atol=1e-4)

    def test_n_frames_approximation(self):
        """Nombre de frames approximativement correct."""
        duration = 2.0
        audio = np.random.randn(int(OV_SR * duration)).astype(np.float32)
        spec = compute_spectrogram(audio)
        expected_frames = int(OV_SR * duration / OV_HOP) + 1
        # Tolerance de quelques frames (padding)
        assert abs(spec.shape[2] - expected_frames) < 5


class TestConstants:
    def test_sample_rate(self):
        assert OV_SR == 22050

    def test_fft_params(self):
        assert OV_N_FFT == 1024
        assert OV_HOP == 256
        assert OV_N_FREQ == 513


class TestOpenVoiceConverterInit:
    def test_missing_models(self, tmp_path):
        """Init avec modeles manquants -> FileNotFoundError."""
        from lectura_vc._openvoice import OpenVoiceConverter

        with pytest.raises(FileNotFoundError):
            OpenVoiceConverter(tmp_path)
