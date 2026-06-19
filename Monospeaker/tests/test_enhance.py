"""Tests unitaires pour le module _enhance."""

import numpy as np
import pytest


def test_enhance_mel_shape():
    from lectura_tts_monospeaker._enhance import enhance_mel

    mel = np.random.randn(80, 100).astype(np.float32) * 2 - 5
    result = enhance_mel(mel)
    assert result.shape == (80, 100)


def test_enhance_mel_clipping():
    from lectura_tts_monospeaker._enhance import enhance_mel

    mel = np.full((80, 50), -20.0, dtype=np.float32)
    result = enhance_mel(mel)
    assert result.min() >= -11.5
    assert result.max() <= 2.0


def test_enhance_mel_spectral_boost():
    from lectura_tts_monospeaker._enhance import enhance_mel

    # Creer un mel avec variance inter-bandes connue
    mel = np.zeros((80, 50), dtype=np.float32)
    mel[0, :] = -2.0
    mel[79, :] = -8.0
    mel[40, :] = -5.0

    result = enhance_mel(mel, spectral_alpha=0.5, temporal_alpha=0.0)

    # Le contraste spectral devrait etre amplifie
    frame_range_orig = mel[:, 0].max() - mel[:, 0].min()
    frame_range_enh = result[:, 25].max() - result[:, 25].min()
    assert frame_range_enh > frame_range_orig


def test_noise_gate():
    from lectura_tts_monospeaker._enhance import noise_gate

    mel = np.full((80, 50), -10.0, dtype=np.float32)
    result = noise_gate(mel, threshold=-8.0, silence_val=-11.5)

    # Frames sous le seuil doivent etre plus proches du silence
    assert result.mean() < mel.mean()


def test_fade_out():
    from lectura_tts_monospeaker._enhance import fade_out

    mel = np.zeros((80, 50), dtype=np.float32) - 5.0
    result = fade_out(mel, n_frames=5, silence_val=-11.5)

    # Les derniers frames doivent se rapprocher de silence_val
    assert result[:, -1].mean() == pytest.approx(-11.5, abs=0.01)
    assert result[:, -5].mean() > result[:, -1].mean()
