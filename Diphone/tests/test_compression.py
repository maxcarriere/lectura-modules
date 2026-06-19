"""Tests pour lectura_tts_diphone._compression."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_fake_diphones(n_types=5, n_frames=20, n_bins=1025):
    """Cree un dict de diphones factices."""
    data = {}
    keys = ["#-b", "b-a", "a-l", "l-a", "a-#"][:n_types]
    for key in keys:
        data[key] = {
            "f0": np.random.uniform(100, 300, n_frames).astype(np.float64),
            "sp": np.random.uniform(1e-8, 1.0, (n_frames, n_bins)).astype(np.float64),
            "ap": np.random.uniform(0.0, 1.0, (n_frames, n_bins)).astype(np.float64),
            "sr": 44100,
            "frame_period": 5.0,
            "n_frames": n_frames,
        }
    return data


def test_save_load_roundtrip():
    """save → load preserves keys and shapes."""
    from lectura_tts_diphone._compression import save_compressed, load_compressed

    data = _make_fake_diphones()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dpk.gz"
        save_compressed(data, path)

        assert path.exists()
        assert path.stat().st_size > 0

        loaded = load_compressed(path)

    assert set(loaded.keys()) == set(data.keys())
    for key in data:
        assert loaded[key]["f0"].shape == data[key]["f0"].shape
        assert loaded[key]["sp"].dtype == np.float64
        assert loaded[key]["ap"].dtype == np.float64
        assert loaded[key]["f0"].dtype == np.float64


def test_f0_preserved():
    """F0 roundtrip: float64 → float32 → float64 (precision OK)."""
    from lectura_tts_diphone._compression import save_compressed, load_compressed

    data = _make_fake_diphones(n_types=1)
    key = list(data.keys())[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dpk.gz"
        save_compressed(data, path)
        loaded = load_compressed(path)

    np.testing.assert_allclose(loaded[key]["f0"], data[key]["f0"],
                               rtol=1e-5, atol=1e-3)


def test_sp_compression_quality():
    """SP log-float16 roundtrip: check relative error."""
    from lectura_tts_diphone._compression import save_compressed, load_compressed

    data = _make_fake_diphones(n_types=1, n_bins=200)
    key = list(data.keys())[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dpk.gz"
        save_compressed(data, path, max_freq=50000)  # no truncation
        loaded = load_compressed(path)

    orig_sp = data[key]["sp"]
    loaded_sp = loaded[key]["sp"]
    # SNR should be reasonable
    noise = orig_sp - loaded_sp
    signal_power = np.mean(orig_sp ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
        assert snr_db > 10, f"SNR trop bas: {snr_db:.1f} dB"


def test_ap_uint8_roundtrip():
    """AP uint8 roundtrip: max error < 1/255."""
    from lectura_tts_diphone._compression import save_compressed, load_compressed

    data = _make_fake_diphones(n_types=1, n_bins=100)
    key = list(data.keys())[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dpk.gz"
        save_compressed(data, path, max_freq=50000)
        loaded = load_compressed(path)

    np.testing.assert_allclose(loaded[key]["ap"], data[key]["ap"],
                               atol=1.0 / 255 + 1e-6)


def test_truncation():
    """FFT truncation reduces bin count."""
    from lectura_tts_diphone._compression import save_compressed, load_compressed

    data = _make_fake_diphones(n_types=1, n_bins=1025)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dpk.gz"
        save_compressed(data, path, max_freq=8000, sr=44100)
        loaded = load_compressed(path)

    key = list(data.keys())[0]
    assert loaded[key]["sp"].shape[1] < 1025
