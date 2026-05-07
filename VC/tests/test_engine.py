"""Tests pour engine.py — VCEngine orchestrateur."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lectura_vc._chargeur import RVC_SPEAKERS
from lectura_vc.engine import VCEngine


@pytest.fixture
def models_dir(tmp_path):
    """Cree un repertoire de modeles factice complet."""
    (tmp_path / "hubert.onnx").write_bytes(b"fake")
    (tmp_path / "rmvpe.onnx").write_bytes(b"fake")
    for speaker in RVC_SPEAKERS:
        (tmp_path / f"synthesizer_{speaker}.onnx").write_bytes(b"fake")
    (tmp_path / "openvoice_se.onnx").write_bytes(b"fake")
    (tmp_path / "openvoice_vc.onnx").write_bytes(b"fake")
    return tmp_path


class TestVCEngineInit:
    def test_basic_init(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        assert engine.models_dir == models_dir
        assert engine.mode == "auto"

    def test_with_speaker(self, models_dir):
        engine = VCEngine(speaker="ezwa", models_dir=models_dir)
        assert engine.default_speaker == "ezwa"

    def test_missing_models_dir(self, tmp_path):
        """Aucun modele trouve -> FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            VCEngine(models_dir=nonexistent)


class TestModeResolution:
    def test_auto_with_speaker(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        mode = engine._resolve_mode("ezwa", None, "auto")
        assert mode == "rvc"

    def test_auto_with_reference(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        mode = engine._resolve_mode(None, "ref.wav", "auto")
        assert mode == "zeroshot"

    def test_auto_with_both(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        mode = engine._resolve_mode("ezwa", "ref.wav", "auto")
        assert mode == "cascade"

    def test_auto_with_nothing(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        with pytest.raises(ValueError, match="necessite"):
            engine._resolve_mode(None, None, "auto")

    def test_explicit_mode(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        mode = engine._resolve_mode("ezwa", "ref.wav", "rvc")
        assert mode == "rvc"


class TestAvailableSpeakers:
    def test_full_models(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        speakers = engine.available_speakers
        assert set(speakers) == set(RVC_SPEAKERS)

    def test_partial_models(self, tmp_path):
        (tmp_path / "hubert.onnx").write_bytes(b"fake")
        (tmp_path / "rmvpe.onnx").write_bytes(b"fake")
        (tmp_path / "synthesizer_ezwa.onnx").write_bytes(b"fake")
        engine = VCEngine(models_dir=tmp_path)
        assert engine.available_speakers == ["ezwa"]


class TestHasOpenvoice:
    def test_with_models(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        assert engine.has_openvoice is True

    def test_without_models(self, tmp_path):
        (tmp_path / "hubert.onnx").write_bytes(b"fake")
        (tmp_path / "rmvpe.onnx").write_bytes(b"fake")
        engine = VCEngine(models_dir=tmp_path)
        assert engine.has_openvoice is False


class TestLoadAudio:
    def test_numpy_with_sr(self):
        audio = np.random.randn(16000).astype(np.float32)
        result, sr = VCEngine._load_audio(audio, 16000)
        assert sr == 16000
        np.testing.assert_array_equal(result, audio)

    def test_numpy_without_sr(self):
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="sr_in requis"):
            VCEngine._load_audio(audio, None)


class TestConvertValidation:
    def test_rvc_unknown_speaker(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="inconnu"):
            engine.convert(audio, speaker="unknown", mode="rvc", sr_in=16000)

    def test_rvc_no_speaker(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="necessite"):
            engine.convert(audio, mode="rvc", sr_in=16000)

    def test_zeroshot_no_reference(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="necessite"):
            engine.convert(audio, mode="zeroshot", sr_in=16000)

    def test_unknown_mode(self, models_dir):
        engine = VCEngine(models_dir=models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="inconnu"):
            engine.convert(audio, mode="invalid", sr_in=16000)
