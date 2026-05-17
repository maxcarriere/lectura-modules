"""Tests pour engine.py --- LocuteursEngine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lectura_vc_locuteurs._chargeur import RVC_SPEAKERS
from lectura_vc_locuteurs.engine import LocuteursEngine


@pytest.fixture
def models_dir(tmp_path):
    """Cree un repertoire de modeles factice complet."""
    (tmp_path / "hubert.onnx").write_bytes(b"fake")
    (tmp_path / "rmvpe.onnx").write_bytes(b"fake")
    for speaker in RVC_SPEAKERS:
        (tmp_path / f"synthesizer_{speaker}.onnx").write_bytes(b"fake")
    return tmp_path


class TestLocuteursEngineInit:
    def test_basic_init(self, models_dir):
        engine = LocuteursEngine(models_dir=models_dir)
        assert engine.models_dir == models_dir
        assert engine.default_speaker is None

    def test_with_speaker(self, models_dir):
        engine = LocuteursEngine(speaker="ezwa", models_dir=models_dir)
        assert engine.default_speaker == "ezwa"

    def test_missing_models_dir(self, tmp_path, monkeypatch):
        """Aucun modele trouve -> FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.delenv("LECTURA_MODELS_DIR", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "fakehome")
        monkeypatch.setattr(
            "lectura_vc_locuteurs._chargeur._PACKAGE_MODELS",
            tmp_path / "nopkg",
        )
        with pytest.raises(FileNotFoundError):
            LocuteursEngine(models_dir=nonexistent)


class TestAvailableSpeakers:
    def test_full_models(self, models_dir):
        engine = LocuteursEngine(models_dir=models_dir)
        speakers = engine.available_speakers
        assert set(speakers) == set(RVC_SPEAKERS)

    def test_partial_models(self, tmp_path):
        (tmp_path / "hubert.onnx").write_bytes(b"fake")
        (tmp_path / "rmvpe.onnx").write_bytes(b"fake")
        (tmp_path / "synthesizer_ezwa.onnx").write_bytes(b"fake")
        engine = LocuteursEngine(models_dir=tmp_path)
        assert engine.available_speakers == ["ezwa"]


class TestConvertValidation:
    def test_unknown_speaker(self, models_dir):
        engine = LocuteursEngine(models_dir=models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="inconnu"):
            engine.convert(audio, speaker="unknown", sr_in=16000)

    def test_no_speaker(self, models_dir):
        engine = LocuteursEngine(models_dir=models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="requis"):
            engine.convert(audio, sr_in=16000)

    def test_numpy_without_sr(self, models_dir):
        engine = LocuteursEngine(models_dir=models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="sr_in requis"):
            engine.convert(audio, speaker="ezwa")


class TestLoadAudio:
    def test_numpy_with_sr(self):
        audio = np.random.randn(16000).astype(np.float32)
        result, sr = LocuteursEngine._load_audio(audio, 16000)
        assert sr == 16000
        np.testing.assert_array_equal(result, audio)

    def test_numpy_without_sr(self):
        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="sr_in requis"):
            LocuteursEngine._load_audio(audio, None)
