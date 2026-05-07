"""Tests pour _chargeur.py — localisateur de modeles VC."""

import os
import tempfile
from pathlib import Path

import pytest

from lectura_vc._chargeur import (
    RVC_SPEAKERS,
    find_models_dir,
    has_openvoice_models,
    has_rvc_models,
    get_model_path,
    load_model_bytes,
    _file_exists,
)


@pytest.fixture
def models_dir(tmp_path):
    """Cree un repertoire de modeles factice."""
    # RVC models
    (tmp_path / "hubert.onnx").write_bytes(b"fake-hubert")
    (tmp_path / "rmvpe.onnx").write_bytes(b"fake-rmvpe")
    for speaker in RVC_SPEAKERS:
        (tmp_path / f"synthesizer_{speaker}.onnx").write_bytes(b"fake-synth")
    # OpenVoice models
    (tmp_path / "openvoice_se.onnx").write_bytes(b"fake-se")
    (tmp_path / "openvoice_vc.onnx").write_bytes(b"fake-vc")
    return tmp_path


@pytest.fixture
def partial_dir(tmp_path):
    """Repertoire avec modeles partiels (RVC seulement)."""
    (tmp_path / "hubert.onnx").write_bytes(b"fake")
    (tmp_path / "rmvpe.onnx").write_bytes(b"fake")
    (tmp_path / "synthesizer_ezwa.onnx").write_bytes(b"fake")
    return tmp_path


class TestFileExists:
    def test_plain_file(self, models_dir):
        assert _file_exists(models_dir, "hubert.onnx")

    def test_encrypted_file(self, tmp_path):
        (tmp_path / "model.onnx.enc").write_bytes(b"encrypted")
        assert _file_exists(tmp_path, "model.onnx")

    def test_missing_file(self, tmp_path):
        assert not _file_exists(tmp_path, "absent.onnx")


class TestHasRvcModels:
    def test_all_present(self, models_dir):
        assert has_rvc_models(models_dir)

    def test_with_speaker(self, models_dir):
        assert has_rvc_models(models_dir, speaker="ezwa")
        assert has_rvc_models(models_dir, speaker="bernard")

    def test_missing_speaker(self, partial_dir):
        assert has_rvc_models(partial_dir, speaker="ezwa")
        assert not has_rvc_models(partial_dir, speaker="bernard")

    def test_empty_dir(self, tmp_path):
        assert not has_rvc_models(tmp_path)

    def test_nonexistent_dir(self):
        assert not has_rvc_models(Path("/nonexistent"))


class TestHasOpenvoiceModels:
    def test_all_present(self, models_dir):
        assert has_openvoice_models(models_dir)

    def test_missing(self, partial_dir):
        assert not has_openvoice_models(partial_dir)


class TestFindModelsDir:
    def test_explicit_path(self, models_dir):
        result = find_models_dir(models_dir)
        assert result == models_dir

    def test_env_variable(self, models_dir, monkeypatch):
        # models_dir doit etre dans un sous-dossier "vc"
        parent = models_dir.parent
        vc_dir = parent / "vc"
        vc_dir.mkdir(exist_ok=True)
        (vc_dir / "hubert.onnx").write_bytes(b"fake")
        monkeypatch.setenv("LECTURA_MODELS_DIR", str(parent))
        result = find_models_dir()
        assert result == vc_dir

    def test_nothing_found(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LECTURA_MODELS_DIR", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = find_models_dir()
        assert result is None


class TestLoadModelBytes:
    def test_plain(self, models_dir):
        data = load_model_bytes(models_dir, "hubert.onnx")
        assert data == b"fake-hubert"

    def test_missing(self, models_dir):
        data = load_model_bytes(models_dir, "absent.onnx")
        assert data is None


class TestGetModelPath:
    def test_found(self, models_dir):
        path = get_model_path(models_dir, "hubert.onnx")
        assert path == models_dir / "hubert.onnx"

    def test_encrypted(self, tmp_path):
        (tmp_path / "model.onnx.enc").write_bytes(b"enc")
        path = get_model_path(tmp_path, "model.onnx")
        assert path == tmp_path / "model.onnx.enc"

    def test_missing(self, tmp_path):
        path = get_model_path(tmp_path, "absent.onnx")
        assert path is None


class TestRvcSpeakers:
    def test_known_speakers(self):
        assert "ezwa" in RVC_SPEAKERS
        assert "bernard" in RVC_SPEAKERS
        assert "siwis" in RVC_SPEAKERS
        assert len(RVC_SPEAKERS) == 6
