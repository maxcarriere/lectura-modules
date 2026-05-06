"""Tests unitaires pour le module _chargeur."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _create_fake_models(directory: Path, speaker: str = "siwis"):
    """Cree un jeu de modeles factices (layout legacy) dans un repertoire."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "decoder.onnx").write_bytes(b"FAKE")
    (directory / "hifigan.onnx").write_bytes(b"FAKE")
    (directory / f"encoder_{speaker}.onnx").write_bytes(b"FAKE")
    (directory / "config.json").write_text("{}")
    (directory / "phoneme_vocab.json").write_text('{"vocab":[],"phone2id":{}}')


def _create_fake_unified_models(directory: Path):
    """Cree un jeu de modeles factices (layout unifie) dans un repertoire."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "decoder.onnx").write_bytes(b"FAKE")
    (directory / "hifigan.onnx").write_bytes(b"FAKE")
    (directory / "encoder.onnx").write_bytes(b"FAKE")
    (directory / "config.json").write_text("{}")
    (directory / "phoneme_vocab.json").write_text('{"vocab":[],"phone2id":{}}')


def test_find_models_dir_explicit(tmp_path):
    """Parametre explicite models_dir."""
    from lectura_tts_multispeaker._chargeur import find_models_dir

    _create_fake_models(tmp_path)
    result = find_models_dir("siwis", tmp_path)
    assert result == tmp_path


def test_find_models_dir_explicit_unified(tmp_path):
    """Parametre explicite models_dir avec layout unifie."""
    from lectura_tts_multispeaker._chargeur import find_models_dir

    _create_fake_unified_models(tmp_path)
    # Unified layout should work for any speaker
    result = find_models_dir("bernard", tmp_path)
    assert result == tmp_path


def test_has_models_missing(tmp_path):
    """Repertoire sans modeles retourne False."""
    from lectura_tts_multispeaker._chargeur import _has_models

    assert _has_models(tmp_path, "siwis") is False


def test_find_models_dir_env(tmp_path, monkeypatch):
    """Variable d'environnement LECTURA_MODELS_DIR."""
    from lectura_tts_multispeaker._chargeur import find_models_dir

    models_dir = tmp_path / "tts_multispeaker"
    _create_fake_models(models_dir)
    monkeypatch.setenv("LECTURA_MODELS_DIR", str(tmp_path))

    result = find_models_dir("siwis")
    assert result == models_dir


def test_has_models_wrong_speaker_legacy(tmp_path):
    """Legacy layout : retourne False si l'encodeur du speaker manque."""
    from lectura_tts_multispeaker._chargeur import _has_models

    _create_fake_models(tmp_path, speaker="siwis")
    assert _has_models(tmp_path, "bernard") is False


def test_has_models_unified_any_speaker(tmp_path):
    """Unified layout : encoder.onnx accepte n'importe quel speaker."""
    from lectura_tts_multispeaker._chargeur import _has_models

    _create_fake_unified_models(tmp_path)
    assert _has_models(tmp_path, "siwis") is True
    assert _has_models(tmp_path, "bernard") is True
    assert _has_models(tmp_path, "unknown") is True


def test_find_models_dir_multiple_speakers(tmp_path):
    """Trouve le repertoire si l'encodeur du speaker est present."""
    from lectura_tts_multispeaker._chargeur import find_models_dir

    _create_fake_models(tmp_path, speaker="siwis")
    (tmp_path / "encoder_bernard.onnx").write_bytes(b"FAKE")

    result = find_models_dir("bernard", tmp_path)
    assert result == tmp_path


def test_has_models_encrypted(tmp_path):
    """Accepte les fichiers .enc."""
    from lectura_tts_multispeaker._chargeur import _has_models

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "decoder.enc").write_bytes(b"FAKE")
    (tmp_path / "hifigan.enc").write_bytes(b"FAKE")
    (tmp_path / "encoder_siwis.enc").write_bytes(b"FAKE")

    assert _has_models(tmp_path, "siwis") is True
    assert _has_models(tmp_path, "bernard") is False


def test_has_models_encrypted_unified(tmp_path):
    """Accepte les fichiers .enc en layout unifie."""
    from lectura_tts_multispeaker._chargeur import _has_models

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "decoder.enc").write_bytes(b"FAKE")
    (tmp_path / "hifigan.enc").write_bytes(b"FAKE")
    (tmp_path / "encoder.enc").write_bytes(b"FAKE")

    assert _has_models(tmp_path, "siwis") is True
    assert _has_models(tmp_path, "bernard") is True


def test_load_model_bytes_plaintext(tmp_path):
    """Charge un fichier .onnx en clair."""
    from lectura_tts_multispeaker._chargeur import load_model_bytes

    data = b"FAKE ONNX DATA"
    (tmp_path / "decoder.onnx").write_bytes(data)

    result = load_model_bytes(tmp_path, "decoder.onnx")
    assert result == data


def test_load_model_bytes_encrypted(tmp_path):
    """Charge un fichier .enc chiffre."""
    from lectura_tts_multispeaker._chargeur import load_model_bytes
    from lectura_tts_multispeaker._crypto import encrypt_model

    data = b"REAL ONNX MODEL BYTES" * 100
    onnx_path = tmp_path / "model_temp.onnx"
    onnx_path.write_bytes(data)
    encrypt_model(onnx_path, tmp_path / "decoder.enc")
    onnx_path.unlink()

    result = load_model_bytes(tmp_path, "decoder.onnx")
    assert result == data


def test_load_model_bytes_not_found(tmp_path):
    """Retourne None si le fichier n'existe pas."""
    from lectura_tts_multispeaker._chargeur import load_model_bytes

    result = load_model_bytes(tmp_path, "inexistant.onnx")
    assert result is None
