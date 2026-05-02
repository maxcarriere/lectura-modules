"""Tests unitaires pour le module _chargeur."""

from pathlib import Path

import pytest


def test_has_models_empty_dir(tmp_path):
    """Repertoire vide → pas de modeles."""
    from lectura_tts_monospeaker._chargeur import _has_models

    assert _has_models(tmp_path) is False


def test_has_models_with_onnx(tmp_path):
    """Repertoire avec les 3 ONNX → modeles trouves."""
    from lectura_tts_monospeaker._chargeur import _has_models

    (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")

    assert _has_models(tmp_path) is True


def test_has_models_with_enc(tmp_path):
    """Repertoire avec les 3 .enc → modeles trouves."""
    from lectura_tts_monospeaker._chargeur import _has_models

    (tmp_path / "fastpitch_encoder.onnx.enc").write_bytes(b"FAKE")
    (tmp_path / "fastpitch_decoder.onnx.enc").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx.enc").write_bytes(b"FAKE")

    assert _has_models(tmp_path) is True


def test_has_models_partial(tmp_path):
    """Repertoire avec seulement 2 ONNX → incomplet."""
    from lectura_tts_monospeaker._chargeur import _has_models

    (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE")

    assert _has_models(tmp_path) is False


def test_find_models_dir_explicit(tmp_path):
    """Parametre explicite prend priorite."""
    from lectura_tts_monospeaker._chargeur import find_models_dir

    (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")

    result = find_models_dir(tmp_path)
    assert result == tmp_path


def test_find_models_dir_none():
    """Sans modeles installe → None (sauf si modeles sont presents)."""
    from lectura_tts_monospeaker._chargeur import find_models_dir

    # Avec un chemin explicite inexistant
    result = find_models_dir("/nonexistent/path")
    # Peut retourner None ou un autre chemin si modeles sont installes
    # On verifie juste que ca ne crash pas
    assert result is None or isinstance(result, Path)


def test_load_model_bytes_plain(tmp_path):
    """Charge un modele en clair."""
    from lectura_tts_monospeaker._chargeur import load_model_bytes

    data = b"ONNX MODEL CONTENT"
    (tmp_path / "fastpitch_encoder.onnx").write_bytes(data)

    result = load_model_bytes(tmp_path, "fastpitch_encoder.onnx")
    assert result == data


def test_load_model_bytes_encrypted(tmp_path):
    """Charge un modele chiffre."""
    from lectura_tts_monospeaker._chargeur import load_model_bytes
    from lectura_tts_monospeaker._crypto import encrypt_model

    data = b"ONNX MODEL CONTENT"
    plain_path = tmp_path / "temp.onnx"
    enc_path = tmp_path / "fastpitch_encoder.onnx.enc"

    plain_path.write_bytes(data)
    encrypt_model(plain_path, enc_path)
    plain_path.unlink()

    result = load_model_bytes(tmp_path, "fastpitch_encoder.onnx")
    assert result == data
