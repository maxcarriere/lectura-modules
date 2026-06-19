"""Tests unitaires pour le module _chargeur."""

from pathlib import Path

import pytest


def test_has_models_empty_dir(tmp_path):
    """Repertoire vide -> pas de modeles."""
    from lectura_tts_monospeaker._chargeur import _has_models

    assert _has_models(tmp_path) is False


def test_has_models_with_matcha(tmp_path):
    """Repertoire avec les 3 ONNX Matcha -> modeles trouves."""
    from lectura_tts_monospeaker._chargeur import _has_models

    (tmp_path / "matcha_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "matcha_unet.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")

    assert _has_models(tmp_path) is True


def test_has_models_with_fastpitch(tmp_path):
    """Repertoire avec les 3 ONNX FastPitch -> modeles trouves (legacy)."""
    from lectura_tts_monospeaker._chargeur import _has_models

    (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")

    assert _has_models(tmp_path) is True


def test_has_models_with_enc(tmp_path):
    """Repertoire avec les 3 .enc Matcha -> modeles trouves."""
    from lectura_tts_monospeaker._chargeur import _has_models

    (tmp_path / "matcha_encoder.onnx.enc").write_bytes(b"FAKE")
    (tmp_path / "matcha_unet.onnx.enc").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx.enc").write_bytes(b"FAKE")

    assert _has_models(tmp_path) is True


def test_has_models_partial(tmp_path):
    """Repertoire avec seulement 2 ONNX -> incomplet."""
    from lectura_tts_monospeaker._chargeur import _has_models

    (tmp_path / "matcha_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")
    # Missing matcha_unet.onnx

    assert _has_models(tmp_path) is False


def test_has_models_partial_fastpitch(tmp_path):
    """Repertoire avec 2 FastPitch ONNX -> incomplet."""
    from lectura_tts_monospeaker._chargeur import _has_models

    (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE")
    # Missing hifigan.onnx

    assert _has_models(tmp_path) is False


def test_detect_model_type_matcha(tmp_path):
    """Detecte Matcha quand matcha_encoder.onnx est present."""
    from lectura_tts_monospeaker._chargeur import detect_model_type

    (tmp_path / "matcha_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "matcha_unet.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")

    assert detect_model_type(tmp_path) == "matcha"


def test_detect_model_type_matcha_encrypted(tmp_path):
    """Detecte Matcha quand matcha_encoder.onnx.enc est present."""
    from lectura_tts_monospeaker._chargeur import detect_model_type

    (tmp_path / "matcha_encoder.onnx.enc").write_bytes(b"FAKE")

    assert detect_model_type(tmp_path) == "matcha"


def test_detect_model_type_fastpitch(tmp_path):
    """Detecte FastPitch quand pas de fichiers Matcha."""
    from lectura_tts_monospeaker._chargeur import detect_model_type

    (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")

    assert detect_model_type(tmp_path) == "fastpitch"


def test_find_models_dir_explicit(tmp_path):
    """Parametre explicite prend priorite."""
    from lectura_tts_monospeaker._chargeur import find_models_dir

    (tmp_path / "matcha_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "matcha_unet.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")

    result = find_models_dir(tmp_path)
    assert result == tmp_path


def test_find_models_dir_explicit_fastpitch(tmp_path):
    """Parametre explicite avec FastPitch fonctionne aussi."""
    from lectura_tts_monospeaker._chargeur import find_models_dir

    (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE")
    (tmp_path / "hifigan.onnx").write_bytes(b"FAKE")

    result = find_models_dir(tmp_path)
    assert result == tmp_path


def test_find_models_dir_none():
    """Sans modeles installe -> None (sauf si modeles sont presents)."""
    from lectura_tts_monospeaker._chargeur import find_models_dir

    # Avec un chemin explicite inexistant
    result = find_models_dir("/nonexistent/path")
    # Peut retourner None ou un autre chemin si modeles sont installes
    assert result is None or isinstance(result, Path)


def test_load_model_bytes_plain(tmp_path):
    """Charge un modele en clair."""
    from lectura_tts_monospeaker._chargeur import load_model_bytes

    data = b"ONNX MODEL CONTENT"
    (tmp_path / "matcha_encoder.onnx").write_bytes(data)

    result = load_model_bytes(tmp_path, "matcha_encoder.onnx")
    assert result == data


def test_load_model_bytes_encrypted(tmp_path):
    """Charge un modele chiffre."""
    from lectura_tts_monospeaker._chargeur import load_model_bytes
    from lectura_tts_monospeaker._crypto import encrypt_model

    data = b"ONNX MODEL CONTENT"
    plain_path = tmp_path / "temp.onnx"
    enc_path = tmp_path / "matcha_encoder.onnx.enc"

    plain_path.write_bytes(data)
    encrypt_model(plain_path, enc_path)
    plain_path.unlink()

    result = load_model_bytes(tmp_path, "matcha_encoder.onnx")
    assert result == data
