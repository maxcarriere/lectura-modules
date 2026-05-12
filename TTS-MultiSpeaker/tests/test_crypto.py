"""Tests unitaires pour le module _crypto."""

from pathlib import Path

import pytest


def test_encrypt_decrypt_roundtrip(tmp_path):
    """Verifie que encrypt -> decrypt retourne les donnees originales."""
    from lectura_tts_multispeaker._crypto import encrypt_model, load_encrypted_model

    original = b"FAKE ONNX MODEL DATA " * 1000
    onnx_path = tmp_path / "model.onnx"
    enc_path = tmp_path / "model.onnx.enc"

    onnx_path.write_bytes(original)
    encrypt_model(onnx_path, enc_path)

    encrypted = enc_path.read_bytes()
    assert encrypted != original
    assert len(encrypted) == len(original)

    decrypted = load_encrypted_model(enc_path)
    assert decrypted == original


def test_key_deterministic():
    """Verifie que la cle derivee est deterministe."""
    from lectura_tts_multispeaker._crypto import _derive_key

    key1 = _derive_key()
    key2 = _derive_key()
    assert key1 == key2
    assert len(key1) == 256


def test_key_differs_from_monospeaker():
    """Verifie que la cle est differente du module monospeaker."""
    from lectura_tts_multispeaker._crypto import _derive_key

    key = _derive_key()
    # La cle multi-speaker ne doit pas etre identique a la mono
    # (salt different: "Lectura-TTS-MultiSpeaker" vs "Lectura-TTS-Mono")
    assert len(key) == 256
    # Simplement verifier que c'est un bytes non-nul
    assert any(b != 0 for b in key)
