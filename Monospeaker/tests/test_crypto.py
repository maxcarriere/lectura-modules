"""Tests unitaires pour le module _crypto."""

from pathlib import Path

import pytest


def test_encrypt_decrypt_roundtrip(tmp_path):
    """Verifie que encrypt → decrypt retourne les donnees originales."""
    from lectura_tts_monospeaker._crypto import encrypt_model, load_encrypted_model

    # Donnees originales
    original = b"FAKE ONNX MODEL DATA " * 1000
    onnx_path = tmp_path / "model.onnx"
    enc_path = tmp_path / "model.onnx.enc"

    onnx_path.write_bytes(original)
    encrypt_model(onnx_path, enc_path)

    # Verifier que le fichier chiffre est different
    encrypted = enc_path.read_bytes()
    assert encrypted != original
    assert len(encrypted) == len(original)

    # Dechiffrer
    decrypted = load_encrypted_model(enc_path)
    assert decrypted == original


def test_key_deterministic():
    """Verifie que la cle derivee est deterministe."""
    from lectura_tts_monospeaker._crypto import _derive_key

    key1 = _derive_key()
    key2 = _derive_key()
    assert key1 == key2
    assert len(key1) == 256
