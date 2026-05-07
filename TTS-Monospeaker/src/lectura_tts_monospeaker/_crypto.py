"""Dechiffrement des modeles ONNX chiffres (.enc) au runtime.

Pattern identique a lectura-g2p/_crypto.py.
Les constantes de derivation sont specifiques au module TTS.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Fragments de sel — specifiques TTS monospeaker
_P1 = "Lectura-TTS-Mono"
_P2 = 0x5454
_P3 = "2025-FastPitch-HiFiGAN"
_P4 = "ONNX-Split-Encoder-Decoder"


def _derive_key() -> bytes:
    """Derive une cle de 256 bytes depuis les constantes."""
    material = f"{_P1}:{_P2:#06x}:{_P3}:{_P4}"
    key = hashlib.sha256(material.encode("utf-8")).digest()
    extended = key
    for _ in range(7):
        key = hashlib.sha256(key + material.encode("utf-8")).digest()
        extended += key
    return extended[:256]


def load_encrypted_model(enc_path: Path) -> bytes:
    """Lit un fichier .enc et retourne les bytes ONNX dechiffres."""
    data = enc_path.read_bytes()
    key = _derive_key()
    key_len = len(key)
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % key_len]
    return bytes(out)


def encrypt_model(onnx_path: Path, enc_path: Path) -> None:
    """Chiffre un fichier ONNX en .enc (pour la preparation du package)."""
    data = onnx_path.read_bytes()
    key = _derive_key()
    key_len = len(key)
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % key_len]
    enc_path.write_bytes(bytes(out))
