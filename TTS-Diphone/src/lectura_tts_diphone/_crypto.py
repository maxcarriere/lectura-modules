"""Dechiffrement des modeles diphone chiffres (.enc) au runtime.

Pattern identique a lectura-g2p/_crypto.py.
Les constantes de derivation sont specifiques au module TTS diphone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Fragments de sel — specifiques TTS diphone
_P1 = "Lectura-TTS-Diphone"
_P2 = 0x4450
_P3 = "2025-WORLD-Concat"
_P4 = "Diphone-LogF16-U8AP"


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
    """Lit un fichier .enc et retourne les bytes dechiffres."""
    data = enc_path.read_bytes()
    key = _derive_key()
    key_len = len(key)
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % key_len]
    return bytes(out)


def encrypt_model(src_path: Path, enc_path: Path) -> None:
    """Chiffre un fichier en .enc (pour la preparation du package)."""
    data = src_path.read_bytes()
    key = _derive_key()
    key_len = len(key)
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % key_len]
    enc_path.write_bytes(bytes(out))
