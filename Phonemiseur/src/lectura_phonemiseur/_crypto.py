"""Déchiffrement du modèle ONNX chiffré (.enc) au runtime.

Les constantes de dérivation de clé sont identiques à encrypt_model.py
mais dispersées différemment dans le code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Fragments de sel — même matériel que encrypt_model.py
_P1 = "Lectura-NLP-G2P"
_P2 = 0x4C45
_P3 = "2025-BiLSTM-Unifie"
_P4 = "CharLevel-MultiHead"


def _derive_key() -> bytes:
    material = f"{_P1}:{_P2:#06x}:{_P3}:{_P4}"
    key = hashlib.sha256(material.encode("utf-8")).digest()
    extended = key
    for _ in range(7):
        key = hashlib.sha256(key + material.encode("utf-8")).digest()
        extended += key
    return extended[:256]


def load_encrypted_model(enc_path: Path) -> bytes:
    """Lit un fichier .enc et retourne les bytes ONNX déchiffrés."""
    data = enc_path.read_bytes()
    key = _derive_key()
    key_len = len(key)
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % key_len]
    return bytes(out)
