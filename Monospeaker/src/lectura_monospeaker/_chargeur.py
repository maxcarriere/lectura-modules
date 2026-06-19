"""Localisateur de modeles — cascade de chemins.

Ordre de recherche :
1. Parametre explicite models_dir
2. Variable d'environnement LECTURA_MODELS_DIR/tts_mono
3. Repertoire utilisateur ~/.lectura/models/tts_mono/
4. Modeles embarques dans le package (site-packages, version privee)

Supporte deux architectures :
- Matcha-Conformer (prioritaire) : matcha_encoder.onnx + matcha_unet.onnx + hifigan.onnx
- FastPitch-Lite (legacy fallback) : fastpitch_encoder.onnx + fastpitch_decoder.onnx + hifigan.onnx
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_PACKAGE_MODELS = Path(__file__).parent / "modeles"

# Fichiers requis par architecture
_MATCHA_FILES = [
    "matcha_encoder.onnx",
    "matcha_unet.onnx",
    "hifigan.onnx",
]

_FASTPITCH_FILES = [
    "fastpitch_encoder.onnx",
    "fastpitch_decoder.onnx",
    "hifigan.onnx",
]

# Legacy — garde la compat avec le code qui importe REQUIRED_FILES
REQUIRED_FILES = _MATCHA_FILES


def find_models_dir(models_dir: str | Path | None = None) -> Path | None:
    """Trouve le repertoire contenant les modeles ONNX.

    Detecte d'abord les fichiers Matcha, puis fallback FastPitch.

    Returns:
        Path du repertoire ou None si aucun modele trouve.
    """
    candidates: list[Path] = []

    # 1. Parametre explicite
    if models_dir is not None:
        candidates.append(Path(models_dir))

    # 2. Variable d'environnement
    env_dir = os.environ.get("LECTURA_MODELS_DIR", "")
    if env_dir:
        candidates.append(Path(env_dir) / "tts_mono")

    # 3. Repertoire utilisateur
    candidates.append(Path.home() / ".lectura" / "models" / "tts_mono")

    # 4. Embarques dans le package
    candidates.append(_PACKAGE_MODELS)

    for candidate in candidates:
        if _has_models(candidate):
            log.debug("Modeles trouves : %s", candidate)
            return candidate

    return None


def _has_models(directory: Path) -> bool:
    """Verifie que le repertoire contient les fichiers ONNX necessaires.

    Accepte Matcha OU FastPitch (les deux sont valides).
    """
    if not directory.is_dir():
        return False

    # Essayer Matcha d'abord, puis FastPitch
    for file_set in (_MATCHA_FILES, _FASTPITCH_FILES):
        all_found = True
        for filename in file_set:
            onnx_path = directory / filename
            enc_path = directory / (filename + ".enc")
            if not onnx_path.exists() and not enc_path.exists():
                all_found = False
                break
        if all_found:
            return True

    return False


def detect_model_type(models_dir: Path) -> str:
    """Detecte le type de modele dans le repertoire.

    Returns:
        "matcha" si matcha_encoder.onnx present, "fastpitch" sinon.
    """
    for filename in ("matcha_encoder.onnx", "matcha_encoder.onnx.enc"):
        if (models_dir / filename).exists():
            return "matcha"
    return "fastpitch"


def load_model_bytes(models_dir: Path, filename: str) -> bytes | None:
    """Charge un modele (clair ou chiffre) depuis le repertoire.

    Returns:
        bytes du modele ONNX ou None si introuvable.
    """
    onnx_path = models_dir / filename
    enc_path = models_dir / (filename + ".enc")

    if onnx_path.exists():
        return onnx_path.read_bytes()
    elif enc_path.exists():
        from lectura_monospeaker._crypto import load_encrypted_model
        return load_encrypted_model(enc_path)

    return None
