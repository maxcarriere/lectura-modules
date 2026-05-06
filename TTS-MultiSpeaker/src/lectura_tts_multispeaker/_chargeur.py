"""Localisateur de modeles multi-speaker — cascade de chemins.

Ordre de recherche :
1. Parametre explicite models_dir
2. Variable d'environnement LECTURA_MODELS_DIR/tts_multispeaker
3. Repertoire utilisateur ~/.lectura/models/tts_multispeaker/
4. Modeles embarques dans le package (site-packages, version privee)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_PACKAGE_MODELS = Path(__file__).parent / "modeles"

# Fichiers partages requis pour l'inference ONNX locale
SHARED_FILES = [
    "decoder.onnx",
    "hifigan.onnx",
]


def find_models_dir(
    speaker: str = "siwis",
    models_dir: str | Path | None = None,
) -> Path | None:
    """Trouve le repertoire contenant les modeles ONNX.

    Args:
        speaker: Nom du speaker dont on verifie la presence de l'encodeur.
        models_dir: Override explicite.

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
        candidates.append(Path(env_dir) / "tts_multispeaker")

    # 3. Repertoire utilisateur
    candidates.append(Path.home() / ".lectura" / "models" / "tts_multispeaker")

    # 4. Embarques dans le package
    candidates.append(_PACKAGE_MODELS)

    for candidate in candidates:
        if _has_models(candidate, speaker):
            log.debug("Modeles trouves : %s", candidate)
            return candidate

    return None


def _has_models(directory: Path, speaker: str = "siwis") -> bool:
    """Verifie que le repertoire contient les fichiers ONNX necessaires."""
    if not directory.is_dir():
        return False

    # Verifier les fichiers partages
    for filename in SHARED_FILES:
        onnx_path = directory / filename
        enc_path = directory / filename.replace(".onnx", ".enc")
        if not onnx_path.exists() and not enc_path.exists():
            return False

    # Unified layout : encoder.onnx (single file for all speakers)
    if (directory / "encoder.onnx").exists() or (directory / "encoder.enc").exists():
        return True

    # Legacy layout : encoder_{speaker}.onnx (per-speaker)
    encoder_name = f"encoder_{speaker}.onnx"
    enc_name = encoder_name.replace(".onnx", ".enc")
    if not (directory / encoder_name).exists() and not (directory / enc_name).exists():
        return False

    return True


def load_model_bytes(models_dir: Path, filename: str) -> bytes | None:
    """Charge un modele (clair ou chiffre) depuis le repertoire.

    Returns:
        bytes du modele ONNX ou None si introuvable.
    """
    onnx_path = models_dir / filename
    enc_path = models_dir / filename.replace(".onnx", ".enc")

    if onnx_path.exists():
        return onnx_path.read_bytes()
    elif enc_path.exists():
        from lectura_tts_multispeaker._crypto import load_encrypted_model
        return load_encrypted_model(enc_path)

    return None
