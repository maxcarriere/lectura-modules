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
# Matcha-Conformer (v2) : matcha_unet.onnx + hifigan.onnx
# FastPitch (v1 legacy) : decoder.onnx + hifigan.onnx
SHARED_FILES_MATCHA = [
    "matcha_unet.onnx",
    "hifigan.onnx",
]
SHARED_FILES_FASTPITCH = [
    "decoder.onnx",
    "hifigan.onnx",
]


def _is_dual_layout(directory: Path, speaker: str = "siwis") -> bool:
    """Verifie si le repertoire utilise le layout dual (FastPitch + Conformer).

    Detecte la presence de dual_config.json ET des modeles dans les
    sous-repertoires fastpitch/ et conformer/.
    """
    if not directory.is_dir():
        return False

    if not (directory / "dual_config.json").exists():
        return False

    # Verifier que les sous-repertoires contiennent des modeles
    fastpitch_dir = directory / "fastpitch"
    conformer_dir = directory / "conformer"

    return (
        _has_models(fastpitch_dir, speaker)
        and _has_models(conformer_dir, speaker)
    )


def find_models_dir(
    speaker: str = "siwis",
    models_dir: str | Path | None = None,
) -> Path | None:
    """Trouve le repertoire contenant les modeles ONNX.

    Detecte trois layouts :
    - dual : dual_config.json + sous-repertoires fastpitch/ et conformer/
    - matcha-conformer (v2) : matcha_unet.onnx + matcha_encoder_{speaker}.onnx
    - fastpitch (v1 legacy) : decoder.onnx + encoder_{speaker}.onnx

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
        if _is_dual_layout(candidate, speaker) or _has_models(candidate, speaker):
            log.debug("Modeles trouves : %s", candidate)
            return candidate

    return None


def _file_exists(directory: Path, filename: str) -> bool:
    """Verifie si un fichier .onnx ou .enc existe."""
    return ((directory / filename).exists()
            or (directory / filename.replace(".onnx", ".enc")).exists())


def _has_models(directory: Path, speaker: str = "siwis") -> bool:
    """Verifie que le repertoire contient les fichiers ONNX necessaires.

    Detecte Matcha-Conformer d'abord, puis fallback FastPitch.
    """
    if not directory.is_dir():
        return False

    # --- Matcha-Conformer (v2) ---
    matcha_shared_ok = all(
        _file_exists(directory, f) for f in SHARED_FILES_MATCHA
    )
    if matcha_shared_ok:
        # Unified matcha encoder
        if _file_exists(directory, "matcha_encoder.onnx"):
            return True
        # Per-speaker matcha encoder
        if _file_exists(directory, f"matcha_encoder_{speaker}.onnx"):
            return True

    # --- FastPitch (v1 legacy) ---
    fastpitch_shared_ok = all(
        _file_exists(directory, f) for f in SHARED_FILES_FASTPITCH
    )
    if fastpitch_shared_ok:
        # Unified encoder
        if _file_exists(directory, "encoder.onnx"):
            return True
        # Per-speaker encoder
        if _file_exists(directory, f"encoder_{speaker}.onnx"):
            return True

    return False


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
        from lectura_multispeaker._crypto import load_encrypted_model
        return load_encrypted_model(enc_path)

    return None
