"""Localisateur de modeles RVC --- cascade de chemins.

Ordre de recherche :
1. Parametre explicite models_dir
2. Variable d'environnement LECTURA_MODELS_DIR/vc
3. Repertoire utilisateur ~/.lectura/models/vc/
4. Modeles embarques dans le package (site-packages)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_PACKAGE_MODELS = Path(__file__).parent / "modeles"

# Fichiers requis par RVC
RVC_REQUIRED = ["hubert.onnx", "rmvpe.onnx"]

# Speakers RVC disponibles
RVC_SPEAKERS = ["ezwa", "nadine", "bernard", "gilles", "zeckou", "siwis"]


def find_models_dir(models_dir: str | Path | None = None) -> Path | None:
    """Trouve le repertoire contenant les modeles RVC.

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
        candidates.append(Path(env_dir) / "vc")

    # 3. Repertoire utilisateur
    candidates.append(Path.home() / ".lectura" / "models" / "vc")

    # 4. Embarques dans le package
    candidates.append(_PACKAGE_MODELS)

    for candidate in candidates:
        if candidate.is_dir():
            log.debug("Repertoire modeles VC-Locuteurs candidat : %s", candidate)
            return candidate

    return None


def has_rvc_models(directory: Path, speaker: str | None = None) -> bool:
    """Verifie la presence des modeles RVC (HuBERT + RMVPE + synthesizer)."""
    if not directory.is_dir():
        return False

    for filename in RVC_REQUIRED:
        if not _file_exists(directory, filename):
            return False

    if speaker is not None:
        synth = f"synthesizer_{speaker}.onnx"
        if not _file_exists(directory, synth):
            return False

    return True


def _file_exists(directory: Path, filename: str) -> bool:
    """Verifie l'existence d'un fichier (clair ou chiffre)."""
    path = directory / filename
    enc_path = directory / (filename + ".enc")
    return path.exists() or enc_path.exists()


def load_model_bytes(models_dir: Path, filename: str) -> bytes | None:
    """Charge un modele (clair ou chiffre) depuis le repertoire.

    Returns:
        bytes du modele ou None si introuvable.
    """
    path = models_dir / filename
    enc_path = models_dir / (filename + ".enc")

    if path.exists():
        return path.read_bytes()
    elif enc_path.exists():
        from lectura_vc_locuteurs._crypto import load_encrypted_model
        return load_encrypted_model(enc_path)

    return None


def get_model_path(models_dir: Path, filename: str) -> Path | None:
    """Retourne le chemin du modele (clair ou chiffre).

    Returns:
        Path du fichier ou None si introuvable.
    """
    path = models_dir / filename
    enc_path = models_dir / (filename + ".enc")

    if path.exists():
        return path
    elif enc_path.exists():
        return enc_path

    return None
