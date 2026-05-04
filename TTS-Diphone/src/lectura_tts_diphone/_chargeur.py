"""Localisateur de modeles — cascade de chemins.

Ordre de recherche :
1. Parametre explicite models_dir
2. Variable d'environnement LECTURA_MODELS_DIR/tts_diphone
3. Repertoire utilisateur ~/.lectura/models/tts_diphone/
4. Modeles embarques dans le package (site-packages)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_PACKAGE_MODELS = Path(__file__).parent / "modeles"

# Fichier requis pour l'inference locale
REQUIRED_FILES = [
    "diphones.dpk.gz",
]

# Fichier optionnel (statistiques prosodiques corpus)
OPTIONAL_FILES = [
    "diphone_statistics.pkl",
]


def find_models_dir(models_dir: str | Path | None = None) -> Path | None:
    """Trouve le repertoire contenant les modeles diphone.

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
        candidates.append(Path(env_dir) / "tts_diphone")

    # 3. Repertoire utilisateur
    candidates.append(Path.home() / ".lectura" / "models" / "tts_diphone")

    # 4. Embarques dans le package
    candidates.append(_PACKAGE_MODELS)

    for candidate in candidates:
        if _has_models(candidate):
            log.debug("Modeles trouves : %s", candidate)
            return candidate

    return None


def _has_models(directory: Path) -> bool:
    """Verifie que le repertoire contient les fichiers necessaires."""
    if not directory.is_dir():
        return False

    for filename in REQUIRED_FILES:
        plain = directory / filename
        enc = directory / (filename + ".enc")
        if not plain.exists() and not enc.exists():
            # Fallback : pkl brut (dev / non compresse)
            if not (directory / "diphone_averaged.pkl").exists():
                return False

    return True


def load_model_bytes(models_dir: Path, filename: str) -> bytes | None:
    """Charge un modele (clair ou chiffre) depuis le repertoire.

    Returns:
        bytes du fichier ou None si introuvable.
    """
    plain = models_dir / filename
    enc = models_dir / (filename + ".enc")

    if plain.exists():
        return plain.read_bytes()
    elif enc.exists():
        from lectura_tts_diphone._crypto import load_encrypted_model
        return load_encrypted_model(enc)

    return None
