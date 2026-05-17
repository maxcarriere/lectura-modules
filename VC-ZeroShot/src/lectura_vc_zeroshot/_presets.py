"""Presets de speaker embeddings pre-calcules.

Chaque preset est un fichier .npy contenant un SE (1, 256, 1) float32,
extrait a partir d'un enregistrement de reference du locuteur.

Locuteurs disponibles :
  siwis, ezwa, nadine, bernard, gilles, zeckou
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PRESETS_DIR = Path(__file__).parent / "presets"

# Noms des presets livres avec le package
PRESET_SPEAKERS = ["siwis", "ezwa", "nadine", "bernard", "gilles", "zeckou"]


def list_presets() -> list[str]:
    """Retourne les noms de presets disponibles."""
    return [p.stem for p in sorted(_PRESETS_DIR.glob("*.npy"))]


def has_preset(name: str) -> bool:
    """Verifie si un preset existe."""
    return (_PRESETS_DIR / f"{name}.npy").exists()


def load_preset(name: str) -> np.ndarray:
    """Charge un preset SE par nom.

    Parameters
    ----------
    name : str
        Nom du locuteur (ex: "siwis", "bernard").

    Returns
    -------
    np.ndarray shape (1, 256, 1) float32.

    Raises
    ------
    FileNotFoundError
        Si le preset n'existe pas.
    """
    path = _PRESETS_DIR / f"{name}.npy"
    if not path.exists():
        available = list_presets()
        raise FileNotFoundError(
            f"Preset '{name}' non trouve. "
            f"Disponibles: {available}"
        )
    se = np.load(path).astype(np.float32)
    if se.ndim == 2:
        se = se[:, :, np.newaxis]
    return se


def blend_presets(specs: dict[str, float]) -> np.ndarray:
    """Melange des presets avec des poids.

    Parameters
    ----------
    specs : dict
        {nom_preset: poids}. Ex: {"siwis": 0.5, "nadine": 0.3, "ezwa": 0.2}.
        Les poids sont normalises automatiquement pour sommer a 1.

    Returns
    -------
    np.ndarray shape (1, 256, 1) float32.
    """
    if not specs:
        raise ValueError("Au moins un preset requis.")

    total_weight = sum(specs.values())
    if total_weight <= 0:
        raise ValueError("Les poids doivent etre positifs.")

    result = np.zeros((1, 256, 1), dtype=np.float32)
    for name, weight in specs.items():
        se = load_preset(name)
        result += se * (weight / total_weight)

    return result
