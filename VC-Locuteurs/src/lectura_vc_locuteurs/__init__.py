"""lectura-vc-locuteurs --- Conversion vocale RVC vers 6 voix pre-entrainees.

Sous-module (~1.4 GB) pour la conversion vers des voix entrainees :
  ezwa, nadine, bernard, gilles, zeckou, siwis.

API publique :
  creer_engine()  : fabrique un LocuteursEngine
  convertir()     : fonction de commodite (cree un engine ephemere)
  RVC_SPEAKERS    : liste des speakers disponibles
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lectura_vc_locuteurs._chargeur import RVC_SPEAKERS
from lectura_vc_locuteurs.engine import LocuteursEngine

__version__ = "1.0.0"
__all__ = ["creer_engine", "convertir", "LocuteursEngine", "RVC_SPEAKERS"]


def creer_engine(
    speaker: str | None = None,
    models_dir: str | Path | None = None,
) -> LocuteursEngine:
    """Cree un moteur de conversion vocale RVC.

    Parameters
    ----------
    speaker : voix RVC cible par defaut.
    models_dir : repertoire des modeles (defaut: auto-detection).

    Returns
    -------
    LocuteursEngine
    """
    return LocuteursEngine(speaker=speaker, models_dir=models_dir)


def convertir(
    audio: np.ndarray | str | Path,
    speaker: str,
    sr_in: int | None = None,
    models_dir: str | Path | None = None,
    **kwargs,
) -> tuple[np.ndarray, int]:
    """Convertit un audio vers une voix pre-entrainee (commodite).

    Cree un engine ephemere. Pour des conversions repetees,
    utiliser creer_engine() pour reutiliser les sessions ONNX.

    Parameters
    ----------
    audio : audio source (array float32 ou chemin fichier).
    speaker : voix RVC cible.
    sr_in : sample rate source (auto-detecte si fichier).
    models_dir : repertoire des modeles.
    **kwargs : protect, pitch_modification.

    Returns
    -------
    (audio_converti @ 48000 Hz, sample_rate)
    """
    engine = creer_engine(speaker=speaker, models_dir=models_dir)
    return engine.convert(
        audio=audio,
        speaker=speaker,
        sr_in=sr_in,
        **kwargs,
    )
