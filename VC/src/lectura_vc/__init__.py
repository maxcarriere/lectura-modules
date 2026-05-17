"""lectura-vc v2 --- Module de conversion vocale Lectura (meta-package).

Regroupe les deux sous-modules :
  - lectura-vc-zeroshot : OpenVoice zero-shot (~126 MB)
  - lectura-vc-locuteurs : RVC 6 voix pre-entrainees (~1.4 GB)

API publique :
  creer_engine()    : fabrique un VCEngine (facade unifiee)
  convertir()       : fonction de commodite (cree un engine ephemere)
  RVC_SPEAKERS      : liste des speakers RVC
  PRESET_SPEAKERS   : presets SE (siwis, ezwa, etc.) pour blend zero-shot
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lectura_vc.engine import VCEngine
from lectura_vc_locuteurs import RVC_SPEAKERS
from lectura_vc_zeroshot import PRESET_SPEAKERS

__version__ = "2.1.0"
__all__ = [
    "creer_engine", "convertir", "VCEngine",
    "RVC_SPEAKERS", "PRESET_SPEAKERS",
]


def creer_engine(
    mode: str = "auto",
    speaker: str | None = None,
    models_dir: Path | None = None,
) -> VCEngine:
    """Cree un moteur de conversion vocale.

    Parameters
    ----------
    mode : "rvc" | "zeroshot" | "cascade" | "auto"
        - rvc      : vers une des 6 voix pre-entrainees
        - zeroshot  : vers une voix arbitraire (5-10s de reference)
        - cascade   : RVC (proxy genre) puis OpenVoice (timbre exact)
        - auto      : choix automatique selon les parametres
    speaker : voix RVC cible (ezwa, nadine, bernard, gilles, zeckou, siwis).
    models_dir : repertoire des modeles (defaut: auto-detection).

    Returns
    -------
    VCEngine
    """
    return VCEngine(mode=mode, speaker=speaker, models_dir=models_dir)


def convertir(
    audio: np.ndarray | str | Path,
    speaker: str | None = None,
    reference: np.ndarray | str | Path | list | dict | None = None,
    mode: str = "auto",
    sr_in: int | None = None,
    models_dir: Path | None = None,
    **kwargs,
) -> tuple[np.ndarray, int]:
    """Convertit un audio vers la voix cible (fonction de commodite).

    Cree un engine ephemere. Pour des conversions repetees,
    utiliser creer_engine() pour reutiliser les sessions ONNX.

    Parameters
    ----------
    audio : audio source (array float32 ou chemin fichier).
    speaker : voix RVC cible.
    reference : audio de reference pour zero-shot (array ou chemin).
    mode : "rvc" | "zeroshot" | "cascade" | "auto".
    sr_in : sample rate source (auto-detecte si fichier).
    models_dir : repertoire des modeles.
    **kwargs : protect, pitch_modification, tau.

    Returns
    -------
    (audio_converti, sample_rate_output)
    """
    engine = creer_engine(mode=mode, speaker=speaker, models_dir=models_dir)
    return engine.convert(
        audio=audio,
        speaker=speaker,
        reference=reference,
        sr_in=sr_in,
        **kwargs,
    )
