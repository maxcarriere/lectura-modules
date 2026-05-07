"""lectura-vc --- Module de conversion vocale Lectura.

Deux backends unifies en ONNX pur :
  - RVC : conversion vers 6 voix pre-entrainees (HuBERT + RMVPE + Synthesizer)
  - OpenVoice v2 : conversion zero-shot vers une voix arbitraire

API publique :
  creer_engine()  : fabrique un VCEngine
  convertir()     : fonction de commodite (cree un engine ephemere)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lectura_vc.engine import VCEngine

__version__ = "1.0.0"
__all__ = ["creer_engine", "convertir", "VCEngine"]


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
    reference: np.ndarray | str | Path | None = None,
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
