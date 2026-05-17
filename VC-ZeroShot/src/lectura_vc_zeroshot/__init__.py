"""lectura-vc-zeroshot --- Conversion vocale zero-shot via OpenVoice (ONNX).

Sous-module leger (~126 MB) pour la conversion zero-shot.
Supporte le trick SR pour decaler les formants (voix homme/enfant).

API publique :
  creer_engine()  : fabrique un ZeroShotEngine
  convertir()     : fonction de commodite (cree un engine ephemere)
  OV_SR           : sample rate de sortie (22050 Hz)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lectura_vc_zeroshot._openvoice import OV_SR
from lectura_vc_zeroshot.engine import ZeroShotEngine

__version__ = "1.0.0"
__all__ = ["creer_engine", "convertir", "ZeroShotEngine", "OV_SR"]


def creer_engine(models_dir: str | Path | None = None) -> ZeroShotEngine:
    """Cree un moteur de conversion vocale zero-shot.

    Parameters
    ----------
    models_dir : repertoire des modeles (defaut: auto-detection).

    Returns
    -------
    ZeroShotEngine
    """
    return ZeroShotEngine(models_dir=models_dir)


def convertir(
    audio: np.ndarray | str | Path,
    reference: np.ndarray | str | Path,
    sr_in: int | None = None,
    ref_sr: int | None = None,
    sr_override: int | None = None,
    tau: float = 0.3,
) -> tuple[np.ndarray, int]:
    """Convertit un audio vers le timbre d'une reference (commodite).

    Cree un engine ephemere. Pour des conversions repetees,
    utiliser creer_engine() pour reutiliser les sessions ONNX.

    Parameters
    ----------
    audio : audio source (array float32 ou chemin fichier).
    reference : audio de reference (array ou chemin).
    sr_in : sample rate source (auto-detecte si fichier).
    ref_sr : sample rate de la reference si ndarray.
    sr_override : trick SR pour decaler les formants.
        11025 → voix aigue/enfant, 44100 → voix grave/homme.
    tau : parametre OpenVoice (0 = deterministe).

    Returns
    -------
    (audio_converti @ 22050 Hz, sample_rate)
    """
    engine = creer_engine()
    return engine.convert(
        audio=audio,
        reference=reference,
        sr_in=sr_in,
        ref_sr=ref_sr,
        sr_override=sr_override,
        tau=tau,
    )
