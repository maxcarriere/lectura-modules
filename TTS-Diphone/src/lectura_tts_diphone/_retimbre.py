"""Retimbrage optionnel via OpenVoice (lectura-vc-zeroshot).

Post-traitement pour remplacer le timbre "moyen" du diphone
par un timbre coherent issu d'une reference locuteur.

Le curseur 'voix_variante' (-1 a +1) decale les formants :
  -1 = grave/masculin, 0 = neutre, +1 = aigu/enfant
Formule : sr_override = 22050 / (2 ** variante)

Requires: pip install 'lectura-tts-diphone[vc]'
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

OV_SR = 22050


def variante_to_sr_override(variante: float) -> int | None:
    """Convertit un curseur variante en sr_override pour le trick SR.

    Parameters
    ----------
    variante : float entre -1 et +1.
        0.0 = neutre (pas de trick).
        +1  = formants montes (enfant/aigu), sr_override = 11025.
        -1  = formants baisses (homme/grave), sr_override = 44100.

    Returns
    -------
    int ou None si neutre.
    """
    if abs(variante) < 0.01:
        return None
    return max(8000, min(48000, int(OV_SR / (2 ** variante))))


class RetimbreEngine:
    """Cache les sessions ONNX et les speaker embeddings."""

    def __init__(self, vc_models_dir: str | Path | None = None):
        self._engine = None       # ZeroShotEngine (lazy)
        self._vc_models_dir = vc_models_dir
        self._se_cache: dict[tuple, np.ndarray] = {}  # (ref_key, sr_override) -> SE

    def _ensure_loaded(self):
        """Lazy-load le ZeroShotEngine."""
        if self._engine is None:
            try:
                from lectura_vc_zeroshot import creer_engine
            except ImportError:
                raise ImportError(
                    "lectura-vc-zeroshot requis pour le retimbre. "
                    "Installez avec: pip install 'lectura-tts-diphone[vc]'"
                )
            self._engine = creer_engine(models_dir=self._vc_models_dir)
        return self._engine

    def retimbre(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | str | Path,
        ref_sr: int | None = None,
        variante: float = 0.0,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Applique le retimbre OpenVoice.

        Parameters
        ----------
        audio : audio source float32.
        sr : sample rate source.
        reference : audio de reference (chemin ou array).
        ref_sr : sample rate de la reference si array.
        variante : curseur -1 (grave) a +1 (aigu).
        tau : parametre OpenVoice.

        Returns
        -------
        (audio_retimbre @ OV_SR Hz, OV_SR)
        """
        engine = self._ensure_loaded()
        sr_override = variante_to_sr_override(variante)

        logger.info(
            "Retimbre: variante=%.2f, sr_override=%s, tau=%.2f",
            variante, sr_override, tau,
        )

        return engine.convert(
            audio=audio,
            reference=reference,
            sr_in=sr,
            ref_sr=ref_sr,
            sr_override=sr_override,
            tau=tau,
        )
