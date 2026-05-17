"""ZeroShotEngine --- conversion vocale zero-shot via OpenVoice.

Supporte le trick SR pour decaler les formants :
  sr_override = 22050 / (2 ** variante)
  → resample la reference avant extraction SE
  → OpenVoice traite comme 22050 → formants decales.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from lectura_vc_zeroshot._chargeur import find_models_dir, has_openvoice_models
from lectura_vc_zeroshot._openvoice import OpenVoiceConverter, OV_SR

logger = logging.getLogger(__name__)


class ZeroShotEngine:
    """Moteur de conversion vocale zero-shot via OpenVoice."""

    def __init__(self, models_dir: str | Path | None = None):
        resolved = find_models_dir(models_dir)
        if resolved is None:
            raise FileNotFoundError(
                "Aucun repertoire de modeles OpenVoice trouve. "
                "Placez les modeles dans ~/.lectura/models/vc/ "
                "ou definissez LECTURA_MODELS_DIR."
            )
        if not has_openvoice_models(resolved):
            raise FileNotFoundError(
                f"Modeles OpenVoice manquants dans {resolved}. "
                "Fichiers requis: openvoice_se.onnx, openvoice_vc.onnx"
            )
        self.models_dir = resolved
        self._converter: OpenVoiceConverter | None = None

    def _ensure_loaded(self) -> OpenVoiceConverter:
        """Lazy-load le converter OpenVoice."""
        if self._converter is None:
            self._converter = OpenVoiceConverter(self.models_dir)
        return self._converter

    def extract_se(
        self,
        audio: np.ndarray | str | Path,
        sr: int | None = None,
    ) -> np.ndarray:
        """Extrait le speaker embedding d'un audio.

        Parameters
        ----------
        audio : audio de reference (array float32 ou chemin).
        sr : sample rate (auto-detecte si chemin).

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        conv = self._ensure_loaded()
        return conv.extract_se(audio, sr=sr)

    def convert_from_se(
        self,
        audio: np.ndarray | str | Path,
        src_se: np.ndarray,
        tgt_se: np.ndarray,
        sr_in: int | None = None,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Conversion avec speaker embeddings pre-calcules.

        Parameters
        ----------
        audio : audio source (array float32 ou chemin).
        src_se : speaker embedding source (1, 256, 1).
        tgt_se : speaker embedding cible (1, 256, 1).
        sr_in : sample rate source (auto si chemin).
        tau : parametre OpenVoice (0 = deterministe).

        Returns
        -------
        (audio_converti @ 22050 Hz, 22050)
        """
        conv = self._ensure_loaded()
        return conv.convert(audio, src_se, tgt_se, sr=sr_in, tau=tau)

    def convert(
        self,
        audio: np.ndarray | str | Path,
        reference: np.ndarray | str | Path,
        sr_in: int | None = None,
        ref_sr: int | None = None,
        sr_override: int | None = None,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Conversion zero-shot complete.

        Parameters
        ----------
        audio : audio source (array float32 ou chemin).
        reference : audio de reference (array float32 ou chemin).
        sr_in : sample rate source (auto si chemin).
        ref_sr : sample rate de la reference si ndarray.
        sr_override : trick SR — resample la reference vers ce SR avant
            extraction SE. OpenVoice traite comme 22050 → formants decales
            par facteur 22050/sr_override.
            Exemples: 11025 → voix aigue/enfant, 44100 → voix grave/homme.
        tau : parametre OpenVoice (0 = deterministe).

        Returns
        -------
        (audio_converti @ 22050 Hz, 22050)
        """
        conv = self._ensure_loaded()

        # Extraire SE source
        src_se = conv.extract_se(audio, sr=sr_in)

        # Extraire SE cible (avec trick SR optionnel)
        if sr_override is not None and sr_override != OV_SR:
            tgt_se = self._extract_se_with_sr_trick(reference, ref_sr, sr_override)
        else:
            tgt_se = conv.extract_se(reference, sr=ref_sr)

        return conv.convert(audio, src_se, tgt_se, sr=sr_in, tau=tau)

    def _extract_se_with_sr_trick(
        self,
        reference: np.ndarray | str | Path,
        ref_sr: int | None,
        sr_override: int,
    ) -> np.ndarray:
        """Extrait SE en trompant OpenVoice sur le SR.

        Resample la reference vers sr_override, puis passe a extract_se()
        sans declarer le SR → OpenVoice traite comme 22050 Hz.
        """
        conv = self._ensure_loaded()

        # Charger la reference a son SR natif
        if isinstance(reference, (str, Path)):
            ref_audio, actual_sr = librosa.load(str(reference), sr=None, mono=True)
        else:
            ref_audio = reference.astype(np.float32)
            actual_sr = ref_sr or OV_SR

        # Resample vers sr_override
        if actual_sr != sr_override:
            ref_audio = librosa.resample(
                ref_audio, orig_sr=actual_sr, target_sr=sr_override,
            )

        # Passer a OpenVoice SANS declarer le SR
        # → il traite comme 22050 → formants decales
        return conv.extract_se(ref_audio)
