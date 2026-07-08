"""LocuteursEngine --- conversion vocale RVC vers voix pre-entrainees.

Supporte 6 voix : ezwa, nadine, bernard, gilles, zeckou, siwis.
Auto-adaptation du pitch et du protect selon la F0 source.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from lectura_vc_locuteurs._chargeur import (
    find_models_dir,
    has_rvc_models,
    RVC_SPEAKERS,
)

logger = logging.getLogger(__name__)


class LocuteursEngine:
    """Moteur de conversion vocale RVC vers voix pre-entrainees."""

    def __init__(
        self,
        speaker: str | None = None,
        models_dir: str | Path | None = None,
    ):
        resolved = find_models_dir(models_dir)
        if resolved is None:
            raise FileNotFoundError(
                "Aucun repertoire de modeles RVC trouve. "
                "Placez les modeles dans ~/.lectura/models/vc/ "
                "ou definissez LECTURA_MODELS_DIR."
            )
        self.models_dir = resolved
        self.default_speaker = speaker
        self._rvc: dict[str, object] = {}  # speaker -> RVCConverter (lazy)

        logger.info(
            "LocuteursEngine initialise (speaker=%s, models_dir=%s)",
            speaker, self.models_dir,
        )

    def _get_rvc(self, speaker: str):
        """Charge le backend RVC pour un speaker (lazy)."""
        if speaker not in self._rvc:
            if not has_rvc_models(self.models_dir, speaker):
                raise FileNotFoundError(
                    f"Modeles RVC manquants pour '{speaker}' dans {self.models_dir}"
                )
            from lectura_vc_locuteurs._rvc import RVCConverter
            self._rvc[speaker] = RVCConverter(self.models_dir, speaker)
        return self._rvc[speaker]

    def convert(
        self,
        audio: np.ndarray | str | Path,
        speaker: str | None = None,
        sr_in: int | None = None,
        protect: float | None = None,
        pitch_modification: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Convertit l'audio vers une voix pre-entrainee.

        Parameters
        ----------
        audio : audio source (array float32 ou chemin fichier).
        speaker : voix RVC cible (ezwa, bernard, etc.).
        sr_in : sample rate source (auto si fichier).
        protect : facteur de protection voix (None = auto).
        pitch_modification : shift en demi-tons (None = auto).

        Returns
        -------
        (audio_converti @ 48000 Hz, 48000)
        """
        speaker = speaker or self.default_speaker
        if speaker is None:
            raise ValueError("'speaker' requis pour la conversion RVC.")
        if speaker not in RVC_SPEAKERS:
            raise ValueError(
                f"Speaker '{speaker}' inconnu. Disponibles: {RVC_SPEAKERS}"
            )

        # Charger audio
        audio_np, sr = self._load_audio(audio, sr_in)

        logger.info(
            "Conversion RVC: speaker=%s, duree=%.1fs",
            speaker, len(audio_np) / sr,
        )

        rvc = self._get_rvc(speaker)

        # Auto-adaptation pitch si non specifie
        if pitch_modification is None or protect is None:
            from lectura_vc_locuteurs._pitch_detect import auto_adapt
            auto_pitch, auto_protect = auto_adapt(audio_np, sr, speaker)
            if pitch_modification is None:
                pitch_modification = auto_pitch
            if protect is None:
                protect = auto_protect

        return rvc.convert(audio_np, sr=sr, protect=protect, pitch_modification=pitch_modification)

    @property
    def available_speakers(self) -> list[str]:
        """Liste des speakers RVC disponibles dans models_dir."""
        if self.models_dir is None:
            return []
        return [
            s for s in RVC_SPEAKERS
            if has_rvc_models(self.models_dir, s)
        ]

    @staticmethod
    def _load_audio(
        audio: np.ndarray | str | Path,
        sr: int | None,
    ) -> tuple[np.ndarray, int]:
        """Charge l'audio depuis un array ou un fichier."""
        if isinstance(audio, (str, Path)):
            audio_np, detected_sr = librosa.load(str(audio), sr=None, mono=True)
            return audio_np.astype(np.float32), detected_sr
        else:
            if sr is None:
                raise ValueError(
                    "sr_in requis quand l'audio est un np.ndarray."
                )
            return audio.astype(np.float32), sr
