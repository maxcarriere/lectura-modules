"""VCEngine --- orchestrateur de conversion vocale.

Supporte 4 modes :
  - rvc      : conversion vers une des 6 voix pre-entrainees (ONNX pur)
  - zeroshot : conversion vers une voix arbitraire via OpenVoice (ONNX)
  - cascade  : RVC (proxy genre) puis OpenVoice (timbre exact)
  - auto     : choix automatique selon les parametres fournis
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from lectura_vc._chargeur import (
    find_models_dir,
    has_openvoice_models,
    has_rvc_models,
    RVC_SPEAKERS,
)

logger = logging.getLogger(__name__)


class VCEngine:
    """Moteur de conversion vocale unifie."""

    def __init__(
        self,
        mode: str = "auto",
        speaker: str | None = None,
        models_dir: Path | None = None,
    ):
        self.mode = mode
        self.default_speaker = speaker
        self.models_dir = find_models_dir(models_dir)

        if self.models_dir is None:
            raise FileNotFoundError(
                "Aucun repertoire de modeles VC trouve. "
                "Placez les modeles dans ~/.lectura/models/vc/ "
                "ou definissez LECTURA_MODELS_DIR."
            )

        self._rvc: dict[str, object] = {}  # speaker -> RVCConverter (lazy)
        self._openvoice: object | None = None  # OpenVoiceConverter (lazy)

        logger.info("VCEngine initialise (mode=%s, models_dir=%s)", mode, self.models_dir)

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _get_rvc(self, speaker: str):
        """Charge le backend RVC pour un speaker (lazy)."""
        if speaker not in self._rvc:
            if not has_rvc_models(self.models_dir, speaker):
                raise FileNotFoundError(
                    f"Modeles RVC manquants pour '{speaker}' dans {self.models_dir}"
                )
            from lectura_vc._rvc import RVCConverter
            self._rvc[speaker] = RVCConverter(self.models_dir, speaker)
        return self._rvc[speaker]

    def _get_openvoice(self):
        """Charge le backend OpenVoice (lazy)."""
        if self._openvoice is None:
            if not has_openvoice_models(self.models_dir):
                raise FileNotFoundError(
                    f"Modeles OpenVoice manquants dans {self.models_dir}"
                )
            from lectura_vc._openvoice import OpenVoiceConverter
            self._openvoice = OpenVoiceConverter(self.models_dir)
        return self._openvoice

    # ── Mode resolution ───────────────────────────────────────────────────

    def _resolve_mode(
        self,
        speaker: str | None,
        reference: np.ndarray | str | Path | None,
        mode: str | None,
    ) -> str:
        """Determine le mode effectif."""
        effective = mode or self.mode

        if effective == "auto":
            if speaker and reference:
                effective = "cascade"
            elif reference:
                effective = "zeroshot"
            elif speaker:
                effective = "rvc"
            else:
                raise ValueError(
                    "Mode 'auto' necessite au moins 'speaker' ou 'reference'."
                )

        return effective

    # ── Public API ────────────────────────────────────────────────────────

    def convert(
        self,
        audio: np.ndarray | str | Path,
        speaker: str | None = None,
        reference: np.ndarray | str | Path | None = None,
        mode: str | None = None,
        sr_in: int | None = None,
        protect: float | None = None,
        pitch_modification: float | None = None,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Convertit l'audio vers la voix cible.

        Parameters
        ----------
        audio : audio source (array float32 ou chemin fichier).
        speaker : voix RVC cible (ezwa, bernard, etc.).
        reference : audio de reference pour zero-shot (array ou chemin).
        mode : forcer un mode (rvc/zeroshot/cascade/auto).
        sr_in : sample rate source (auto si fichier).
        protect : facteur de protection voix (None = auto).
        pitch_modification : shift en demi-tons (None = auto).
        tau : parametre OpenVoice (0 = deterministe).

        Returns
        -------
        (audio_converti, sample_rate)
        """
        speaker = speaker or self.default_speaker
        effective_mode = self._resolve_mode(speaker, reference, mode)

        # Load audio
        audio_np, sr = self._load_audio(audio, sr_in)

        logger.info(
            "Conversion: mode=%s, speaker=%s, duree=%.1fs",
            effective_mode, speaker, len(audio_np) / sr,
        )

        if effective_mode == "rvc":
            return self._convert_rvc(audio_np, sr, speaker, protect, pitch_modification)
        elif effective_mode == "zeroshot":
            return self._convert_zeroshot(audio_np, sr, reference, tau)
        elif effective_mode == "cascade":
            return self._convert_cascade(
                audio_np, sr, speaker, reference, protect, pitch_modification, tau
            )
        else:
            raise ValueError(f"Mode inconnu: {effective_mode}")

    # ── Mode implementations ─────────────────────────────────────────────

    def _convert_rvc(
        self,
        audio: np.ndarray,
        sr: int,
        speaker: str | None,
        protect: float | None,
        pitch_modification: float | None,
    ) -> tuple[np.ndarray, int]:
        """Conversion RVC directe."""
        if speaker is None:
            raise ValueError("Mode 'rvc' necessite un 'speaker'.")
        if speaker not in RVC_SPEAKERS:
            raise ValueError(
                f"Speaker '{speaker}' inconnu. Disponibles: {RVC_SPEAKERS}"
            )

        rvc = self._get_rvc(speaker)

        # Auto-adaptation pitch si non specifie
        if pitch_modification is None or protect is None:
            from lectura_vc._pitch_detect import auto_adapt
            auto_pitch, auto_protect = auto_adapt(audio, sr, speaker)
            if pitch_modification is None:
                pitch_modification = auto_pitch
            if protect is None:
                protect = auto_protect

        return rvc.convert(audio, sr=sr, protect=protect, pitch_modification=pitch_modification)

    def _convert_zeroshot(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | str | Path | None,
        tau: float,
    ) -> tuple[np.ndarray, int]:
        """Conversion zero-shot via OpenVoice."""
        if reference is None:
            raise ValueError("Mode 'zeroshot' necessite une 'reference'.")

        ov = self._get_openvoice()

        # Extraire SE de la source et de la reference
        src_se = ov.extract_se(audio, sr=sr)
        tgt_se = self._resolve_reference_se(reference)

        return ov.convert(audio, src_se, tgt_se, sr=sr, tau=tau)

    def _convert_cascade(
        self,
        audio: np.ndarray,
        sr: int,
        speaker: str | None,
        reference: np.ndarray | str | Path | None,
        protect: float | None,
        pitch_modification: float | None,
        tau: float,
    ) -> tuple[np.ndarray, int]:
        """Cascade: RVC (proxy) puis OpenVoice (timbre)."""
        if speaker is None:
            raise ValueError("Mode 'cascade' necessite un 'speaker'.")
        if reference is None:
            raise ValueError("Mode 'cascade' necessite une 'reference'.")

        # Phase 1: RVC vers le speaker proxy
        rvc_audio, rvc_sr = self._convert_rvc(
            audio, sr, speaker, protect, pitch_modification,
        )

        # Phase 2: OpenVoice pour ajuster le timbre
        ov = self._get_openvoice()
        src_se = ov.extract_se(rvc_audio, sr=rvc_sr)
        tgt_se = self._resolve_reference_se(reference)

        return ov.convert(rvc_audio, src_se, tgt_se, sr=rvc_sr, tau=tau)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _resolve_reference_se(
        self,
        reference: np.ndarray | str | Path,
    ) -> np.ndarray:
        """Extrait le speaker embedding de la reference."""
        ov = self._get_openvoice()

        if isinstance(reference, (list, tuple)):
            return ov.extract_se_multi(reference)
        return ov.extract_se(reference)

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

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def available_speakers(self) -> list[str]:
        """Liste des speakers RVC disponibles dans models_dir."""
        if self.models_dir is None:
            return []
        return [
            s for s in RVC_SPEAKERS
            if has_rvc_models(self.models_dir, s)
        ]

    @property
    def has_openvoice(self) -> bool:
        """True si les modeles OpenVoice sont disponibles."""
        if self.models_dir is None:
            return False
        return has_openvoice_models(self.models_dir)
