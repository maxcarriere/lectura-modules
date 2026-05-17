"""VCEngine v2 --- facade unifiee delegant aux sous-moteurs.

Supporte 4 modes :
  - rvc      : conversion vers une des 6 voix pre-entrainees (via lectura-vc-locuteurs)
  - zeroshot : conversion vers une voix arbitraire via OpenVoice (via lectura-vc-zeroshot)
  - cascade  : RVC (proxy genre) puis OpenVoice (timbre exact)
  - auto     : choix automatique selon les parametres fournis
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lectura_vc_locuteurs import RVC_SPEAKERS
from lectura_vc_locuteurs.engine import LocuteursEngine
from lectura_vc_zeroshot.engine import ZeroShotEngine

logger = logging.getLogger(__name__)


class VCEngine:
    """Moteur de conversion vocale unifie — delegue aux sous-moteurs."""

    def __init__(
        self,
        mode: str = "auto",
        speaker: str | None = None,
        models_dir: Path | None = None,
    ):
        self.mode = mode
        self.default_speaker = speaker
        self._models_dir = str(models_dir) if models_dir else None

        # Sous-moteurs (lazy)
        self._locuteurs_engine: LocuteursEngine | None = None
        self._zeroshot_engine: ZeroShotEngine | None = None

        logger.info("VCEngine v2 initialise (mode=%s)", mode)

    def _get_locuteurs(self) -> LocuteursEngine:
        """Charge le sous-moteur RVC (lazy)."""
        if self._locuteurs_engine is None:
            self._locuteurs_engine = LocuteursEngine(
                speaker=self.default_speaker,
                models_dir=self._models_dir,
            )
        return self._locuteurs_engine

    def _get_zeroshot(self) -> ZeroShotEngine:
        """Charge le sous-moteur OpenVoice (lazy)."""
        if self._zeroshot_engine is None:
            self._zeroshot_engine = ZeroShotEngine(
                models_dir=self._models_dir,
            )
        return self._zeroshot_engine

    # -- Mode resolution ---------------------------------------------------------

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

    # -- Public API --------------------------------------------------------------

    def convert(
        self,
        audio: np.ndarray | str | Path,
        speaker: str | None = None,
        reference: np.ndarray | str | Path | None = None,
        mode: str | None = None,
        sr_in: int | None = None,
        ref_sr: int | None = None,
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
        ref_sr : sample rate de la reference si c'est un ndarray.
        protect : facteur de protection voix (None = auto).
        pitch_modification : shift en demi-tons (None = auto).
        tau : parametre OpenVoice (0 = deterministe).

        Returns
        -------
        (audio_converti, sample_rate)
        """
        speaker = speaker or self.default_speaker
        effective_mode = self._resolve_mode(speaker, reference, mode)

        logger.info("Conversion: mode=%s, speaker=%s", effective_mode, speaker)

        if effective_mode == "rvc":
            return self._get_locuteurs().convert(
                audio, speaker=speaker, sr_in=sr_in,
                protect=protect, pitch_modification=pitch_modification,
            )
        elif effective_mode == "zeroshot":
            if reference is None:
                raise ValueError("Mode 'zeroshot' necessite une 'reference'.")
            return self._get_zeroshot().convert(
                audio, reference, sr_in=sr_in, ref_sr=ref_sr, tau=tau,
            )
        elif effective_mode == "cascade":
            if speaker is None:
                raise ValueError("Mode 'cascade' necessite un 'speaker'.")
            if reference is None:
                raise ValueError("Mode 'cascade' necessite une 'reference'.")
            # Phase 1: RVC vers le speaker proxy
            rvc_audio, rvc_sr = self._get_locuteurs().convert(
                audio, speaker=speaker, sr_in=sr_in,
                protect=protect, pitch_modification=pitch_modification,
            )
            # Phase 2: OpenVoice pour ajuster le timbre
            return self._get_zeroshot().convert(
                rvc_audio, reference, sr_in=rvc_sr, ref_sr=ref_sr, tau=tau,
            )
        else:
            raise ValueError(f"Mode inconnu: {effective_mode}")

    # -- Info --------------------------------------------------------------------

    @property
    def available_speakers(self) -> list[str]:
        """Liste des speakers RVC disponibles dans models_dir."""
        try:
            return self._get_locuteurs().available_speakers
        except FileNotFoundError:
            return []

    @property
    def has_openvoice(self) -> bool:
        """True si les modeles OpenVoice sont disponibles."""
        try:
            self._get_zeroshot()
            return True
        except FileNotFoundError:
            return False

    @property
    def models_dir(self) -> Path | None:
        """Repertoire des modeles (pour retro-compatibilite)."""
        loc = self._locuteurs_engine
        if loc is not None:
            return loc.models_dir
        zs = self._zeroshot_engine
        if zs is not None:
            return zs.models_dir
        # Tenter de resoudre via locuteurs
        try:
            return self._get_locuteurs().models_dir
        except FileNotFoundError:
            try:
                return self._get_zeroshot().models_dir
            except FileNotFoundError:
                return None
