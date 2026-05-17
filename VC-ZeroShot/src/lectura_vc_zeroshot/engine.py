"""ZeroShotEngine --- conversion vocale zero-shot via OpenVoice.

Supporte :
  - Conversion avec une ou plusieurs references audio
  - Presets de locuteurs pre-calcules (SE en .npy)
  - Blend pondere de voix (presets ou audio)
  - Trick SR pour decaler les formants :
      sr_override = 22050 / (2 ** variante)
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from lectura_vc_zeroshot._chargeur import find_models_dir, has_openvoice_models
from lectura_vc_zeroshot._openvoice import OpenVoiceConverter, OV_SR
from lectura_vc_zeroshot._presets import (
    has_preset,
    list_presets,
    load_preset,
    blend_presets,
)

logger = logging.getLogger(__name__)

# Type alias pour les references polymorphes
_RefSpec = (
    str                              # chemin audio ou nom de preset
    | Path                           # chemin audio
    | np.ndarray                     # audio array ou SE (1,256,1)
    | list                           # liste de references (poids egaux)
    | dict                           # {ref: poids} pour blend pondere
)


def _is_se(arr: np.ndarray) -> bool:
    """Detecte si un ndarray est un SE (1, 256, 1) vs audio (N,)."""
    return arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 256


def blend_se(
    se_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Melange des speaker embeddings avec des poids.

    Parameters
    ----------
    se_list : list[ndarray]
        Liste de SE (1, 256, 1).
    weights : list[float] | None
        Poids pour chaque SE. None = poids egaux.
        Normalises automatiquement pour sommer a 1.

    Returns
    -------
    np.ndarray shape (1, 256, 1) float32.
    """
    if not se_list:
        raise ValueError("Au moins un SE requis.")
    if len(se_list) == 1:
        return se_list[0].astype(np.float32)

    if weights is None:
        weights = [1.0] * len(se_list)
    if len(weights) != len(se_list):
        raise ValueError(
            f"Nombre de poids ({len(weights)}) != nombre de SE ({len(se_list)})"
        )

    total = sum(weights)
    if total <= 0:
        raise ValueError("Les poids doivent etre positifs.")

    result = np.zeros((1, 256, 1), dtype=np.float32)
    for se, w in zip(se_list, weights):
        result += se.astype(np.float32).reshape(1, 256, 1) * (w / total)
    return result


class ZeroShotEngine:
    """Moteur de conversion vocale zero-shot via OpenVoice.

    Supporte les presets de voix, le blend pondere, et le trick SR
    pour decaler les formants.
    """

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

    # -- Extraction SE ---------------------------------------------------------

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

    def extract_se_multi(
        self,
        audios: list[np.ndarray | str | Path],
        sr: int | None = None,
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """Extrait et melange les SE de plusieurs audios.

        Parameters
        ----------
        audios : liste d'audios (arrays ou chemins).
        sr : sample rate commun (auto si chemins).
        weights : poids pour chaque audio. None = poids egaux.

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        ses = [self.extract_se(a, sr=sr) for a in audios]
        return blend_se(ses, weights)

    # -- Presets ---------------------------------------------------------------

    @staticmethod
    def get_preset_se(name: str) -> np.ndarray:
        """Charge un speaker embedding pre-calcule par nom.

        Parameters
        ----------
        name : nom du locuteur (ex: "siwis", "bernard").

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        return load_preset(name)

    @staticmethod
    def available_presets() -> list[str]:
        """Liste les presets disponibles."""
        return list_presets()

    @staticmethod
    def blend_preset_se(specs: dict[str, float]) -> np.ndarray:
        """Melange des presets avec des poids.

        Parameters
        ----------
        specs : {nom_preset: poids}.
            Ex: {"siwis": 0.5, "nadine": 0.3, "ezwa": 0.2}

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        return blend_presets(specs)

    # -- Resolution de reference polymorphe ------------------------------------

    def resolve_target_se(
        self,
        reference: _RefSpec,
        ref_sr: int | None = None,
        sr_override: int | None = None,
    ) -> np.ndarray:
        """Resout une reference polymorphe en speaker embedding cible.

        Parameters
        ----------
        reference : specification de la voix cible.
            - str : nom de preset ("siwis") ou chemin fichier audio.
            - Path : chemin fichier audio.
            - ndarray (1,256,1) : SE pre-calcule directement.
            - ndarray (N,) : audio array (necessite ref_sr).
            - list[str|Path|ndarray] : plusieurs references (poids egaux).
            - dict[str, float] : blend pondere de presets et/ou chemins.
        ref_sr : sample rate si reference est un ndarray audio.
        sr_override : trick SR pour decaler les formants.

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        # -- dict : blend pondere --
        if isinstance(reference, dict):
            return self._resolve_dict_reference(reference, ref_sr, sr_override)

        # -- list : plusieurs references (poids egaux) --
        if isinstance(reference, list):
            ses = [
                self._resolve_single_to_se(ref, ref_sr, sr_override)
                for ref in reference
            ]
            return blend_se(ses)

        # -- scalaire --
        return self._resolve_single_to_se(reference, ref_sr, sr_override)

    def _resolve_single_to_se(
        self,
        reference: np.ndarray | str | Path,
        ref_sr: int | None,
        sr_override: int | None,
    ) -> np.ndarray:
        """Resout une reference unique en SE."""
        # ndarray : SE directe ou audio
        if isinstance(reference, np.ndarray):
            if _is_se(reference):
                return reference.astype(np.float32)
            # audio array
            if sr_override is not None and sr_override != OV_SR:
                return self._extract_se_with_sr_trick(reference, ref_sr, sr_override)
            return self.extract_se(reference, sr=ref_sr)

        # str : preset ou chemin
        ref_str = str(reference)
        if has_preset(ref_str):
            return load_preset(ref_str)

        # Chemin fichier
        if sr_override is not None and sr_override != OV_SR:
            return self._extract_se_with_sr_trick(reference, ref_sr, sr_override)
        return self.extract_se(reference, sr=ref_sr)

    def _resolve_dict_reference(
        self,
        specs: dict[str, float],
        ref_sr: int | None,
        sr_override: int | None,
    ) -> np.ndarray:
        """Resout un dict {ref: poids} en SE blende."""
        ses = []
        weights = []
        for ref, weight in specs.items():
            se = self._resolve_single_to_se(ref, ref_sr, sr_override)
            ses.append(se)
            weights.append(weight)
        return blend_se(ses, weights)

    # -- Conversion ------------------------------------------------------------

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
        reference: _RefSpec,
        sr_in: int | None = None,
        ref_sr: int | None = None,
        sr_override: int | None = None,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Conversion zero-shot complete.

        Parameters
        ----------
        audio : audio source (array float32 ou chemin).
        reference : voix cible — polymorphe :
            - str : nom de preset ("siwis") ou chemin fichier audio.
            - Path : chemin fichier audio.
            - ndarray (1,256,1) : SE pre-calcule.
            - ndarray (N,) : audio array (necessite ref_sr).
            - list : plusieurs references (poids egaux).
            - dict[str, float] : blend pondere.
        sr_in : sample rate source (auto si chemin).
        ref_sr : sample rate de la reference si ndarray audio.
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

        # Resoudre SE cible (polymorphe)
        tgt_se = self.resolve_target_se(reference, ref_sr, sr_override)

        return conv.convert(audio, src_se, tgt_se, sr=sr_in, tau=tau)

    # -- Trick SR --------------------------------------------------------------

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
