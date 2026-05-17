"""Detection F0 moyenne et calcul d'auto-adaptation pitch.

Utilise librosa.pyin pour une detection F0 legere (sans RMVPE).
Comparaison avec les F0 moyens de speakers.json pour calculer
le pitch_modification optimal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_SPEAKERS_PATH = Path(__file__).parent / "data" / "speakers.json"
_speakers_cache: dict | None = None


def _load_speakers() -> dict:
    """Charge les metadonnees speakers."""
    global _speakers_cache
    if _speakers_cache is None:
        with open(_SPEAKERS_PATH, "r", encoding="utf-8") as f:
            _speakers_cache = json.load(f)
    return _speakers_cache


def detect_mean_f0(
    audio: np.ndarray,
    sr: int = 16000,
    fmin: float = 50.0,
    fmax: float = 600.0,
) -> float:
    """Detecte la F0 moyenne d'un signal audio.

    Utilise librosa.pyin (probabilistic YIN), rapide et fiable.

    Parameters
    ----------
    audio : 1-D float32.
    sr : sample rate.
    fmin, fmax : plage de recherche F0.

    Returns
    -------
    F0 moyenne en Hz (voix seulement, frames non-voisees exclues).
    Retourne 0.0 si aucune F0 detectee.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(np.float32),
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=2048,
    )

    if f0 is None:
        return 0.0

    voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
    if len(voiced_f0) == 0:
        return 0.0

    return float(np.median(voiced_f0))


def get_speaker_f0(speaker: str) -> float:
    """Retourne la F0 moyenne d'un speaker depuis speakers.json.

    Returns 0.0 si speaker inconnu.
    """
    speakers = _load_speakers()
    info = speakers.get(speaker)
    if info is None:
        return 0.0
    return float(info.get("mean_f0", 0.0))


def get_speaker_gender(speaker: str) -> str:
    """Retourne le genre d'un speaker ('M' ou 'F').

    Returns '' si inconnu.
    """
    speakers = _load_speakers()
    info = speakers.get(speaker)
    if info is None:
        return ""
    return str(info.get("gender", ""))


def compute_pitch_shift(
    source_f0: float,
    target_speaker: str,
) -> float:
    """Calcule le pitch_modification en demi-tons.

    Parameters
    ----------
    source_f0 : F0 moyen de la source en Hz.
    target_speaker : nom du speaker cible.

    Returns
    -------
    Shift en demi-tons (positif = monter, negatif = descendre).
    Retourne 0.0 si impossible a calculer.
    """
    target_f0 = get_speaker_f0(target_speaker)

    if source_f0 <= 0 or target_f0 <= 0:
        logger.warning(
            "Impossible de calculer le pitch shift: source_f0=%.1f, target_f0=%.1f",
            source_f0, target_f0,
        )
        return 0.0

    shift = 12 * np.log2(target_f0 / source_f0)
    logger.info(
        "Auto-pitch: source=%.0f Hz, cible=%s (%.0f Hz), shift=%.1f demi-tons",
        source_f0, target_speaker, target_f0, shift,
    )
    return float(shift)


def compute_protect(pitch_shift: float) -> float:
    """Calcule le facteur protect optimal selon le pitch shift.

    Plus le shift est grand, plus on protege les harmoniques.
    """
    if abs(pitch_shift) > 4:
        return 0.33
    return 0.5


def auto_adapt(
    audio: np.ndarray,
    sr: int,
    target_speaker: str,
) -> tuple[float, float]:
    """Calcule automatiquement pitch_modification et protect.

    Parameters
    ----------
    audio : signal source.
    sr : sample rate.
    target_speaker : nom du speaker cible.

    Returns
    -------
    (pitch_modification, protect)
    """
    source_f0 = detect_mean_f0(audio, sr=sr)
    pitch_shift = compute_pitch_shift(source_f0, target_speaker)
    protect = compute_protect(pitch_shift)
    return pitch_shift, protect
