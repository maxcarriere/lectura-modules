"""lectura-tts-multispeaker — Synthese vocale neuronale multi-speaker francais.

Exports publics :
    - creer_engine(mode, speaker, models_dir, ...) -> engine
    - synthetiser(texte, speaker, **kwargs) -> numpy array float32
    - liste_speakers() -> list[dict]
    - OnnxTTSEngine, ApiTTSEngine
    - TTSResult, PhonemeTiming
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__version__ = "1.1.0"

_SPEAKERS_DATA: list[dict] | None = None
_DEFAULT_SPEAKER: str | None = None


def _load_speakers() -> tuple[list[dict], str]:
    """Charge le fichier speakers.json (singleton)."""
    global _SPEAKERS_DATA, _DEFAULT_SPEAKER
    if _SPEAKERS_DATA is not None:
        return _SPEAKERS_DATA, _DEFAULT_SPEAKER

    speakers_path = Path(__file__).parent / "data" / "speakers.json"
    with open(speakers_path, encoding="utf-8") as f:
        data = json.load(f)
    _SPEAKERS_DATA = data["speakers"]
    _DEFAULT_SPEAKER = data.get("default", "siwis")
    return _SPEAKERS_DATA, _DEFAULT_SPEAKER


def liste_speakers() -> list[dict]:
    """Retourne la liste des speakers disponibles avec metadata.

    Returns
    -------
    list[dict]
        Chaque dict contient : id, name, gender, label
    """
    speakers, _ = _load_speakers()
    return list(speakers)


def creer_engine(
    mode: str = "auto",
    speaker: str = "siwis",
    models_dir: str | Path | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
):
    """Factory pour creer un engine d'inference TTS multi-speaker.

    Parameters
    ----------
    mode : str
        "auto" : ONNX local si disponible, sinon API
        "local" : force l'inference ONNX locale
        "api" : force l'API distante
    speaker : str
        Nom du speaker (siwis, ezwa, nadine, bernard, gilles, zeckou)
    models_dir : Path | None
        Repertoire des modeles ONNX (override la detection auto)
    api_url : str | None
        URL du serveur API
    api_key : str | None
        Cle API

    Returns
    -------
    OnnxTTSEngine | ApiTTSEngine
        Engine avec interface unifiee (synthesize, synthesize_phonemes)
    """
    if mode in ("auto", "local"):
        engine = _try_local(speaker, models_dir)
        if engine is not None:
            return engine
        if mode == "local":
            raise FileNotFoundError(
                "Modeles ONNX introuvables. Verifiez l'installation ou "
                "specifiez models_dir. Voir README pour les emplacements."
            )
        log.info("Modeles locaux non disponibles, fallback vers API")

    from lectura_tts_multispeaker.inference_api import ApiTTSEngine
    return ApiTTSEngine(api_url=api_url, api_key=api_key, speaker=speaker)


def _try_local(speaker: str, models_dir: str | Path | None = None):
    """Tente de creer un engine ONNX local."""
    try:
        import onnxruntime  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return None

    from lectura_tts_multispeaker._chargeur import find_models_dir

    resolved = find_models_dir(speaker, models_dir)
    if resolved is None:
        return None

    from lectura_tts_multispeaker.inference_onnx import OnnxTTSEngine
    return OnnxTTSEngine(resolved, speaker=speaker)


def synthetiser(
    texte: str,
    speaker: str = "siwis",
    mode: str = "auto",
    models_dir: str | Path | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    phrase_type: int | None = None,
    duration_scale: float = 1.0,
    pitch_shift: float = 0.0,
    pitch_range: float = 1.3,
    energy_scale: float = 1.0,
    pause_scale: float = 1.0,
    style: str | None = None,
    style_vector: list[float] | None = None,
) -> Any:
    """Convenience : texte -> numpy audio float32.

    Parameters
    ----------
    texte : str
        Texte francais a synthetiser
    speaker : str
        Nom du speaker (siwis, ezwa, nadine, bernard, gilles, zeckou)
    mode, models_dir, api_url, api_key :
        Parametres de creer_engine()
    phrase_type, duration_scale, pitch_shift, pitch_range, energy_scale, pause_scale :
        Controles prosodiques
    style : str | None
        Nom d'un preset de style (ex: "expressive", "calm")
    style_vector : list[float] | None
        Vecteur de style explicite [n_style_dims]

    Returns
    -------
    numpy.ndarray
        Audio float32 mono, 22050 Hz
    """
    engine = creer_engine(mode=mode, speaker=speaker, models_dir=models_dir,
                          api_url=api_url, api_key=api_key)
    result = engine.synthesize(
        texte,
        phrase_type=phrase_type,
        duration_scale=duration_scale,
        pitch_shift=pitch_shift,
        pitch_range=pitch_range,
        energy_scale=energy_scale,
        pause_scale=pause_scale,
        style=style,
        style_vector=style_vector,
    )
    return result.samples


__all__ = [
    "creer_engine",
    "synthetiser",
    "liste_speakers",
    "__version__",
]
