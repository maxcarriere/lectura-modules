"""lectura-tts-monospeaker — Synthese vocale neuronale monospeaker francais.

Exports publics :
    - creer_engine(mode, models_dir, api_url, api_key) → engine
    - synthetiser(texte, **kwargs) → numpy array float32
    - OnnxTTSEngine, ApiTTSEngine
    - TTSResult, PhonemeTiming
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__version__ = "1.1.1"


def creer_engine(
    mode: str = "auto",
    models_dir: str | Path | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
):
    """Factory pour creer un engine d'inference TTS.

    Parameters
    ----------
    mode : str
        "auto" : ONNX local si disponible, sinon API
        "local" : force l'inference ONNX locale
        "api" : force l'API distante
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

    Raises
    ------
    ImportError
        Si mode="local" et onnxruntime non installe
    FileNotFoundError
        Si mode="local" et modeles introuvables
    """
    if mode in ("auto", "local"):
        engine = _try_local(models_dir)
        if engine is not None:
            return engine
        if mode == "local":
            raise FileNotFoundError(
                "Modeles ONNX introuvables. Verifiez l'installation ou "
                "specifiez models_dir. Voir README pour les emplacements."
            )
        # mode="auto" → fallback API
        log.info("Modeles locaux non disponibles, fallback vers API")

    # API
    from lectura_tts_monospeaker.inference_api import ApiTTSEngine
    return ApiTTSEngine(api_url=api_url, api_key=api_key)


def _try_local(models_dir: str | Path | None = None):
    """Tente de creer un engine ONNX local."""
    try:
        import onnxruntime  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return None

    from lectura_tts_monospeaker._chargeur import find_models_dir

    resolved = find_models_dir(models_dir)
    if resolved is None:
        return None

    from lectura_tts_monospeaker.inference_onnx import OnnxTTSEngine
    return OnnxTTSEngine(resolved)


def synthetiser(
    texte: str,
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
) -> Any:
    """Convenience : texte → numpy audio float32.

    Cree un engine (ou reutilise le cache) et synthetise.

    Parameters
    ----------
    texte : str
        Texte francais a synthetiser
    mode, models_dir, api_url, api_key :
        Parametres de creer_engine()
    phrase_type, duration_scale, pitch_shift, pitch_range, energy_scale, pause_scale :
        Controles prosodiques

    Returns
    -------
    numpy.ndarray
        Audio float32 mono, 22050 Hz
    """
    engine = creer_engine(mode=mode, models_dir=models_dir,
                          api_url=api_url, api_key=api_key)
    result = engine.synthesize(
        texte,
        phrase_type=phrase_type,
        duration_scale=duration_scale,
        pitch_shift=pitch_shift,
        pitch_range=pitch_range,
        energy_scale=energy_scale,
        pause_scale=pause_scale,
    )
    return result.samples


__all__ = [
    "creer_engine",
    "synthetiser",
    "__version__",
]
