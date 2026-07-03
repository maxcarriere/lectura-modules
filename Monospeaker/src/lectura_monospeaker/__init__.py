"""lectura-monospeaker — Synthese vocale neuronale monospeaker francais.

Architecture v2 : Matcha-Conformer (17.9M params) + HiFi-GAN (3.6M params).
Fallback v1 : FastPitch-Lite + HiFi-GAN.

Exports publics :
    - creer_engine(mode, models_dir, api_url, api_key) -> engine
    - synthetiser(texte, **kwargs) -> numpy array float32
    - OnnxTTSEngine, ApiTTSEngine
    - TTSResult, PhonemeTiming
    - STYLE_PRESETS
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__version__ = "4.0.0"


def creer_engine(
    mode: str = "auto",
    models_dir: str | Path | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str = "high",
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
    model : str
        Choix du modele : "high" (Matcha-Conformer) ou "light" (FastPitch).

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
        engine = _try_local(models_dir, model=model)
        if engine is not None:
            return engine
        if mode == "local":
            raise FileNotFoundError(
                "Modeles ONNX introuvables. Verifiez l'installation ou "
                "specifiez models_dir. Voir README pour les emplacements."
            )
        # mode="auto" -> fallback API
        log.info("Modeles locaux non disponibles, fallback vers API")

    # API
    from lectura_monospeaker.inference_api import ApiTTSEngine
    return ApiTTSEngine(api_url=api_url, api_key=api_key, model=model)


def _try_local(models_dir: str | Path | None = None, model: str = "high"):
    """Tente de creer un engine ONNX local."""
    try:
        import onnxruntime  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return None

    from lectura_monospeaker._chargeur import find_models_dir

    resolved = find_models_dir(models_dir, model_variant=model)
    if resolved is None:
        return None

    from lectura_monospeaker.inference_onnx import OnnxTTSEngine
    return OnnxTTSEngine(resolved)


def synthetiser(
    texte: str,
    mode: str = "auto",
    models_dir: str | Path | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    phrase_type: int | None = None,
    duration_scale: float | None = None,
    pitch_shift: float | None = None,
    pitch_range: float | None = None,
    energy_scale: float | None = None,
    pause_scale: float | None = None,
    duration_noise: float | None = None,
    style: str | None = None,
    style_vector: list[float] | None = None,
    n_ode_steps: int | None = None,
    # -- Retimbre (OpenVoice zero-shot) --
    voix: str | list[str] | dict[str, float] | None = None,
    voix_variante: float = 0.0,
    voix_tau: float = 0.3,
    vc_models_dir: str | Path | None = None,
    model: str = "high",
) -> Any:
    """Convenience : texte -> numpy audio float32.

    Cree un engine (ou reutilise le cache) et synthetise.

    Parameters
    ----------
    texte : str
        Texte francais a synthetiser
    mode, models_dir, api_url, api_key :
        Parametres de creer_engine()
    phrase_type, duration_scale, pitch_shift, pitch_range, energy_scale, pause_scale :
        Controles prosodiques
    duration_noise : float | None
        Amplitude du bruit de duree lisse (0.0=off, ~0.1=subtil, ~0.2=prononce).
        Cree une variation de rythme naturelle. None = valeur du style preset.
    style : str | None
        Preset de style : neutre, narratif, dialogue, expressif, meditatif, rapide, lent
    style_vector : list[float] | None
        Vecteur de style explicite [5 dims]. Prioritaire sur style.
    n_ode_steps : int | None
        Nombre de pas ODE pour Matcha (None=defaut config, typ. 4).
        Plus = meilleure qualite, moins = plus rapide.
    voix : str | list[str] | dict[str, float] | None
        Voix cible pour retimbre OpenVoice. Polymorphe :
        - str : nom de preset ("siwis") ou chemin fichier audio.
        - list[str] : plusieurs references (poids egaux).
        - dict[str, float] : blend pondere.
          Ex: {"siwis": 0.5, "nadine": 0.3, "ezwa": 0.2}
        None = pas de retimbre.
        Presets: siwis, ezwa, nadine, bernard, gilles, zeckou.
        Requires: pip install 'lectura-tts-monospeaker[vc]'
    voix_variante : float
        Curseur de variante vocale (-1 a +1).
        -1 = grave/masculin, 0 = neutre, +1 = aigu/enfant.
    voix_tau : float
        Parametre tau d'OpenVoice (0 = deterministe, 0.3 = defaut).
    vc_models_dir : str | Path | None
        Repertoire des modeles VC (defaut: auto-detection).
    model : str
        Choix du modele : "high" (Matcha-Conformer) ou "light" (FastPitch).

    Returns
    -------
    numpy.ndarray
        Audio float32 mono, 22050 Hz
    """
    engine = creer_engine(mode=mode, models_dir=models_dir,
                          api_url=api_url, api_key=api_key, model=model)
    result = engine.synthesize(
        texte,
        phrase_type=phrase_type,
        duration_scale=duration_scale,
        pitch_shift=pitch_shift,
        pitch_range=pitch_range,
        energy_scale=energy_scale,
        pause_scale=pause_scale,
        duration_noise=duration_noise,
        style=style,
        style_vector=style_vector,
        n_ode_steps=n_ode_steps,
        voix=voix,
        voix_variante=voix_variante,
        voix_tau=voix_tau,
        vc_models_dir=vc_models_dir,
    )
    return result.samples


__all__ = [
    "creer_engine",
    "synthetiser",
    "__version__",
]
