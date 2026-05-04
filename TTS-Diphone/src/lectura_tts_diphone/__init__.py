"""lectura-tts-diphone — Synthese vocale par concatenation de diphones WORLD.

Exports publics :
    - creer_engine(mode, g2p, models_dir, api_url) → DiphoneEngine
    - synthetiser(texte, **kwargs) → numpy array float32 @ 44100 Hz
    - DiphoneEngine, SynthMode
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

__version__ = "1.0.0"


def creer_engine(
    mode: str = "auto",
    g2p: Any = None,
    models_dir: str | Path | None = None,
    api_url: str | None = None,
):
    """Factory pour creer un engine de synthese diphone.

    Parameters
    ----------
    mode : str
        "auto" : local si pyworld disponible
        "local" : force l'inference locale
    g2p : G2PBackend | callable | None
        Backend G2P. Si None, auto-detection.
        Peut etre un callable(text) → list[dict].
    models_dir : Path | None
        Repertoire des modeles (override la detection auto)
    api_url : str | None
        URL du serveur API G2P

    Returns
    -------
    DiphoneEngine
        Engine avec interface synthesize_groups/synthesize_phones

    Raises
    ------
    ImportError
        Si pyworld/numpy/scipy non installes
    FileNotFoundError
        Si modeles introuvables
    """
    # Verifier deps
    try:
        import numpy  # noqa: F401
        import pyworld  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Dependance manquante: {e}. "
            "Installez avec: pip install 'lectura-tts-diphone[local]'"
        ) from e

    from lectura_tts_diphone.engine import DiphoneEngine

    engine = DiphoneEngine()
    engine.load(models_dir=models_dir)

    # Attacher le G2P
    if g2p is not None:
        if callable(g2p) and not hasattr(g2p, "phonemize"):
            from lectura_tts_diphone.g2p import CallableG2P
            engine._g2p_backend = CallableG2P(g2p)
        else:
            engine._g2p_backend = g2p
    else:
        from lectura_tts_diphone.g2p import auto_detect_g2p
        engine._g2p_backend = auto_detect_g2p(api_url=api_url)

    return engine


def synthetiser(
    texte: str,
    mode: str = "auto",
    models_dir: str | Path | None = None,
    api_url: str | None = None,
    g2p: Any = None,
    synth_mode: str = "FLUIDE",
    duration_scale: float = 1.2,
    pause_scale: float = 1.2,
) -> Any:
    """Convenience : texte → numpy audio float32 @ 44100 Hz.

    Parameters
    ----------
    texte : str
        Texte francais a synthetiser
    mode, models_dir, api_url, g2p :
        Parametres de creer_engine()
    synth_mode : str
        "FLUIDE", "MOT_A_MOT", ou "SYLLABES"
    duration_scale : float
        Facteur de vitesse (>1 = plus lent)
    pause_scale : float
        Facteur de pauses

    Returns
    -------
    numpy.ndarray
        Audio float32 mono, 44100 Hz
    """
    engine = creer_engine(mode=mode, g2p=g2p, models_dir=models_dir,
                          api_url=api_url)

    # G2P
    groups = engine._g2p_backend.phonemize(texte)
    if not groups:
        import numpy as np
        return np.array([], dtype=np.float32)

    return engine.synthesize_groups(
        groups,
        mode=synth_mode,
        duration_scale=duration_scale,
        pause_scale=pause_scale,
    )


__all__ = [
    "creer_engine",
    "synthetiser",
    "__version__",
]
