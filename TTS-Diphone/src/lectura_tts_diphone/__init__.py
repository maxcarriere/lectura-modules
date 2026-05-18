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

__version__ = "1.7.0"


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
    duration_scale: float = 1.0,
    pause_scale: float = 1.0,
    macro_expressivity: float = 1.0,
    micro_expressivity: float = 1.0,
    seed: int | None = None,
    prosody_style: str = "regles",
    ap_cleanup: float = 1.5,
    formant_sharpening: float = 1.3,
    vtln_alpha: float = 1.0,
    timbre: str | None = None,
    base_f0: float = 175.0,
    sentence_pause_ms: float = 400.0,
    # -- Retimbre (OpenVoice zero-shot) --
    voix: str | list[str] | dict[str, float] | None = None,
    voix_variante: float = 0.0,
    voix_tau: float = 0.3,
    vc_models_dir: str | Path | None = None,
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
    macro_expressivity : float
        Facteur des gestes prosodiques (0=plat, 1=normal, 2=exagere)
    micro_expressivity : float
        Facteur des micro-variations (0=robot, 1=normal, 2=tres expressif).
        Actif en mode FLUIDE uniquement.
    seed : int | None
        Graine aleatoire pour la micro-prosodie.
        None = aleatoire, meme seed = meme resultat.
    prosody_style : str
        Style prosodique : "regles" (defaut, prosodie a base de regles LHiLH*)
        ou "corpus" (prosodie extraite du corpus SIWIS, plus variee).
    ap_cleanup : float
        Compression AP (1.0=off, 1.5=defaut, max 3.0). Reduit la raucite.
    formant_sharpening : float
        Affutage formants (1.0=off, 1.3=defaut, max 2.0). Restaure la nettete.
    vtln_alpha : float
        Warping VTLN (0.8=grave, 1.0=neutre, 1.2=aigu).
    timbre : str | None
        Nom de signature de timbre ("homme", "enfant", etc.) ou chemin
        vers un fichier .json. None = pas de transfert de timbre.
    base_f0 : float
        Pitch de base en Hz (defaut 175.0). homme ~120, femme ~200,
        enfant ~280.
    sentence_pause_ms : float
        Pause inter-phrase en ms (defaut 400). Controle la duree du
        silence entre deux phrases (separees par . ! ? ...).
    voix : str | list[str] | dict[str, float] | None
        Voix cible pour retimbre OpenVoice. Polymorphe :
        - str : nom de preset ("siwis") ou chemin fichier audio.
        - list[str] : plusieurs references (poids egaux).
        - dict[str, float] : blend pondere.
          Ex: {"siwis": 0.5, "nadine": 0.3, "ezwa": 0.2}
        None = pas de retimbre.
        Presets: siwis, ezwa, nadine, bernard, gilles, zeckou.
        Requires: pip install 'lectura-tts-diphone[vc]'
    voix_variante : float
        Curseur de variante vocale (-1 a +1).
        -1 = grave/masculin, 0 = neutre, +1 = aigu/enfant.
    voix_tau : float
        Parametre tau d'OpenVoice (0 = deterministe, 0.3 = defaut).
    vc_models_dir : str | Path | None
        Repertoire des modeles VC (defaut: auto-detection).

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
        macro_expressivity=macro_expressivity,
        micro_expressivity=micro_expressivity,
        seed=seed,
        prosody_style=prosody_style,
        ap_cleanup=ap_cleanup,
        formant_sharpening=formant_sharpening,
        vtln_alpha=vtln_alpha,
        timbre=timbre,
        base_f0=base_f0,
        sentence_pause_ms=sentence_pause_ms,
        voix=voix,
        voix_variante=voix_variante,
        voix_tau=voix_tau,
        vc_models_dir=vc_models_dir,
    )


__all__ = [
    "creer_engine",
    "synthetiser",
    "__version__",
]
