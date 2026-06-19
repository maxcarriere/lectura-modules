"""lectura-tts-dipho — Pipeline TTS diphonique du francais.

Orchestre le pipeline complet : texte → G2P → concatenation diphones WORLD → audio.
Retimbre optionnel via OpenVoice (pip install lectura-tts-dipho[retimbre]).

Exports publics :
    - creer_engine(mode, g2p, models_dir, api_url) -> DiphoneEngine
    - synthetiser(texte, **kwargs) -> numpy array float32 @ 44100 Hz
"""

from __future__ import annotations

__version__ = "1.0.0"

# Re-export depuis le moteur brut
from lectura_diphone import creer_engine, synthetiser

__all__ = [
    "creer_engine",
    "synthetiser",
    "__version__",
]
