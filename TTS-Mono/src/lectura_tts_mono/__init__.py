"""lectura-tts-mono — Pipeline TTS monospeaker du francais.

Orchestre le pipeline complet : texte → G2P → Matcha-Conformer → audio.
Retimbre optionnel via OpenVoice (pip install lectura-tts-mono[retimbre]).

Exports publics :
    - creer_engine(mode, models_dir, api_url, api_key) -> engine
    - synthetiser(texte, **kwargs) -> numpy array float32
"""

from __future__ import annotations

__version__ = "1.0.0"

# Re-export depuis le moteur brut
from lectura_monospeaker import creer_engine, synthetiser

__all__ = [
    "creer_engine",
    "synthetiser",
    "__version__",
]
