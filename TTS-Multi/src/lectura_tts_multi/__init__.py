"""lectura-tts-multi — Pipeline TTS multi-speaker du francais.

Orchestre le pipeline complet : texte → G2P → Matcha-Conformer multi-speaker → audio.
Retimbre optionnel via OpenVoice (pip install lectura-tts-multi[retimbre]).

Exports publics :
    - creer_engine(mode, speaker, models_dir, ...) -> engine
    - synthetiser(texte, speaker, **kwargs) -> numpy array float32
    - liste_speakers() -> list[dict]
"""

from __future__ import annotations

__version__ = "1.0.0"

# Re-export depuis le moteur brut
from lectura_multispeaker import creer_engine, synthetiser, liste_speakers

__all__ = [
    "creer_engine",
    "synthetiser",
    "liste_speakers",
    "__version__",
]
