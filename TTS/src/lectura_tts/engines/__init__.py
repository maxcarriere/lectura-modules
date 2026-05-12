"""Charge tous les moteurs TTS → déclenche register() pour chacun.

Les imports sont protégés : si un moteur échoue à l'import (dépendance
manquante), il est simplement ignoré.
"""

import importlib

_ENGINE_MODULES = [
    "lectura_tts.engines.espeak",
    "lectura_tts.engines.mbrola",
    "lectura_tts.engines.piper",
    "lectura_tts.engines.kokoro",
    "lectura_tts.engines.monospeaker",
    "lectura_tts.engines.multispeaker",
    "lectura_tts.engines.diphone",
    "lectura_tts.engines.gtts",
    "lectura_tts.engines.edge",
    "lectura_tts.engines.cloud_google",
    "lectura_tts.engines.cloud_aws",
    "lectura_tts.engines.cloud_azure",
    "lectura_tts.engines.elevenlabs",
]

for _mod in _ENGINE_MODULES:
    try:
        importlib.import_module(_mod)
    except Exception:
        pass
