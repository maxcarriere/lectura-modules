"""Stub Google Cloud TTS — nécessite clé API.

Pré-requis : pip install lectura-tts[cloud-google]
"""

from __future__ import annotations

import numpy as np

from lectura_tts.models import TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_DEFAULT_SAMPLE_RATE = 24000


class GoogleCloudTTSEngine:
    """Stub pour Google Cloud TTS."""

    def __init__(self, *, api_key: str = "", voice: str = "fr-FR-Neural2-A") -> None:
        self._api_key = api_key
        self._voice = voice

    def synthesize(self, text: str) -> TTSResult:
        raise NotImplementedError("Google Cloud TTS : implémentation à venir.")

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        return TTSResult(samples=np.array([], dtype=np.float32),
                         sample_rate=_DEFAULT_SAMPLE_RATE, phoneme_timings=[])


# ── Auto-enregistrement ──

def _check() -> bool:
    try:
        import google.cloud.texttospeech  # noqa: F401
        return True
    except ImportError:
        return False


register(EngineInfo(
    key="cloud_google",
    name="Google Cloud TTS",
    description="Voix neuronales Google Cloud. Nécessite clé API et internet.",
    supports_phonemes=False,
    supports_text=True,
    requires_internet=True,
    requires_api_key=True,
    install_instructions="pip install lectura-tts[cloud-google]",
    check_available=_check,
    factory=lambda p: GoogleCloudTTSEngine(**p),
    params=[
        EngineParam("api_key", "Clé API", "str", ""),
        EngineParam("voice", "Voix", "str", "fr-FR-Neural2-A"),
    ],
))
