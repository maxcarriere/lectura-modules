"""Stub Amazon Polly — nécessite credentials AWS.

Pré-requis : pip install lectura-tts[cloud-aws]
"""

from __future__ import annotations

import numpy as np

from lectura_tts.models import TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_DEFAULT_SAMPLE_RATE = 24000


class AwsPollyTTSEngine:
    """Stub pour Amazon Polly TTS."""

    def __init__(self, *, voice: str = "Lea", region: str = "eu-west-3") -> None:
        self._voice = voice
        self._region = region

    def synthesize(self, text: str) -> TTSResult:
        raise NotImplementedError("AWS Polly : implémentation à venir.")

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        return TTSResult(samples=np.array([], dtype=np.float32),
                         sample_rate=_DEFAULT_SAMPLE_RATE, phoneme_timings=[])


# ── Auto-enregistrement ──

def _check() -> bool:
    try:
        import boto3  # noqa: F401
        return True
    except ImportError:
        return False


register(EngineInfo(
    key="cloud_aws",
    name="Amazon Polly",
    description="Voix neuronales Amazon. Nécessite credentials AWS et internet.",
    supports_phonemes=False,
    supports_text=True,
    requires_internet=True,
    requires_api_key=True,
    install_instructions="pip install lectura-tts[cloud-aws]",
    check_available=_check,
    factory=lambda p: AwsPollyTTSEngine(**p),
    params=[
        EngineParam("voice", "Voix", "choice", "Lea", choices=["Lea", "Remi"]),
        EngineParam("region", "Région AWS", "str", "eu-west-3"),
    ],
))
