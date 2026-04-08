"""Moteur TTS gTTS (Google Translate) — synthèse cloud.

Pré-requis : pip install lectura-tts[gtts]
"""

from __future__ import annotations

import io

import numpy as np

from lectura_tts.models import TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_DEFAULT_SAMPLE_RATE = 24000


class GTtsTTSEngine:
    """Implémentation TTSEngine avec gTTS (Google Translate TTS)."""

    def __init__(self, *, slow: bool = False) -> None:
        self._slow = slow

    def synthesize(self, text: str) -> TTSResult:
        from gtts import gTTS
        import soundfile as sf

        tts = gTTS(text, lang="fr", slow=self._slow)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)

        data, sr = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        return TTSResult(samples=data.astype(np.float32), sample_rate=sr,
                         phoneme_timings=[])

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        return TTSResult(samples=np.array([], dtype=np.float32),
                         sample_rate=_DEFAULT_SAMPLE_RATE, phoneme_timings=[])


# ── Auto-enregistrement ──

def _check() -> bool:
    try:
        import gtts  # noqa: F401
        return True
    except ImportError:
        return False


register(EngineInfo(
    key="gtts",
    name="gTTS (Google)",
    description="Synthèse via Google Translate. Voix naturelle, nécessite internet.",
    supports_phonemes=False,
    supports_text=True,
    requires_internet=True,
    requires_api_key=False,
    install_instructions="pip install lectura-tts[gtts]",
    check_available=_check,
    factory=lambda p: GTtsTTSEngine(**p),
    params=[
        EngineParam("slow", "Parole lente", "bool", False),
    ],
))
