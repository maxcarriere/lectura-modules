"""Moteur TTS Edge (Microsoft) — voix neuronales haute qualité.

Pré-requis : pip install lectura-tts[edge]
"""

from __future__ import annotations

import asyncio
import io

import numpy as np

from lectura_tts.models import TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_DEFAULT_SAMPLE_RATE = 24000


class EdgeTtsTTSEngine:
    """Implémentation TTSEngine avec Edge TTS (voix neuronales Microsoft)."""

    def __init__(
        self,
        *,
        voice: str = "fr-FR-DeniseNeural",
        rate: str = "+0%",
    ) -> None:
        self._voice = voice
        self._rate = rate

    def synthesize(self, text: str) -> TTSResult:
        # Créer un event loop si nécessaire (appel depuis un thread non-async)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Déjà dans un event loop — utiliser un thread séparé
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                mp3_bytes = pool.submit(
                    lambda: asyncio.run(self._synthesize_async(text))
                ).result(timeout=30)
        else:
            mp3_bytes = asyncio.run(self._synthesize_async(text))

        if not mp3_bytes:
            return TTSResult(samples=np.array([], dtype=np.float32),
                             sample_rate=_DEFAULT_SAMPLE_RATE, phoneme_timings=[])

        import soundfile as sf
        data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        return TTSResult(samples=data.astype(np.float32), sample_rate=sr,
                         phoneme_timings=[])

    async def _synthesize_async(self, text: str) -> bytes:
        import edge_tts
        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        return TTSResult(samples=np.array([], dtype=np.float32),
                         sample_rate=_DEFAULT_SAMPLE_RATE, phoneme_timings=[])


# ── Auto-enregistrement ──

def _check() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


register(EngineInfo(
    key="edge",
    name="Edge TTS (Microsoft)",
    description="Voix neuronales Microsoft, haute qualité. Nécessite internet.",
    supports_phonemes=False,
    supports_text=True,
    requires_internet=True,
    requires_api_key=False,
    install_instructions="pip install lectura-tts[edge]",
    check_available=_check,
    factory=lambda p: EdgeTtsTTSEngine(**p),
    params=[
        EngineParam("voice", "Voix", "choice", "fr-FR-DeniseNeural",
                     choices=["fr-FR-DeniseNeural", "fr-FR-HenriNeural"]),
        EngineParam("rate", "Vitesse", "choice", "+0%",
                     choices=["-50%", "-20%", "+0%", "+20%", "+50%"]),
    ],
))
