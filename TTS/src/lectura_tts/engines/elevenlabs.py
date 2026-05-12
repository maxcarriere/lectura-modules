"""Elevenlabs TTS — REST pur, zéro dépendance SDK.

Utilise l'API REST v1 text-to-speech avec une clé API.
Audio demandé en MP3, décodé via soundfile + numpy.
"""

from __future__ import annotations

import io
import json
import urllib.request
import urllib.error

import numpy as np

from lectura_tts.models import TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_SAMPLE_RATE = 24000

# Voix prédéfinies : "Nom (voice_id)"
_VOICES = [
    "Charlotte (XB0fDUnXU5powFXDhCwa)",
    "Alice (Xb7hH8MSUJpSbSDYk0k2)",
    "Aria (9BWtsMINqrJLrRacOk9x)",
    "Sarah (EXAVITQu4vr4xnSDxMaL)",
    "Laura (FGY2WhTYpPnrIDTdsKH5)",
    "Charlie (IKne3meq5aSn9XLyUdCD)",
    "George (JBFqnCBsd6RMkjVDRZzb)",
    "Callum (N2lVS1w4EtoT3dr4eOWO)",
    "River (SAz9YHcvj6GT2YYXdXww)",
    "Liam (TX3LPaxmHKxFdv7VOQHJ)",
    "Bill (pqHfZKP75CvOlQylNhV4)",
    "Lily (pFZP5JQG7iQjIQuC4Bku)",
]


def _parse_voice_id(voice: str) -> str:
    """Extrait le voice_id depuis 'Nom (id)' ou retourne tel quel."""
    if "(" in voice and voice.endswith(")"):
        return voice.rsplit("(", 1)[1].rstrip(")")
    return voice


class ElevenlabsTTSEngine:
    """Moteur Elevenlabs TTS via API REST."""

    def __init__(
        self,
        *,
        api_key: str = "",
        voice_id: str = "Charlotte (XB0fDUnXU5powFXDhCwa)",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        speed: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._voice_id = _parse_voice_id(voice_id)
        self._stability = max(0.0, min(1.0, stability))
        self._similarity_boost = max(0.0, min(1.0, similarity_boost))
        self._speed = max(0.5, min(2.0, speed))

    def synthesize(self, text: str) -> TTSResult:
        if not text or not text.strip():
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        if not self._api_key:
            raise ValueError(
                "Elevenlabs : clé API manquante. "
                "Renseignez-la dans Paramètres > TTS."
            )

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}"
        )

        body = json.dumps({
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": self._stability,
                "similarity_boost": self._similarity_boost,
                "speed": self._speed,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                mp3_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            try:
                err_body = json.loads(exc.read())
                msg = err_body.get("detail", {})
                if isinstance(msg, dict):
                    msg = msg.get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"Elevenlabs : {msg}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Elevenlabs : erreur réseau — {exc.reason}"
            ) from exc

        if not mp3_bytes:
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        import soundfile as sf
        data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        return TTSResult(
            samples=data.astype(np.float32),
            sample_rate=sr,
            phoneme_timings=[],
        )

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        return TTSResult(
            samples=np.array([], dtype=np.float32),
            sample_rate=_SAMPLE_RATE,
            phoneme_timings=[],
        )


# ── Auto-enregistrement ──

def _check() -> bool:
    return True


register(EngineInfo(
    key="elevenlabs",
    name="Elevenlabs",
    description="Voix neuronales Elevenlabs, haute qualité. Nécessite clé API et internet.",
    supports_phonemes=False,
    supports_text=True,
    requires_internet=True,
    requires_api_key=True,
    install_instructions="Aucune installation requise. Renseigner la clé API.",
    check_available=_check,
    factory=lambda p: ElevenlabsTTSEngine(**p),
    params=[
        EngineParam("api_key", "Clé API", "str", ""),
        EngineParam("voice_id", "Voix", "str", "Charlotte (XB0fDUnXU5powFXDhCwa)",
                     choices=_VOICES, role="voice"),
        EngineParam("stability", "Stabilité", "float", 0.5, min_val=0.0, max_val=1.0),
        EngineParam("similarity_boost", "Similarité", "float", 0.75,
                     min_val=0.0, max_val=1.0),
        EngineParam("speed", "Vitesse", "float", 1.0, min_val=0.5, max_val=2.0,
                     role="speed"),
    ],
    category="cloud",
))
