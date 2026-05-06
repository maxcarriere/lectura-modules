"""Backend API pour le TTS multi-speaker — delegue l'inference au serveur Lectura.

Meme interface que OnnxTTSEngine.
Utilise uniquement la stdlib (urllib), zero dependance externe.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.lec-tu-ra.com"
_TIMEOUT = 30


@dataclass
class PhonemeTiming:
    """Timing d'un phoneme dans l'audio synthetise."""
    ipa: str
    start_ms: float
    end_ms: float


@dataclass
class TTSResult:
    """Resultat d'une synthese TTS."""
    samples: Any  # numpy array si disponible, sinon bytes
    sample_rate: int
    phoneme_timings: list[PhonemeTiming] = field(default_factory=list)


class ApiTTSEngine:
    """Backend API multi-speaker — meme interface que OnnxTTSEngine.

    Parameters
    ----------
    api_url : str | None
        URL de base du serveur (defaut : LECTURA_API_URL ou https://api.lec-tu-ra.com)
    api_key : str | None
        Cle API (defaut : LECTURA_API_KEY ou vide pour le mode demo)
    speaker : str
        Speaker par defaut
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        speaker: str = "siwis",
    ) -> None:
        self._url = (
            api_url
            or os.environ.get("LECTURA_API_URL", "")
            or _DEFAULT_API_URL
        )
        self._key = api_key or os.environ.get("LECTURA_API_KEY", "")
        self._speaker = speaker
        self._sample_rate = 22050

    @property
    def speaker(self) -> str:
        """Speaker actuellement selectionne."""
        return self._speaker

    def set_speaker(self, speaker: str) -> None:
        """Change le speaker actif."""
        self._speaker = speaker

    def synthesize(
        self,
        text: str,
        phrase_type: int | None = None,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
        variability: bool = False,
    ) -> TTSResult:
        """Synthetise du texte via l'API."""
        payload: dict = {
            "text": text,
            "speaker": self._speaker,
            "duration_scale": duration_scale,
            "pitch_shift": pitch_shift,
            "pitch_range": pitch_range,
            "energy_scale": energy_scale,
            "pause_scale": pause_scale,
        }
        if phrase_type is not None:
            payload["phrase_type"] = phrase_type
        return self._call_api(payload)

    def synthesize_phonemes(
        self,
        phonemes_ipa: str,
        phrase_type: int = 0,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
        variability: bool = False,
    ) -> TTSResult:
        """Synthetise des phonemes IPA via l'API."""
        return self._call_api({
            "ipa": phonemes_ipa,
            "speaker": self._speaker,
            "phrase_type": phrase_type,
            "duration_scale": duration_scale,
            "pitch_shift": pitch_shift,
            "pitch_range": pitch_range,
            "energy_scale": energy_scale,
            "pause_scale": pause_scale,
        })

    def _call_api(self, payload: dict) -> TTSResult:
        """Appel HTTP POST vers /tts/synthesize."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        url = f"{self._url.rstrip('/')}/tts/synthesize"
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Erreur API TTS {exc.code} : {msg}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Impossible de contacter le serveur Lectura ({self._url}) : {exc.reason}"
            ) from None

        # Decoder l'audio base64
        audio_b64 = result.get("audio_base64", "")
        sample_rate = result.get("sample_rate", 22050)

        try:
            import numpy as np
            audio_bytes = base64.b64decode(audio_b64)
            samples = np.frombuffer(audio_bytes, dtype=np.float32)
        except ImportError:
            samples = base64.b64decode(audio_b64)

        # Timings
        timings_raw = result.get("phoneme_timings", [])
        timings = [
            PhonemeTiming(ipa=t["ipa"], start_ms=t["start_ms"], end_ms=t["end_ms"])
            for t in timings_raw
        ]

        return TTSResult(
            samples=samples,
            sample_rate=sample_rate,
            phoneme_timings=timings,
        )
