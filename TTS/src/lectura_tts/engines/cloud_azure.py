"""Azure TTS — REST pur, zéro dépendance SDK.

Utilise l'API REST Cognitive Services avec une clé d'abonnement.
Audio demandé en WAV (Riff24Khz16BitMonoPcm), décodé via wave + numpy.
"""

from __future__ import annotations

import io
import urllib.request
import urllib.error
import wave

import numpy as np

from lectura_tts.models import TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_SAMPLE_RATE = 24000


class AzureTTSEngine:
    """Moteur Azure TTS via API REST."""

    def __init__(
        self,
        *,
        subscription_key: str = "",
        region: str = "westeurope",
        voice: str = "fr-FR-DeniseNeural",
        rate: float = 1.0,
        pitch: float = 0.0,
    ) -> None:
        self._subscription_key = subscription_key
        self._region = region or "westeurope"
        self._voice = voice
        self._rate = max(0.5, min(2.0, rate))
        self._pitch = max(-50.0, min(50.0, pitch))

    def _send_ssml(self, ssml: str) -> TTSResult:
        """Envoie le SSML à Azure et retourne un TTSResult."""
        if not self._subscription_key:
            raise ValueError(
                "Azure TTS : clé d'abonnement manquante. "
                "Renseignez-la dans Paramètres > TTS."
            )

        url = (
            f"https://{self._region}.tts.speech.microsoft.com"
            f"/cognitiveservices/v1"
        )

        req = urllib.request.Request(
            url,
            data=ssml.encode("utf-8"),
            headers={
                "Ocp-Apim-Subscription-Key": self._subscription_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                "User-Agent": "LecturaTTS",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                wav_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(exc)
            raise RuntimeError(f"Azure TTS : {err_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Azure TTS : erreur réseau — {exc.reason}"
            ) from exc

        if not wav_bytes:
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        return TTSResult(
            samples=samples,
            sample_rate=sample_rate,
            phoneme_timings=[],
        )

    def synthesize(self, text: str) -> TTSResult:
        if not text or not text.strip():
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        rate_pct = f"{(self._rate - 1.0) * 100:+.0f}%"
        pitch_pct = f"{self._pitch:+.0f}%"
        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xml:lang="fr-FR">'
            f'<voice name="{self._voice}">'
            f'<prosody rate="{rate_pct}" pitch="{pitch_pct}">'
            f'{_escape_xml(text)}'
            '</prosody></voice></speak>'
        )
        return self._send_ssml(ssml)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        if not phonemes_ipa or not phonemes_ipa.strip():
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        rate_pct = f"{(self._rate - 1.0) * 100:+.0f}%"
        pitch_pct = f"{self._pitch:+.0f}%"
        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xml:lang="fr-FR">'
            f'<voice name="{self._voice}">'
            f'<prosody rate="{rate_pct}" pitch="{pitch_pct}">'
            f'<phoneme alphabet="ipa" ph="{_escape_xml(phonemes_ipa)}">.</phoneme>'
            '</prosody></voice></speak>'
        )
        return self._send_ssml(ssml)


def _escape_xml(text: str) -> str:
    """Échappe les caractères spéciaux XML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ── Auto-enregistrement ──

def _check() -> bool:
    return True


register(EngineInfo(
    key="cloud_azure",
    name="Azure TTS",
    description="Voix neuronales Microsoft Azure. Nécessite clé d'abonnement et internet.",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=True,
    requires_api_key=True,
    install_instructions="Aucune installation requise. Renseigner la clé d'abonnement.",
    check_available=_check,
    factory=lambda p: AzureTTSEngine(**p),
    params=[
        EngineParam("subscription_key", "Clé d'abonnement", "str", ""),
        EngineParam("region", "Région", "str", "westeurope"),
        EngineParam("voice", "Voix", "choice", "fr-FR-DeniseNeural",
                     choices=[
                         "fr-FR-DeniseNeural",
                         "fr-FR-HenriNeural",
                         "fr-FR-EloiseNeural",
                         "fr-FR-CoralieNeural",
                         "fr-FR-VivienneMultilingualNeural",
                         "fr-FR-RemyMultilingualNeural",
                         "fr-FR-LucienMultilingualNeural",
                         "fr-FR-AlainNeural",
                         "fr-FR-BrigitteNeural",
                         "fr-FR-CelesteNeural",
                         "fr-FR-ClaudeNeural",
                         "fr-FR-JacquelineNeural",
                         "fr-FR-JeromeNeural",
                         "fr-FR-JosephineNeural",
                         "fr-FR-MauriceNeural",
                         "fr-FR-YvesNeural",
                     ],
                     role="voice"),
        EngineParam("rate", "Vitesse", "float", 1.0, min_val=0.5, max_val=2.0,
                     role="speed"),
        EngineParam("pitch", "Hauteur", "float", 0.0, min_val=-50.0, max_val=50.0,
                     role="pitch"),
    ],
    category="cloud",
))
