"""Amazon Polly TTS — REST pur avec signature SigV4, zéro dépendance SDK.

Utilise l'API REST Polly avec signature AWS Signature Version 4 (hashlib/hmac).
Audio demandé en PCM 16 bits 16 kHz, décodé directement via numpy.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import urllib.parse
import urllib.request
import urllib.error

import numpy as np

from lectura_tts.models import TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_SAMPLE_RATE = 16000

# Mapping rétrocompat : ancien format "VoiceId" seul → (VoiceId, Engine)
_VOICE_COMPAT = {
    "Lea": ("Lea", "neural"),
    "Remi": ("Remi", "neural"),
    "Celine": ("Celine", "standard"),
    "Mathieu": ("Mathieu", "standard"),
}


def _parse_voice(voice: str) -> tuple[str, str]:
    """Parse 'Lea (neural)' → ('Lea', 'neural') avec rétrocompat."""
    if "(" in voice and voice.endswith(")"):
        name, engine = voice.rsplit("(", 1)
        return name.strip(), engine.rstrip(")").strip()
    # Rétrocompat : ancienne valeur sans parenthèses
    if voice in _VOICE_COMPAT:
        return _VOICE_COMPAT[voice]
    # Inconnu → fallback neural
    return voice, "neural"


# ── Helpers SigV4 ──────────────────────────────────────────────────────

def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(
    secret_key: str, date_stamp: str, region: str, service: str
) -> bytes:
    k_date = _hmac_sha256(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()
    return k_signing


def _build_sigv4_headers(
    method: str,
    host: str,
    path: str,
    headers: dict[str, str],
    payload: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = "polly",
) -> dict[str, str]:
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    # Canonical headers (must be sorted, lowercase)
    headers_to_sign = {
        "host": host,
        "x-amz-date": amz_date,
        "content-type": headers.get("Content-Type", "application/json"),
    }
    sorted_keys = sorted(headers_to_sign.keys())
    canonical_headers = "".join(
        f"{k}:{headers_to_sign[k]}\n" for k in sorted_keys
    )
    signed_headers = ";".join(sorted_keys)

    payload_hash = hashlib.sha256(payload).hexdigest()

    canonical_request = "\n".join([
        method,
        path,
        "",  # query string (empty)
        canonical_headers,
        signed_headers,
        payload_hash,
    ])

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    signing_key = _get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    return {
        "Content-Type": headers.get("Content-Type", "application/json"),
        "x-amz-date": amz_date,
        "Authorization": authorization,
    }


# ── Engine ─────────────────────────────────────────────────────────────

class AwsPollyTTSEngine:
    """Moteur Amazon Polly TTS via API REST + SigV4."""

    def __init__(
        self,
        *,
        access_key: str = "",
        secret_key: str = "",
        region: str = "eu-west-3",
        voice: str = "Lea (neural)",
        engine: str = "",  # ignoré, conservé pour rétrocompat
        speed: float = 1.0,
        pitch: float = 0.0,
    ) -> None:
        self._access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID", "")
        self._secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        self._region = region
        self._voice_id, self._engine = _parse_voice(voice)
        self._speed = speed
        self._pitch = pitch

    def _send_request(self, body_dict: dict) -> TTSResult:
        """Envoie la requête à Polly et retourne un TTSResult."""
        if not self._access_key or not self._secret_key:
            raise ValueError(
                "Amazon Polly : credentials manquants. "
                "Renseignez Access Key et Secret Key dans Paramètres > TTS, "
                "ou définissez AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
            )

        host = f"polly.{self._region}.amazonaws.com"
        path = "/v1/speech"
        url = f"https://{host}{path}"

        body = json.dumps(body_dict).encode("utf-8")
        base_headers = {"Content-Type": "application/json"}

        signed_headers = _build_sigv4_headers(
            method="POST",
            host=host,
            path=path,
            headers=base_headers,
            payload=body,
            access_key=self._access_key,
            secret_key=self._secret_key,
            region=self._region,
        )

        req = urllib.request.Request(
            url,
            data=body,
            headers=signed_headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                pcm_data = resp.read()
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
                msg = err_body
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"Amazon Polly : {msg}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Amazon Polly : erreur réseau — {exc.reason}"
            ) from exc

        if not pcm_data:
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        samples = np.frombuffer(pcm_data, dtype="<i2").astype(np.float32) / 32768.0

        return TTSResult(
            samples=samples,
            sample_rate=_SAMPLE_RATE,
            phoneme_timings=[],
        )

    def _needs_prosody(self) -> bool:
        return self._speed != 1.0 or self._pitch != 0.0

    def _prosody_attrs(self) -> str:
        parts: list[str] = []
        if self._speed != 1.0:
            parts.append(f'rate="{int(self._speed * 100)}%"')
        if self._pitch != 0.0:
            sign = "+" if self._pitch > 0 else ""
            parts.append(f'pitch="{sign}{self._pitch:.0f}%"')
        return " ".join(parts)

    def synthesize(self, text: str) -> TTSResult:
        if not text or not text.strip():
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        if self._needs_prosody():
            ssml = (
                f'<speak><prosody {self._prosody_attrs()}>'
                f'{_escape_xml(text)}</prosody></speak>'
            )
            return self._send_request({
                "Text": ssml,
                "TextType": "ssml",
                "VoiceId": self._voice_id,
                "OutputFormat": "pcm",
                "SampleRate": str(_SAMPLE_RATE),
                "Engine": self._engine,
                "LanguageCode": "fr-FR",
            })

        return self._send_request({
            "Text": text,
            "VoiceId": self._voice_id,
            "OutputFormat": "pcm",
            "SampleRate": str(_SAMPLE_RATE),
            "Engine": self._engine,
            "LanguageCode": "fr-FR",
        })

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        if not phonemes_ipa or not phonemes_ipa.strip():
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        phoneme_tag = f'<phoneme alphabet="ipa" ph="{_escape_xml(phonemes_ipa)}">.</phoneme>'
        if self._needs_prosody():
            ssml = (
                f'<speak><prosody {self._prosody_attrs()}>'
                f'{phoneme_tag}</prosody></speak>'
            )
        else:
            ssml = f'<speak>{phoneme_tag}</speak>'

        return self._send_request({
            "Text": ssml,
            "TextType": "ssml",
            "VoiceId": self._voice_id,
            "OutputFormat": "pcm",
            "SampleRate": str(_SAMPLE_RATE),
            "Engine": self._engine,
            "LanguageCode": "fr-FR",
        })


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
    key="cloud_aws",
    name="Amazon Polly",
    description="Voix neuronales Amazon. Nécessite credentials AWS et internet.",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=True,
    requires_api_key=True,
    install_instructions="Aucune installation requise. Renseigner Access Key et Secret Key.",
    check_available=_check,
    factory=lambda p: AwsPollyTTSEngine(**p),
    params=[
        EngineParam("access_key", "Access Key", "str", ""),
        EngineParam("secret_key", "Secret Key", "str", ""),
        EngineParam("region", "Région AWS", "str", "eu-west-3"),
        EngineParam("voice", "Voix", "choice", "Lea (neural)",
                     choices=[
                         "Lea (neural)",
                         "Remi (neural)",
                         "Celine (standard)",
                         "Mathieu (standard)",
                     ], role="voice"),
        EngineParam("speed", "Vitesse", "float", 1.0, min_val=0.5, max_val=2.0,
                     role="speed"),
        EngineParam("pitch", "Hauteur (%)", "float", 0.0, min_val=-30.0, max_val=30.0,
                     role="pitch"),
    ],
    category="cloud",
))
