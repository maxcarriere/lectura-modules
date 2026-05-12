"""Google Cloud TTS — REST pur, zéro dépendance SDK.

Deux méthodes d'authentification :
  - Clé API : header X-Goog-Api-Key (le plus simple)
  - Service account JSON : JWT → access token OAuth2 (via cryptography)

Audio demandé en LINEAR16 (WAV), décodé via wave + numpy.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

import numpy as np

from lectura_tts.models import TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_SAMPLE_RATE = 24000


# ── JWT / Service Account helpers ─────────────────────────────────────

def _b64url(data: bytes) -> bytes:
    """Base64url sans padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _make_jwt(client_email: str, private_key_pem: str) -> str:
    """Crée un JWT signé RS256 pour l'échange OAuth2."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    now = int(time.time())
    claims = _b64url(json.dumps({
        "iss": client_email,
        "scope": _SCOPE,
        "aud": _TOKEN_URL,
        "iat": now,
        "exp": now + 3600,
    }).encode())

    signing_input = header + b"." + claims
    key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return (signing_input + b"." + _b64url(signature)).decode()


def _get_access_token(creds: dict) -> str:
    """Échange un JWT contre un access token Google OAuth2."""
    jwt_token = _make_jwt(creds["client_email"], creds["private_key"])
    data = urllib.parse.urlencode({
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": jwt_token,
    }).encode()
    req = urllib.request.Request(_TOKEN_URL, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())["access_token"]


def _escape_xml(text: str) -> str:
    """Échappe les caractères spéciaux XML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


class GoogleCloudTTSEngine:
    """Moteur Google Cloud TTS via API REST."""

    def __init__(
        self,
        *,
        api_key: str = "",
        credentials_file: str = "",
        voice: str = "fr-FR-Neural2-A",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ) -> None:
        self._api_key = api_key
        self._credentials_file = credentials_file
        self._voice = voice
        self._speaking_rate = speaking_rate
        self._pitch = pitch
        self._language_code = voice[:5] if len(voice) >= 5 else "fr-FR"
        self._access_token: str = ""
        self._token_expiry: float = 0.0

    def _get_auth_headers(self) -> dict[str, str]:
        """Retourne les headers d'authentification (API key ou Bearer token)."""
        if self._credentials_file:
            # Rafraîchir le token si expiré (marge de 5 min)
            if time.time() >= self._token_expiry - 300:
                with open(self._credentials_file, encoding="utf-8") as f:
                    creds = json.load(f)
                self._access_token = _get_access_token(creds)
                self._token_expiry = time.time() + 3600
            return {"Authorization": f"Bearer {self._access_token}"}

        if self._api_key:
            return {"X-Goog-Api-Key": self._api_key}

        raise ValueError(
            "Google Cloud TTS : clé API ou fichier service account manquant. "
            "Renseignez-les dans Paramètres > TTS."
        )

    def _send_request(self, body_dict: dict) -> TTSResult:
        """Envoie la requête à l'API Google TTS et retourne un TTSResult."""
        body = json.dumps(body_dict).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        headers.update(self._get_auth_headers())

        req = urllib.request.Request(
            _API_URL,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                err_body = json.loads(exc.read())
                msg = err_body.get("error", {}).get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"Google Cloud TTS : {msg}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Google Cloud TTS : erreur réseau — {exc.reason}"
            ) from exc

        audio_b64 = result.get("audioContent", "")
        if not audio_b64:
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        wav_bytes = base64.b64decode(audio_b64)

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

        return self._send_request({
            "input": {"text": text},
            "voice": {
                "languageCode": self._language_code,
                "name": self._voice,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "speakingRate": self._speaking_rate,
                "pitch": self._pitch,
                "sampleRateHertz": _SAMPLE_RATE,
            },
        })

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        if not phonemes_ipa or not phonemes_ipa.strip():
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        ssml = (
            '<speak>'
            f'<phoneme alphabet="ipa" ph="{_escape_xml(phonemes_ipa)}">.</phoneme>'
            '</speak>'
        )
        return self._send_request({
            "input": {"ssml": ssml},
            "voice": {
                "languageCode": self._language_code,
                "name": self._voice,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "speakingRate": self._speaking_rate,
                "pitch": self._pitch,
                "sampleRateHertz": _SAMPLE_RATE,
            },
        })


# ── Auto-enregistrement ──

def _check() -> bool:
    return True


register(EngineInfo(
    key="cloud_google",
    name="Google Cloud TTS",
    description="Voix neuronales Google Cloud. Nécessite clé API ou service account.",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=True,
    requires_api_key=True,
    install_instructions="Aucune installation requise. Renseigner clé API ou fichier service account.",
    check_available=_check,
    factory=lambda p: GoogleCloudTTSEngine(**p),
    params=[
        EngineParam("api_key", "Clé API", "str", ""),
        EngineParam("credentials_file", "Fichier service account (JSON)", "file", "",
                     file_filter="JSON (*.json);;Tous (*)"),
        EngineParam("voice", "Voix", "choice", "fr-FR-Neural2-A",
                     choices=[
                         "fr-FR-Neural2-A", "fr-FR-Neural2-B",
                         "fr-FR-Neural2-C", "fr-FR-Neural2-D", "fr-FR-Neural2-E",
                         "fr-FR-Wavenet-A", "fr-FR-Wavenet-B",
                         "fr-FR-Wavenet-C", "fr-FR-Wavenet-D", "fr-FR-Wavenet-E",
                         "fr-FR-Standard-A", "fr-FR-Standard-B",
                         "fr-FR-Standard-C", "fr-FR-Standard-D", "fr-FR-Standard-E",
                         "fr-FR-Studio-A", "fr-FR-Studio-D",
                     ], role="voice"),
        EngineParam("speaking_rate", "Vitesse", "float", 1.0, min_val=0.25, max_val=4.0,
                     role="speed"),
        EngineParam("pitch", "Hauteur", "float", 0.0, min_val=-20.0, max_val=20.0,
                     role="pitch"),
    ],
    category="cloud",
))
