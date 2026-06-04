"""Backend API serveur pour le modele CTC.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later
"""

from __future__ import annotations

import base64
import io
import json
import logging
import struct

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.lec-tu-ra.com"


def _audio_vers_wav_base64(audio: np.ndarray, sr: int) -> str:
    """Encode un signal audio float32 en WAV base64 (PCM 16 bits)."""
    audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    n_samples = len(audio_int16)
    data_size = n_samples * 2  # 16 bits = 2 octets/sample
    # Header WAV
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))        # chunk size
    buf.write(struct.pack("<H", 1))         # PCM
    buf.write(struct.pack("<H", 1))         # mono
    buf.write(struct.pack("<I", sr))        # sample rate
    buf.write(struct.pack("<I", sr * 2))    # byte rate
    buf.write(struct.pack("<H", 2))         # block align
    buf.write(struct.pack("<H", 16))        # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ApiCTCEngine:
    """Inference CTC via l'API serveur Lectura.

    Parameters
    ----------
    api_url : str
        URL de base du serveur (ex: ``https://api.lec-tu-ra.com``).
    api_key : str | None
        Cle API optionnelle.
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.api_url = (api_url or _DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key

    def transcrire(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Transcrit un signal audio en chaine IPA via l'API.

        Parameters
        ----------
        audio : np.ndarray
            Signal PCM float32 mono, shape (N,).
        sr : int
            Sample rate.

        Returns
        -------
        str
            Chaine IPA.
        """
        import urllib.request

        audio_b64 = _audio_vers_wav_base64(audio, sr)
        payload = json.dumps({"audio_base64": audio_b64, "sample_rate": sr}).encode()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.api_url}/ctc/transcribe",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        return data["ipa"]

    def transcrire_batch(
        self, audios: list[np.ndarray], sr: int = 16000,
    ) -> list[str]:
        """Transcrit un batch d'audios via l'API (sequentiel)."""
        return [self.transcrire(audio, sr) for audio in audios]

    def __repr__(self) -> str:
        return f"ApiCTCEngine(url={self.api_url!r})"
