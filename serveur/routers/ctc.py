"""Router CTC — POST /ctc/transcribe

Transcription phonetique audio → phones IPA via le modele CTC.
Accepte de l'audio en base64 ou multipart (tout format : WAV, webm, mp3, ogg...).
Conversion automatique via ffmpeg vers PCM 16kHz mono.
Retourne la transcription phonetique IPA.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton engine CTC
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from lectura_ctc import creer_engine
        _engine = creer_engine(mode="onnx")
        logger.info("CTC engine charge (mode ONNX)")
    return _engine


def _decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode n'importe quel format audio en array float32 mono 16kHz via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f_in:
        f_in.write(audio_bytes)
        in_path = f_in.name

    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", in_path,
                "-ar", "16000",     # resample 16kHz
                "-ac", "1",         # mono
                "-f", "s16le",      # PCM 16 bits little-endian brut
                "-acodec", "pcm_s16le",
                out_path,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-500:]
            raise ValueError(f"ffmpeg erreur : {stderr}")

        raw = Path(out_path).read_bytes()
        if len(raw) == 0:
            raise ValueError("Audio vide apres conversion")

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 16000

    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


class TranscribeRequest(BaseModel):
    """Requete de transcription CTC."""
    audio_base64: str = Field(..., description="Audio encode en base64 (tout format)")
    sample_rate: int = Field(16000, description="Sample rate de l'audio")


class TranscribeResponse(BaseModel):
    """Reponse de transcription CTC."""
    ipa: str = Field(..., description="Transcription phonetique IPA")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    """Transcrit un audio en phonemes IPA via le modele CTC.

    Accepte tout format audio encode en base64 (WAV, webm, mp3, ogg...).
    """
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="audio_base64 invalide")

    try:
        audio, sr = _decode_audio_bytes(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine = _get_engine()
    ipa = engine.transcrire(audio, sr=sr)
    return TranscribeResponse(ipa=ipa)


@router.post("/transcribe-file", response_model=TranscribeResponse)
async def transcribe_file(file: UploadFile = File(...)):
    """Transcrit un fichier audio en phonemes IPA.

    Accepte tout format audio (WAV, webm, mp3, ogg, flac...).
    """
    content = await file.read()

    try:
        audio, sr = _decode_audio_bytes(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine = _get_engine()
    ipa = engine.transcrire(audio, sr=sr)
    return TranscribeResponse(ipa=ipa)
