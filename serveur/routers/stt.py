"""Router STT — POST /stt/transcribe

Pipeline STT complet : audio → phones IPA + texte orthographique.
Combine le decodeur CTC avec le pipeline P2G.
Accepte de l'audio en base64 ou multipart (tout format : WAV, webm, mp3, ogg...).
Conversion automatique via ffmpeg vers PCM 16kHz mono.
"""

from __future__ import annotations

import base64
import logging
import struct
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton engine STT (standard)
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from lectura_stt import creer_engine
        _engine = creer_engine(mode="onnx")
        p2g_status = type(_engine.p2g).__name__ if _engine.p2g else "None"
        logger.info("STT engine charge (CTC=ONNX, P2G=%s)", p2g_status)
    return _engine


# Singleton engine FormulaCTC (STT-Formules, vocabulaire semantique 87 tokens)
_engine_formules = None


def _get_engine_formules():
    global _engine_formules
    if _engine_formules is None:
        from lectura_stt_formules import creer_engine
        _engine_formules = creer_engine()
        logger.info("STT-Formules engine charge (FormulaCTC ONNX)")
    return _engine_formules


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
                "-ar", "16000",
                "-ac", "1",
                "-f", "s16le",
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
    """Requete de transcription STT."""
    audio_base64: str = Field(..., description="Audio encode en base64 (tout format)")
    sample_rate: int = Field(16000, description="Sample rate de l'audio")


class TranscribeResponse(BaseModel):
    """Reponse de transcription STT."""
    ipa: str = Field(..., description="Transcription phonetique IPA")
    texte: str | None = Field(None, description="Texte orthographique (None si P2G absent)")
    mots_ipa: list[str] = Field(default_factory=list, description="Mots IPA")
    mots: list[str] | None = Field(None, description="Mots orthographiques")


class TranscribeFormuleResponse(BaseModel):
    """Reponse de transcription STT-Formules (vocabulaire semantique)."""
    tokens: list[int] = Field(..., description="Token IDs semantiques")
    names: list[str] = Field(..., description="Noms des tokens")
    texte: str = Field(..., description="Noms joints (espace)")


def _write_wav(samples: np.ndarray, sr: int, path: str) -> None:
    """Ecrit un array float32 mono en fichier WAV PCM 16-bit."""
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    n_bytes = pcm.nbytes
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n_bytes))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", n_bytes))
        f.write(pcm.tobytes())


def _transcribe_formules(audio: np.ndarray, sr: int) -> TranscribeFormuleResponse:
    """Transcrit un audio via FormulaCTC et retourne la reponse formules."""
    engine = _get_engine_formules()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        _write_wav(audio, sr, wav_path)
        result = engine.transcrire(wav_path)
    finally:
        Path(wav_path).unlink(missing_ok=True)

    # Filtrer les tokens de controle pour le texte lisible
    names_clean = [n for n in result["names"] if not n.startswith("<")]
    return TranscribeFormuleResponse(
        tokens=result["tokens"],
        names=result["names"],
        texte=" ".join(names_clean),
    )


@router.post("/transcribe", response_model=TranscribeResponse | TranscribeFormuleResponse)
async def transcribe(
    req: TranscribeRequest,
    mode: str = Query("auto", description="Mode : auto, formule, texte, ipa, formules"),
):
    """Transcrit un audio en texte via le pipeline STT (CTC + P2G).

    Accepte tout format audio encode en base64 (WAV, webm, mp3, ogg...).

    Modes :
        - auto : pipeline STT complet, formules courantes (defaut)
        - formule : detection maximale (math, symboles, nombres sans garde)
        - texte : texte brut (sigles uniquement)
        - ipa : CTC seul (retourne uniquement les phones IPA)
        - formules : FormulaCTC legacy (tokens semantiques)
    """
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="audio_base64 invalide")

    try:
        audio, sr = _decode_audio_bytes(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Modes legacy (inchanges)
    if mode == "formules":
        return _transcribe_formules(audio, sr)

    engine = _get_engine()

    if mode == "ipa":
        ipa = engine.ctc.transcrire(audio, sr=sr)
        from lectura_stt._parse_ctc import parse_ctc_output
        parsed = parse_ctc_output(ipa)
        return TranscribeResponse(
            ipa=ipa,
            texte=None,
            mots_ipa=parsed.mots_ipa,
            mots=None,
        )

    # Modes CTC+P2G (auto/formule/texte)
    stt_mode = mode if mode in ("auto", "formule", "texte") else "auto"
    result = engine.transcrire(audio, sr=sr, stt_mode=stt_mode)
    return TranscribeResponse(
        ipa=result.ipa,
        texte=result.texte,
        mots_ipa=result.mots_ipa,
        mots=result.mots,
    )


@router.post("/transcribe-file", response_model=TranscribeResponse | TranscribeFormuleResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    mode: str = Query("auto", description="Mode : auto, formule, texte, ipa, formules"),
):
    """Transcrit un fichier audio en texte via le pipeline STT.

    Accepte tout format audio (WAV, webm, mp3, ogg, flac...).

    Modes :
        - auto : pipeline STT complet, formules courantes (defaut)
        - formule : detection maximale (math, symboles, nombres sans garde)
        - texte : texte brut (sigles uniquement)
        - ipa : CTC seul (retourne uniquement les phones IPA)
        - formules : FormulaCTC legacy (tokens semantiques)
    """
    content = await file.read()

    try:
        audio, sr = _decode_audio_bytes(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Modes legacy (inchanges)
    if mode == "formules":
        return _transcribe_formules(audio, sr)

    engine = _get_engine()

    if mode == "ipa":
        # CTC seul — pas de P2G
        ipa = engine.ctc.transcrire(audio, sr=sr)
        from lectura_stt._parse_ctc import parse_ctc_output
        parsed = parse_ctc_output(ipa)
        return TranscribeResponse(
            ipa=ipa,
            texte=None,
            mots_ipa=parsed.mots_ipa,
            mots=None,
        )

    # Modes CTC+P2G (auto/formule/texte)
    stt_mode = mode if mode in ("auto", "formule", "texte") else "auto"
    result = engine.transcrire(audio, sr=sr, stt_mode=stt_mode)
    return TranscribeResponse(
        ipa=result.ipa,
        texte=result.texte,
        mots_ipa=result.mots_ipa,
        mots=result.mots,
    )
