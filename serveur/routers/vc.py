"""Router VC -- POST /vc/convert, GET /vc/speakers

Conversion vocale via lectura-vc.
Accepte un fichier audio en multipart/form-data + parametres de conversion.
Retourne audio base64 (WAV) + metadonnees.
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton engine VC
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from lectura_vc import creer_engine
        _engine = creer_engine(mode="auto")
        logger.info("VC engine charge (mode auto)")
    return _engine


class ConvertResponse(BaseModel):
    """Reponse de conversion vocale."""
    audio_base64: str
    sample_rate: int
    duration_s: float
    speaker: str | None
    mode: str


async def _read_audio_upload(upload: UploadFile) -> tuple[np.ndarray, int]:
    """Lit un fichier audio uploade et retourne (samples, sr)."""
    content = await upload.read()
    try:
        audio_np, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
    except Exception:
        # Fallback via fichier temporaire pour formats non supportes par soundfile
        import librosa
        with tempfile.NamedTemporaryFile(suffix=Path(upload.filename or "audio.wav").suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            audio_np, sr = librosa.load(tmp.name, sr=None, mono=True)
            audio_np = audio_np.astype(np.float32)
    # Mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    return audio_np, sr


def _encode_wav_base64(audio: np.ndarray, sr: int) -> str:
    """Encode un array audio en WAV base64."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


@router.post("/convert", response_model=ConvertResponse)
async def convert(
    audio: UploadFile = File(..., description="Fichier audio source (WAV, MP3, etc.)"),
    speaker: str | None = Form(None, description="Speaker RVC cible (ezwa, nadine, bernard, gilles, zeckou, siwis)"),
    mode: str = Form("auto", description="Mode de conversion (rvc, zeroshot, cascade, auto)"),
    reference: UploadFile | None = File(None, description="Audio de reference pour zero-shot"),
    protect: float | None = Form(None, description="Facteur de protection voix (0.0-0.5)"),
    pitch_modification: float | None = Form(None, description="Shift en demi-tons"),
    tau: float = Form(0.3, description="Parametre OpenVoice (0 = deterministe)"),
):
    """Convertit un audio vers la voix cible.

    Accepte multipart/form-data avec le fichier audio source et les parametres.
    Retourne l'audio converti en WAV base64.
    """
    # Valider qu'on a au moins speaker ou reference
    if speaker is None and (reference is None or reference.filename is None or reference.filename == ""):
        raise HTTPException(
            status_code=400,
            detail="Au moins 'speaker' ou 'reference' est requis.",
        )

    engine = _get_engine()

    # Lire l'audio source
    try:
        audio_np, sr_in = await _read_audio_upload(audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture audio: {e}")

    # Lire la reference si fournie
    ref_audio = None
    if reference is not None and reference.filename and reference.filename != "":
        try:
            ref_np, ref_sr = await _read_audio_upload(reference)
            # Sauver en fichier temporaire (l'engine attend un path ou array+sr)
            ref_audio = ref_np
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur lecture reference: {e}")

    # Construire les kwargs de conversion
    kwargs = {}
    if protect is not None:
        kwargs["protect"] = protect
    if pitch_modification is not None:
        kwargs["pitch_modification"] = pitch_modification
    kwargs["tau"] = tau

    try:
        result_audio, result_sr = engine.convert(
            audio=audio_np,
            speaker=speaker,
            reference=ref_audio,
            mode=mode if mode != "auto" else None,
            sr_in=sr_in,
            **kwargs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Encoder le resultat en WAV base64
    audio_b64 = _encode_wav_base64(result_audio, result_sr)
    duration_s = len(result_audio) / result_sr

    return ConvertResponse(
        audio_base64=audio_b64,
        sample_rate=result_sr,
        duration_s=duration_s,
        speaker=speaker,
        mode=mode,
    )


@router.get("/speakers")
async def speakers():
    """Retourne la liste des speakers RVC disponibles."""
    from lectura_vc._chargeur import RVC_SPEAKERS
    return RVC_SPEAKERS
