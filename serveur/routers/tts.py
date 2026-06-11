"""Router TTS — POST /tts/synthesize

Synthese vocale via lectura-tts-monospeaker.
Accepte du texte ou des phonemes IPA + parametres prosodiques.
Retourne audio base64 + sample_rate + phoneme_timings.
"""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton engine TTS
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from lectura_tts_monospeaker import creer_engine
        _engine = creer_engine(mode="local")
        logger.info("TTS engine charge (mode local)")
    return _engine


class SynthesizeRequest(BaseModel):
    """Requete de synthese TTS."""
    text: str | None = Field(None, description="Texte a synthetiser (necessite G2P)")
    ipa: str | None = Field(None, description="Phonemes IPA a synthetiser")
    phrase_type: int | None = Field(None, ge=0, le=3, description="0=decl, 1=inter, 2=excl, 3=susp (null=auto)")
    duration_scale: float = Field(1.0, gt=0.1, le=5.0)
    pitch_shift: float = Field(0.0, ge=-12.0, le=12.0)
    pitch_range: float = Field(1.3, gt=0.0, le=5.0)
    energy_scale: float = Field(1.0, gt=0.0, le=3.0)
    pause_scale: float = Field(1.0, gt=0.0, le=5.0)
    voix: str | None = Field(None, description="Preset de voix pour retimbre OpenVoice (siwis, ezwa, nadine, bernard, gilles, zeckou)")
    voix_variante: float = Field(0.0, ge=-1.0, le=1.0, description="Variante formants (-1=grave, 0=neutre, +1=aigu)")


class PhonemeTimingResponse(BaseModel):
    ipa: str
    start_ms: float
    end_ms: float


class SynthesizeResponse(BaseModel):
    """Reponse de synthese TTS."""
    audio_base64: str
    sample_rate: int
    duration_s: float
    phoneme_timings: list[PhonemeTimingResponse]


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest):
    """Synthetise du texte ou des phonemes IPA en audio.

    Retourne l'audio en base64 (float32 PCM) + sample_rate + timings.
    """
    if req.text is None and req.ipa is None:
        raise HTTPException(status_code=400, detail="text ou ipa requis")

    engine = _get_engine()

    # Parametres prosodiques communs
    prosody = dict(
        duration_scale=req.duration_scale,
        pitch_shift=req.pitch_shift,
        pitch_range=req.pitch_range,
        energy_scale=req.energy_scale,
        pause_scale=req.pause_scale,
    )

    if req.ipa is not None:
        result = engine.synthesize_phonemes(
            req.ipa,
            phrase_type=req.phrase_type or 0,
            **prosody,
        )
        # Retimbre en post si demande (synthesize_phonemes ne le fait pas)
        if req.voix:
            result = engine._apply_retimbre(
                result, req.voix, req.voix_variante, 0.3, None,
            )
    else:
        result = engine.synthesize(
            req.text,
            phrase_type=req.phrase_type,
            voix=req.voix,
            voix_variante=req.voix_variante,
            **prosody,
        )

    # Encoder audio en base64
    audio_bytes = result.samples.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    timings = [
        PhonemeTimingResponse(ipa=t.ipa, start_ms=t.start_ms, end_ms=t.end_ms)
        for t in result.phoneme_timings
    ]

    duration_s = len(result.samples) / result.sample_rate

    return SynthesizeResponse(
        audio_base64=audio_b64,
        sample_rate=result.sample_rate,
        duration_s=duration_s,
        phoneme_timings=timings,
    )
