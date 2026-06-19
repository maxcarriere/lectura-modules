"""Router TTS — POST /tts/synthesize

Synthese vocale via lectura-tts-monospeaker.
Accepte du texte ou des phonemes IPA + parametres prosodiques.
Supporte les style presets et le controle ODE (Matcha-Conformer).
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
        from lectura_monospeaker import creer_engine
        _engine = creer_engine(mode="local")
        logger.info("TTS engine charge (mode local)")
    return _engine


class SynthesizeRequest(BaseModel):
    """Requete de synthese TTS."""
    text: str | None = Field(None, description="Texte a synthetiser (necessite G2P)")
    ipa: str | None = Field(None, description="Phonemes IPA a synthetiser")
    phrase_type: int | None = Field(None, ge=0, le=3, description="0=decl, 1=inter, 2=excl, 3=susp (null=auto)")
    duration_scale: float | None = Field(None, gt=0.1, le=5.0)
    pitch_shift: float | None = Field(None, ge=-12.0, le=12.0)
    pitch_range: float | None = Field(None, gt=0.0, le=5.0)
    energy_scale: float | None = Field(None, gt=0.0, le=3.0)
    pause_scale: float | None = Field(None, gt=0.0, le=5.0)
    duration_noise: float | None = Field(None, ge=0.0, le=1.0, description="Bruit de duree lisse (0=off, ~0.1=subtil, ~0.2=prononce)")
    style: str | None = Field(None, description="Preset de style (neutre, narratif, dialogue, expressif, meditatif, rapide, lent)")
    style_vector: list[float] | None = Field(None, description="Vecteur style [5 dims]")
    n_ode_steps: int | None = Field(None, ge=1, le=50, description="Pas ODE Matcha (defaut=4, plus=meilleur)")
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
        duration_noise=req.duration_noise,
    )

    # Parametres de style
    style_kwargs = {}
    if req.style is not None:
        style_kwargs["style"] = req.style
    if req.style_vector is not None:
        style_kwargs["style_vector"] = req.style_vector
    if req.n_ode_steps is not None:
        style_kwargs["n_ode_steps"] = req.n_ode_steps

    if req.ipa is not None:
        result = engine.synthesize_phonemes(
            req.ipa,
            phrase_type=req.phrase_type or 0,
            **prosody,
            **style_kwargs,
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
            **style_kwargs,
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
