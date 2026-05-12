"""Router TTS Diphone — POST /tts-diphone/synthesize

Synthese vocale par concatenation de diphones WORLD via lectura-tts-diphone.
Accepte du texte ou des groupes prosodiques pre-phonemises.
Retourne audio base64 + sample_rate + duration.
"""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton engine TTS diphone
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from lectura_tts_diphone import creer_engine
        _engine = creer_engine(mode="local")
        logger.info("TTS diphone engine charge (mode local)")
    return _engine


class PhonesGroup(BaseModel):
    """Groupe prosodique avec phones IPA et type de frontiere."""
    phones: list[str] = Field(..., description="Phones IPA du groupe")
    boundary: str = Field("none", description="Type de frontiere: none, comma, period, question, exclamation, suspensive")
    word_boundaries: list[int] | None = Field(None, description="Indices de debut de mot (micro-pauses)")


class SynthesizeRequest(BaseModel):
    """Requete de synthese TTS diphone."""
    text: str | None = Field(None, description="Texte a synthetiser (necessite G2P)")
    groups: list[PhonesGroup] | None = Field(None, description="Groupes prosodiques pre-phonemises")
    mode: str = Field("FLUIDE", description="Mode de synthese: FLUIDE, MOT_A_MOT, SYLLABES")
    duration_scale: float = Field(1.0, gt=0.1, le=5.0, description="Vitesse (>1 = plus lent)")
    pause_scale: float = Field(1.0, gt=0.0, le=5.0, description="Facteur pauses inter-groupes")
    macro_expressivity: float = Field(1.0, ge=0.0, le=4.0, description="Gestes prosodiques (0=plat, 1=normal, 2=exagere)")
    micro_expressivity: float = Field(1.0, ge=0.0, le=4.0, description="Micro-variations (0=robot, 1=normal, 2=tres expressif)")
    spectral_contrast: float = Field(1.3, ge=1.0, le=3.0, description="Contraste spectral GV (1.0=off, 2.0=fort)")
    ap_cleanup: float = Field(1.5, ge=1.0, le=3.0, description="Compression AP (1.0=off, 1.5=defaut). Reduit la raucite")
    formant_sharpening: float = Field(1.3, ge=1.0, le=2.0, description="Affutage formants (1.0=off, 1.3=defaut). Nettete spectrale")
    vtln_alpha: float = Field(1.0, ge=0.8, le=1.2, description="Warping VTLN (0.8=grave, 1.0=neutre, 1.2=aigu)")
    prosody_style: str = Field("auto", description="Style prosodique: auto, declaratif, question, exclamation, suspensif, neutre")
    seed: int | None = Field(None, description="Graine aleatoire pour micro-prosodie reproductible")


class SynthesizeResponse(BaseModel):
    """Reponse de synthese TTS diphone."""
    audio_base64: str
    sample_rate: int
    duration_s: float


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest):
    """Synthetise du texte ou des groupes prosodiques en audio.

    Retourne l'audio en base64 (float32 PCM) + sample_rate + duration.
    """
    if req.text is None and req.groups is None:
        raise HTTPException(status_code=400, detail="text ou groups requis")

    engine = _get_engine()

    synth_kwargs = dict(
        mode=req.mode,
        duration_scale=req.duration_scale,
        pause_scale=req.pause_scale,
        macro_expressivity=req.macro_expressivity,
        micro_expressivity=req.micro_expressivity,
        spectral_contrast=req.spectral_contrast,
        ap_cleanup=req.ap_cleanup,
        formant_sharpening=req.formant_sharpening,
        vtln_alpha=req.vtln_alpha,
        prosody_style=req.prosody_style,
        seed=req.seed,
    )

    if req.groups is not None:
        # Groupes pre-phonemises
        groups = [g.model_dump(exclude_none=True) for g in req.groups]
        audio = engine.synthesize_groups(groups, **synth_kwargs)
    else:
        # Texte : passer par le G2P (auto-detecte par le module)
        groups = engine._g2p_backend.phonemize(req.text)
        if not groups:
            raise HTTPException(status_code=400, detail="G2P n'a produit aucun groupe")
        audio = engine.synthesize_groups(groups, **synth_kwargs)

    # Encoder audio en base64
    audio_bytes = audio.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    sample_rate = 44100
    duration_s = len(audio) / sample_rate

    return SynthesizeResponse(
        audio_base64=audio_b64,
        sample_rate=sample_rate,
        duration_s=duration_s,
    )
