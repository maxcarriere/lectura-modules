"""Moteur TTS Lectura Mono — FastPitch-Lite + HiFi-GAN (ONNX).

Inference neuronale locale via lectura-tts-monospeaker.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lectura_tts.models import PhonemeTiming, TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

log = logging.getLogger(__name__)


class MonospeakerTTSEngine:
    """Wrapper lectura_tts pour le moteur TTS monospeaker ONNX."""

    def __init__(
        self,
        models_dir: str = "",
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
        phrase_type: int | str = "Auto",
        variability: str = "off",
        seed: str = "",
        determinism: float = 0.5,
    ) -> None:
        self._models_dir = models_dir or None
        self._duration_scale = duration_scale
        self._pitch_shift = pitch_shift
        self._pitch_range = pitch_range
        self._energy_scale = energy_scale
        self._pause_scale = pause_scale
        if phrase_type in (None, "Auto"):
            self._phrase_type = None
        else:
            self._phrase_type = int(str(phrase_type)[0])
        self._seed: int | None = int(seed) if str(seed).strip() else None
        self._determinism = float(determinism)
        # Derive variability from determinism (backward compat)
        if self._determinism < 0.5:
            self._variability = True
        else:
            self._variability = str(variability).lower() in ("on", "true", "1")
        self._engine = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from lectura_tts_monospeaker import creer_engine
        self._engine = creer_engine(mode="local", models_dir=self._models_dir)

    def synthesize(self, text: str) -> TTSResult:
        """Synthetise du texte (necessite lectura-g2p)."""
        self._ensure_loaded()
        result = self._engine.synthesize(
            text,
            phrase_type=self._phrase_type,
            duration_scale=self._duration_scale,
            pitch_shift=self._pitch_shift,
            pitch_range=self._pitch_range,
            energy_scale=self._energy_scale,
            pause_scale=self._pause_scale,
            variability=self._variability,
        )
        return self._convert_result(result)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        """Synthetise des phonemes IPA."""
        self._ensure_loaded()
        result = self._engine.synthesize_phonemes(
            phonemes_ipa,
            phrase_type=self._phrase_type if self._phrase_type is not None else 0,
            duration_scale=self._duration_scale,
            pitch_shift=self._pitch_shift,
            pitch_range=self._pitch_range,
            energy_scale=self._energy_scale,
            pause_scale=self._pause_scale,
            variability=self._variability,
        )
        return self._convert_result(result)

    def _convert_result(self, result) -> TTSResult:
        """Convertit un TTSResult du module mono vers lectura_tts.TTSResult."""
        timings = [
            PhonemeTiming(ipa=t.ipa, start_ms=t.start_ms, end_ms=t.end_ms)
            for t in result.phoneme_timings
        ]
        return TTSResult(
            samples=result.samples,
            sample_rate=result.sample_rate,
            phoneme_timings=timings,
        )


# ── Auto-enregistrement ──

def _check() -> bool:
    """Verifie si lectura-tts-monospeaker + onnxruntime sont disponibles."""
    try:
        import lectura_tts_monospeaker  # noqa: F401
        import onnxruntime  # noqa: F401
        from lectura_tts_monospeaker._chargeur import find_models_dir
        return find_models_dir() is not None
    except ImportError:
        return False


register(EngineInfo(
    key="lectura-mono",
    name="MonoSpeaker",
    description="Synthese neuronale monospeaker francais — FastPitch + HiFi-GAN (ONNX).",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=False,
    requires_api_key=False,
    install_instructions="pip install lectura-tts-monospeaker[all]",
    check_available=_check,
    factory=lambda p: MonospeakerTTSEngine(**p),
    params=[
        EngineParam("phrase_type", "Type de phrase", "choice", "Auto",
                    choices=["Auto",
                             "0 (declaratif)", "1 (interrogatif)",
                             "2 (exclamatif)", "3 (suspensif)"],
                    role="voice"),
        EngineParam("duration_scale", "Vitesse", "float", 1.0,
                    min_val=0.5, max_val=3.0, role="speed"),
        EngineParam("pitch_shift", "Hauteur (demi-tons)", "float", 0.0,
                    min_val=-12.0, max_val=12.0, role="pitch"),
        EngineParam("determinism", "Determinisme", "float", 0.5,
                    min_val=0.0, max_val=1.0, role="determinism"),
        EngineParam("seed", "Graine", "str", "", role="seed"),
        EngineParam("pitch_range", "Variation F0", "float", 1.3,
                    min_val=0.5, max_val=3.0, role="pitch_range"),
        EngineParam("energy_scale", "Intensite", "float", 1.0,
                    min_val=0.5, max_val=2.0, role="energy"),
        EngineParam("pause_scale", "Pauses", "float", 1.0,
                    min_val=0.5, max_val=3.0, role="pause"),
        EngineParam("variability", "Variabilite", "choice", "off",
                    choices=["off", "on"], role="variability"),
    ],
    category="builtin",
    pip_packages=["lectura-tts-monospeaker[all]"],
    check_modules=["lectura_tts_monospeaker", "onnxruntime"],
))
