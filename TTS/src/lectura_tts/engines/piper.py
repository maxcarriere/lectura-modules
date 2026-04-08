"""Moteur TTS Piper — VITS ONNX avec durées phonèmes.

Pré-requis : pip install lectura-tts[piper]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lectura_tts.models import PhonemeTiming, TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_DEFAULT_MODEL = "fr_FR-siwis-medium"


class PiperTTSEngine:
    """Implémentation TTSEngine avec Piper (VITS ONNX + durées phonèmes)."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        models_dir: str = "",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        self._model_name = model_name
        self._models_dir = Path(models_dir) if models_dir else Path.home() / ".local/share/piper"
        self._length_scale = max(0.5, min(3.0, length_scale))
        self._noise_scale = noise_scale
        self._noise_w = noise_w
        self._voice = None

    def _ensure_loaded(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice

        model_path = self._models_dir / f"{self._model_name}.onnx"
        config_path = self._models_dir / f"{self._model_name}.onnx.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle Piper introuvable : {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config Piper introuvable : {config_path}")

        self._voice = PiperVoice.load(str(model_path), str(config_path))

    def synthesize(self, text: str) -> TTSResult:
        self._ensure_loaded()
        assert self._voice is not None
        from piper.voice import SynthesisConfig

        syn_config = SynthesisConfig(
            length_scale=self._length_scale,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w,
        )

        all_audio = []
        all_timings: list[PhonemeTiming] = []
        audio_offset_samples = 0

        for audio_chunk in self._voice.synthesize(
            text, syn_config=syn_config, include_alignments=True
        ):
            audio_float = audio_chunk.audio_float_array
            all_audio.append(audio_float)
            sr = audio_chunk.sample_rate

            if audio_chunk.phoneme_alignments:
                cursor_samples = audio_offset_samples
                for alignment in audio_chunk.phoneme_alignments:
                    phoneme = alignment.phoneme
                    num_samples = alignment.num_samples
                    if phoneme in ("^", "$", "_"):
                        cursor_samples += num_samples
                        continue
                    start_ms = cursor_samples / sr * 1000
                    end_ms = (cursor_samples + num_samples) / sr * 1000
                    all_timings.append(PhonemeTiming(
                        ipa=phoneme, start_ms=start_ms, end_ms=end_ms,
                    ))
                    cursor_samples += num_samples

            audio_offset_samples += len(audio_float)

        if not all_audio:
            return TTSResult(samples=np.array([], dtype=np.float32),
                             sample_rate=22050, phoneme_timings=[])

        samples = np.concatenate(all_audio)
        sample_rate = self._voice.config.sample_rate
        return TTSResult(samples=samples, sample_rate=sample_rate, phoneme_timings=all_timings)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        return self.synthesize(f"[[{phonemes_ipa}]]")


# ── Auto-enregistrement ──

def _check() -> bool:
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


register(EngineInfo(
    key="piper",
    name="Piper",
    description="Synthèse neuronale VITS avec alignement phonème natif.",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=False,
    requires_api_key=False,
    install_instructions="pip install lectura-tts[piper]",
    check_available=_check,
    factory=lambda p: PiperTTSEngine(**p),
    params=[
        EngineParam("model_name", "Modèle", "str", _DEFAULT_MODEL),
        EngineParam("length_scale", "Durée (0.5-3.0)", "float", 1.0,
                     min_val=0.5, max_val=3.0),
    ],
))
