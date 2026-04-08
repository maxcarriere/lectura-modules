"""Moteur TTS Kokoro-82M ONNX — transformer avec durées phonèmes.

Pré-requis : pip install lectura-tts[kokoro]
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from lectura_tts.models import PhonemeTiming, TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_VOICE = "ff_siwis"
_LANG = "fr-fr"
_SAMPLE_RATE = 24000
_FRAME_MS = 25.0
_DUR_NODE = "/encoder/Clip_output_0"


class KokoroTTSEngine:
    """Implémentation TTSEngine avec Kokoro-82M ONNX + durées phonèmes."""

    def __init__(
        self,
        model_path: str = "kokoro-v1.0.onnx",
        voices_path: str = "voices-v1.0.bin",
        voice: str = _VOICE,
        lang: str = _LANG,
        speed: float = 1.0,
    ) -> None:
        self._voice_name = voice
        self._lang = lang
        self._speed = max(0.5, min(2.0, speed))
        self._model_path = model_path
        self._voices_path = voices_path
        self._kokoro = None
        self._sess_dur = None

    def _ensure_loaded(self) -> None:
        if self._kokoro is not None:
            return
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(self._model_path, self._voices_path)
        self._sess_dur = _make_duration_session(self._model_path)

    def synthesize(self, text: str) -> TTSResult:
        self._ensure_loaded()
        assert self._kokoro is not None and self._sess_dur is not None

        phonemes_str = self._kokoro.tokenizer.phonemize(text, self._lang)
        return self._synth_from_phonemes(phonemes_str)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        self._ensure_loaded()
        assert self._kokoro is not None and self._sess_dur is not None
        return self._synth_from_phonemes(phonemes_ipa)

    def _synth_from_phonemes(self, phonemes_str: str) -> TTSResult:
        tokens = self._kokoro.tokenizer.tokenize(phonemes_str)
        voice_style = self._kokoro.get_voice_style(self._voice_name)
        voice_for_len = voice_style[len(tokens)]

        tokens_padded = np.array([[0] + tokens + [0]], dtype=np.int64)

        input_names = [i.name for i in self._sess_dur.get_inputs()]
        if "input_ids" in input_names:
            inputs = {
                "input_ids": tokens_padded,
                "style": np.array(voice_for_len, dtype=np.float32),
                "speed": np.array([self._speed], dtype=np.int32),
            }
        else:
            inputs = {
                "tokens": tokens_padded,
                "style": voice_for_len,
                "speed": np.array([self._speed], dtype=np.float32),
            }

        audio, dur_frames = self._sess_dur.run(None, inputs)
        dur_frames = dur_frames.flatten()

        padded_phonemes = ["_"] + list(phonemes_str) + ["_"]
        timings = _build_phoneme_timings(padded_phonemes, dur_frames)

        return TTSResult(
            samples=np.asarray(audio, dtype=np.float32),
            sample_rate=_SAMPLE_RATE,
            phoneme_timings=timings,
        )


def _make_duration_session(model_path: str):
    import onnx
    import onnxruntime as rt
    from onnx import TensorProto, helper

    model = onnx.load(model_path)
    existing_outputs = {o.name for o in model.graph.output}
    if _DUR_NODE not in existing_outputs:
        dur_output = helper.make_tensor_value_info(_DUR_NODE, TensorProto.FLOAT, None)
        model.graph.output.append(dur_output)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        patched_path = f.name

    sess = rt.InferenceSession(patched_path, providers=["CPUExecutionProvider"])
    Path(patched_path).unlink(missing_ok=True)
    return sess


def _build_phoneme_timings(
    padded_phonemes: list[str], dur_frames: np.ndarray,
) -> list[PhonemeTiming]:
    timings: list[PhonemeTiming] = []
    cursor_ms = 0.0
    for ph, frames in zip(padded_phonemes, dur_frames):
        dur_ms = float(frames) * _FRAME_MS
        if ph != "_":
            timings.append(PhonemeTiming(ipa=ph, start_ms=cursor_ms,
                                         end_ms=cursor_ms + dur_ms))
        cursor_ms += dur_ms
    return timings


# ── Auto-enregistrement ──

def _check() -> bool:
    try:
        import kokoro_onnx  # noqa: F401
        return True
    except ImportError:
        return False


register(EngineInfo(
    key="kokoro",
    name="Kokoro",
    description="Synthèse neuronale transformer 82M avec durées phonèmes.",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=False,
    requires_api_key=False,
    install_instructions="pip install lectura-tts[kokoro]",
    check_available=_check,
    factory=lambda p: KokoroTTSEngine(**p),
    params=[
        EngineParam("speed", "Vitesse (0.5-2.0)", "float", 1.0, min_val=0.5, max_val=2.0),
        EngineParam("voice", "Voix", "str", _VOICE),
    ],
))
