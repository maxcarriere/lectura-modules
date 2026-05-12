"""Moteur TTS Kokoro-82M ONNX — inférence directe (sans kokoro-onnx GPL).

Charge un modèle Kokoro (.onnx + voices .bin) et fait l'inférence
directement via onnxruntime (MIT). Aucune dépendance GPL.

Pré-requis : onnxruntime (déjà disponible si Piper est installé)
Modèles : téléchargés depuis https://github.com/thewh1teagle/kokoro-onnx
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from lectura_tts.models import PhonemeTiming, TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

log = logging.getLogger(__name__)

_VOICE = "ff_siwis"
_SAMPLE_RATE = 24000
_FRAME_MS = 25.0
_MAX_PHONEME_LENGTH = 510
_DUR_NODE = "/encoder/Clip_output_0"
_MODELS_DIR = Path.home() / ".local/share/kokoro"

# ── Vocabulaire complet Kokoro v1.0 (115 tokens sur 178 indices) ──
# Source : kokoro-onnx config.json — licence MIT
# Token 0 réservé pour le padding/BOS/EOS
_VOCAB: dict[str, int] = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6,
    "\u2014": 9, "\u2026": 10, '"': 11, "(": 12, ")": 13,
    "\u201c": 14, "\u201d": 15,
    " ": 16,
    "\u0303": 17,       # combining tilde (nasales : ɔ̃ = ɔ + \u0303)
    "\u02a3": 18, "\u02a5": 19, "\u02a6": 20, "\u02a8": 21,
    "\u1d5d": 22, "\uab67": 23,
    "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35, "T": 36,
    "W": 39, "Y": 41, "\u1d4a": 42,
    "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48,
    "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55,
    "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61,
    "t": 62, "u": 63, "v": 64, "w": 65, "x": 66, "y": 67,
    "z": 68,
    "\u0251": 69,   # ɑ
    "\u0250": 70,   # ɐ
    "\u0252": 71,   # ɒ
    "\u00e6": 72,   # æ
    "\u03b2": 75,   # β
    "\u0254": 76,   # ɔ
    "\u0255": 77,   # ɕ
    "\u00e7": 78,   # ç
    "\u0256": 80,   # ɖ
    "\u00f0": 81,   # ð
    "\u02a4": 82,   # ʤ
    "\u0259": 83,   # ə
    "\u025a": 85,   # ɚ
    "\u025b": 86,   # ɛ
    "\u025c": 87,   # ɜ
    "\u025f": 90,   # ɟ
    "\u0261": 92,   # ɡ
    "\u0265": 99,   # ɥ
    "\u0268": 101,  # ɨ
    "\u026a": 102,  # ɪ
    "\u029d": 103,  # ʝ
    "\u026f": 110,  # ɯ
    "\u0270": 111,  # ɰ
    "\u014b": 112,  # ŋ
    "\u0273": 113,  # ɳ
    "\u0272": 114,  # ɲ
    "\u0274": 115,  # ɴ
    "\u00f8": 116,  # ø
    "\u0278": 118,  # ɸ
    "\u03b8": 119,  # θ
    "\u0153": 120,  # œ
    "\u0279": 123,  # ɹ
    "\u027e": 125,  # ɾ
    "\u027b": 126,  # ɻ
    "\u0281": 128,  # ʁ
    "\u027d": 129,  # ɽ
    "\u0282": 130,  # ʂ
    "\u0283": 131,  # ʃ
    "\u0288": 132,  # ʈ
    "\u02a7": 133,  # ʧ
    "\u028a": 135,  # ʊ
    "\u028b": 136,  # ʋ
    "\u028c": 138,  # ʌ
    "\u0263": 139,  # ɣ
    "\u0264": 140,  # ɤ
    "\u03c7": 142,  # χ
    "\u028e": 143,  # ʎ
    "\u0292": 147,  # ʒ
    "\u0294": 148,  # ʔ
    "\u02c8": 156,  # ˈ (accent primaire)
    "\u02cc": 157,  # ˌ (accent secondaire)
    "\u02d0": 158,  # ː (voyelle longue)
    "\u02b0": 162,  # ʰ (aspiration)
    "\u02b2": 164,  # ʲ (palatalisation)
    "\u2193": 169,  # ↓
    "\u2192": 171,  # →
    "\u2197": 172,  # ↗
    "\u2198": 173,  # ↘
    "\u1d7b": 177,  # ᵻ
}


def _tokenize(phonemes: str) -> list[int]:
    """Convertit une chaîne de phonèmes IPA en IDs de tokens.

    Chaque caractère Unicode est mappé individuellement.
    Les combining marks (ex. ̃ dans ɔ̃) sont des tokens séparés.
    """
    return [_VOCAB[c] for c in phonemes if c in _VOCAB]


def _split_phonemes(phonemes: str) -> list[str]:
    """Découpe les séquences de phonèmes longues aux ponctuations."""
    if len(phonemes) <= _MAX_PHONEME_LENGTH:
        return [phonemes]

    batches: list[str] = []
    current = ""
    for ch in phonemes:
        current += ch
        if ch in ".!?;:" and len(current) >= _MAX_PHONEME_LENGTH // 2:
            batches.append(current)
            current = ""
    if current:
        batches.append(current)
    return batches


def _trim_silence(
    audio: np.ndarray,
    top_db: float = 60.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, int]:
    """Supprime le silence en début et fin d'audio (algorithme RMS).

    Retourne (audio_trimé, samples_supprimés_au_début).
    """
    if len(audio) < frame_length:
        return audio, 0

    n_frames = 1 + (len(audio) - frame_length) // hop_length
    energy = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + frame_length]
        energy[i] = np.sqrt(np.mean(frame.astype(np.float64) ** 2))

    ref = np.max(energy)
    if ref < 1e-10:
        return audio, 0
    energy_db = 20.0 * np.log10(np.maximum(energy, 1e-10) / ref)

    non_silent = np.where(energy_db > -top_db)[0]
    if len(non_silent) == 0:
        return audio, 0

    start_sample = non_silent[0] * hop_length
    end_sample = min(non_silent[-1] * hop_length + frame_length, len(audio))
    return audio[start_sample:end_sample], start_sample


def _make_duration_session(model_path: str):
    """Crée une session ONNX avec sortie durées (nécessite le package onnx)."""
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
            timings.append(PhonemeTiming(
                ipa=ph, start_ms=cursor_ms, end_ms=cursor_ms + dur_ms,
            ))
        cursor_ms += dur_ms
    return timings


class KokoroTTSEngine:
    """Inférence Kokoro-82M ONNX directe — zéro dépendance GPL."""

    def __init__(
        self,
        model_path: str = str(_MODELS_DIR / "kokoro-v1.0.onnx"),
        voices_path: str = str(_MODELS_DIR / "voices-v1.0.bin"),
        voice: str = _VOICE,
        speed: float = 1.0,
    ) -> None:
        self._voice_name = voice
        self._speed = max(0.5, min(2.0, speed))
        self._model_path = model_path
        self._voices_path = voices_path
        # Lazy-loaded
        self._session = None
        self._voices = None
        self._has_durations = False

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import onnxruntime

        model_path = Path(self._model_path)
        voices_path = Path(self._voices_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle Kokoro introuvable : {model_path}")
        if not voices_path.exists():
            raise FileNotFoundError(f"Voix Kokoro introuvables : {voices_path}")

        # Charger les voix (fichier numpy npz)
        self._voices = np.load(str(voices_path))

        # Essayer de créer la session avec durées (nécessite le package onnx)
        try:
            self._session = _make_duration_session(str(model_path))
            self._has_durations = True
            log.debug("Kokoro : session avec durées phonèmes activée")
        except ImportError:
            self._session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"],
            )
            self._has_durations = False
            log.debug("Kokoro : session audio seule (package onnx non disponible)")

    # ── API publique ──

    def synthesize(self, text: str) -> TTSResult:
        return self.synthesize_phonemes(text)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        self._ensure_loaded()
        assert self._session is not None and self._voices is not None

        batches = _split_phonemes(phonemes_ipa)
        all_audio: list[np.ndarray] = []
        all_timings: list[PhonemeTiming] = []
        time_offset_ms = 0.0

        for batch in batches:
            audio, timings = self._synth_batch(batch)
            if len(audio) == 0:
                continue

            audio, trim_start = _trim_silence(audio)

            # Décaler les timings : soustraire le silence initial trimé
            trim_offset_ms = trim_start / _SAMPLE_RATE * 1000.0
            for t in timings:
                s = t.start_ms - trim_offset_ms + time_offset_ms
                e = t.end_ms - trim_offset_ms + time_offset_ms
                if e > 0:
                    all_timings.append(PhonemeTiming(
                        ipa=t.ipa,
                        start_ms=max(0.0, s),
                        end_ms=e,
                    ))
            time_offset_ms += len(audio) / _SAMPLE_RATE * 1000.0
            all_audio.append(audio)

        if not all_audio:
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=_SAMPLE_RATE,
                phoneme_timings=[],
            )

        audio = np.concatenate(all_audio).astype(np.float32)
        return TTSResult(
            samples=audio,
            sample_rate=_SAMPLE_RATE,
            phoneme_timings=all_timings,
        )

    def _synth_batch(
        self, phonemes_str: str,
    ) -> tuple[np.ndarray, list[PhonemeTiming]]:
        """Synthétise un seul batch de phonèmes."""
        tokens = _tokenize(phonemes_str)
        if not tokens:
            return np.array([], dtype=np.float32), []

        # Récupérer le style vocal indexé par la longueur des tokens
        voice_style = self._voices[self._voice_name]
        voice_for_len = voice_style[len(tokens)]

        # Tokens paddés : [0, tok1, tok2, ..., 0]
        tokens_padded = np.array([[0] + tokens + [0]], dtype=np.int64)

        # Détecter le format des entrées du modèle
        input_names = [inp.name for inp in self._session.get_inputs()]
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
                "speed": np.ones(1, dtype=np.float32) * self._speed,
            }

        result = self._session.run(None, inputs)
        audio = result[0].squeeze()

        # Construire les timings si les durées sont disponibles
        timings: list[PhonemeTiming] = []
        if self._has_durations and len(result) > 1:
            dur_frames = result[1].flatten()
            padded_phonemes = ["_"] + list(phonemes_str) + ["_"]
            timings = _build_phoneme_timings(padded_phonemes, dur_frames)

        return audio, timings


# ── Auto-enregistrement ──

def _check() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


register(EngineInfo(
    key="kokoro",
    name="Kokoro",
    description="Synthèse neuronale transformer 82M — inférence ONNX directe (MIT).",
    supports_phonemes=True,
    supports_text=False,
    requires_internet=False,
    requires_api_key=False,
    install_instructions="Modèles téléchargés automatiquement.",
    check_available=_check,
    factory=lambda p: KokoroTTSEngine(**p),
    params=[
        EngineParam("voice", "Voix", "choice", _VOICE,
                     choices=["ff_siwis"], role="voice"),
        EngineParam("speed", "Vitesse", "float", 1.0, min_val=0.5, max_val=2.0,
                     role="speed"),
    ],
    category="builtin",
    model_urls=[
        (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            str(_MODELS_DIR / "kokoro-v1.0.onnx"),
        ),
        (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            str(_MODELS_DIR / "voices-v1.0.bin"),
        ),
    ],
))
