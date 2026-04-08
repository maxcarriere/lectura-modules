"""Moteur TTS eSpeak-NG — synthèse formantique avec entrée phonémique.

Pré-requis système : sudo apt install espeak-ng
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave

import numpy as np

from lectura_tts.ipa import iter_phonemes
from lectura_tts.models import PhonemeTiming, TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_ESPEAK_SAMPLE_RATE = 22050

# Mapping IPA → notation eSpeak-NG (français)
_IPA_TO_ESPEAK: dict[str, str] = {
    "i": "i", "e": "e", "ɛ": "E", "a": "a", "ɑ": "a",
    "ɔ": "O", "o": "o", "u": "u", "y": "y",
    "ø": "Y", "œ": "W", "ə": "@",
    "ɑ̃": "A~", "ɛ̃": "E~", "ɔ̃": "O~", "œ̃": "W~",
    "j": "j", "w": "w", "ɥ": "H",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k",
    "ɡ": "g", "g": "g",
    "f": "f", "v": "v", "s": "s", "z": "z", "ʃ": "S", "ʒ": "Z",
    "m": "m", "n": "n", "ɲ": "n^", "ŋ": "N",
    "l": "l", "ʁ": "R", "r": "R",
}

# Durées relatives pour estimer les timings phonémiques
_RELATIVE_DURATIONS: dict[str, float] = {
    "i": 1.2, "e": 1.2, "E": 1.2, "a": 1.2, "O": 1.2,
    "o": 1.2, "u": 1.2, "y": 1.2, "Y": 1.2, "W": 1.2,
    "@": 0.9,
    "A~": 1.4, "E~": 1.4, "O~": 1.4, "W~": 1.4,
    "j": 0.6, "w": 0.6, "H": 0.6,
    "p": 0.8, "b": 0.8, "t": 0.8, "d": 0.8, "k": 0.8, "g": 0.8,
    "f": 1.0, "v": 1.0, "s": 1.0, "z": 1.0, "S": 1.0, "Z": 1.0,
    "m": 0.8, "n": 0.8, "n^": 0.9, "N": 0.8,
    "l": 0.7, "R": 0.7,
}


def _ipa_to_espeak(ipa: str) -> str:
    """Convertit une chaîne IPA en notation eSpeak-NG française."""
    phonemes = iter_phonemes(ipa)
    parts: list[str] = []
    for ph in phonemes:
        mapped = _IPA_TO_ESPEAK.get(ph)
        if mapped is not None:
            parts.append(mapped)
    return "".join(parts)


def _parse_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Parse un WAV depuis des bytes → (samples_float32, sample_rate)."""
    if len(data) < 44:
        return np.array([], dtype=np.float32), _ESPEAK_SAMPLE_RATE

    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples_int16 = np.frombuffer(raw, dtype=np.int16)
    return samples_int16.astype(np.float32) / 32768.0, sr


def _estimate_timings(
    espeak_phonemes: list[str],
    total_duration_ms: float,
) -> list[PhonemeTiming]:
    """Estime les timings phonémiques proportionnellement aux durées relatives."""
    if not espeak_phonemes or total_duration_ms <= 0:
        return []

    weights = [_RELATIVE_DURATIONS.get(ph, 0.8) for ph in espeak_phonemes]
    total_weight = sum(weights)
    if total_weight <= 0:
        return []

    timings: list[PhonemeTiming] = []
    cursor_ms = 0.0

    for ph, w in zip(espeak_phonemes, weights):
        duration = total_duration_ms * w / total_weight
        timings.append(PhonemeTiming(ipa=ph, start_ms=cursor_ms, end_ms=cursor_ms + duration))
        cursor_ms += duration

    return timings


class EspeakTTSEngine:
    """Implémentation TTSEngine avec eSpeak-NG (formant, entrée phonémique)."""

    def __init__(self, speed: int = 130, pitch: int = 50) -> None:
        self._speed = max(80, min(450, speed))
        self._pitch = max(0, min(99, pitch))

    def _run_espeak(self, text: str, is_phonemes: bool = False) -> bytes:
        if is_phonemes:
            input_text = f"[[{text}]]"
        else:
            input_text = text

        proc = subprocess.run(
            ["espeak-ng", "-v", "fr", "-s", str(self._speed),
             "-p", str(self._pitch), "--stdout", input_text],
            capture_output=True,
            timeout=10,
        )
        if proc.returncode != 0 and not proc.stdout:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"eSpeak-NG erreur : {stderr}")
        return proc.stdout

    def synthesize(self, text: str) -> TTSResult:
        wav_data = self._run_espeak(text)
        if not wav_data:
            return TTSResult(samples=np.array([], dtype=np.float32),
                             sample_rate=_ESPEAK_SAMPLE_RATE, phoneme_timings=[])

        samples, sr = _parse_wav(wav_data)

        # Obtenir les phonèmes pour estimer les timings
        try:
            proc = subprocess.run(
                ["espeak-ng", "-v", "fr", "-x", "-q", text],
                capture_output=True, timeout=5,
            )
            espeak_phones = proc.stdout.decode("utf-8", errors="replace").strip()
            phone_list = [ch for ch in espeak_phones if ch.isalpha() or ch in "~^@"]
        except Exception:
            phone_list = []

        total_ms = len(samples) / sr * 1000 if sr > 0 else 0
        timings = _estimate_timings(phone_list, total_ms) if phone_list else []

        return TTSResult(samples=samples, sample_rate=sr, phoneme_timings=timings)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        espeak_notation = _ipa_to_espeak(phonemes_ipa)
        if not espeak_notation:
            return TTSResult(samples=np.array([], dtype=np.float32),
                             sample_rate=_ESPEAK_SAMPLE_RATE, phoneme_timings=[])

        wav_data = self._run_espeak(espeak_notation, is_phonemes=True)
        if not wav_data:
            return TTSResult(samples=np.array([], dtype=np.float32),
                             sample_rate=_ESPEAK_SAMPLE_RATE, phoneme_timings=[])

        samples, sr = _parse_wav(wav_data)

        ipa_phonemes = iter_phonemes(phonemes_ipa)
        espeak_phones: list[str] = []
        for ph in ipa_phonemes:
            mapped = _IPA_TO_ESPEAK.get(ph)
            if mapped is not None:
                espeak_phones.append(mapped)

        total_ms = len(samples) / sr * 1000 if sr > 0 else 0
        timings = _estimate_timings(espeak_phones, total_ms)

        return TTSResult(samples=samples, sample_rate=sr, phoneme_timings=timings)


# ── Auto-enregistrement ──

def _check() -> bool:
    return shutil.which("espeak-ng") is not None


register(EngineInfo(
    key="espeak",
    name="eSpeak-NG",
    description="Synthèse formantique avec entrée phonémique directe. Léger et portable.",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=False,
    requires_api_key=False,
    install_instructions="sudo apt install espeak-ng",
    check_available=_check,
    factory=lambda p: EspeakTTSEngine(**p),
    params=[
        EngineParam("speed", "Vitesse (mots/min)", "int", 130, min_val=80, max_val=450),
        EngineParam("pitch", "Hauteur (0-99)", "int", 50, min_val=0, max_val=99),
    ],
))
