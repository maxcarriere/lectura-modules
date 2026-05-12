"""Moteur TTS MBROLA — synthèse diphone avec entrée PHO directe.

Pré-requis système : sudo apt install mbrola mbrola-fr1 mbrola-fr2 mbrola-fr4
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from lectura_tts.ipa import iter_phonemes
from lectura_tts.models import PhonemeTiming, TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

_MBROLA_DATA_DIRS = [
    Path("/usr/share/mbrola"),
    Path("/usr/local/share/mbrola"),
]

_DEFAULT_VOICE = "fr4"
_MBROLA_SAMPLE_RATE = 16000

# Durées par défaut (ms) par catégorie phonétique
_DUR_VOYELLE = 120
_DUR_VOYELLE_NASALE = 140
_DUR_CONSONNE_OCCLUSIVE = 80
_DUR_CONSONNE_FRICATIVE = 100
_DUR_CONSONNE_NASALE = 80
_DUR_CONSONNE_LIQUIDE = 70
_DUR_SEMI_VOYELLE = 60
_DUR_SCHWA = 90
_DUR_DEFAULT = 90

# Mapping IPA → SAMPA MBROLA (français)
_IPA_TO_SAMPA: dict[str, str | list[str]] = {
    "i": "i", "e": "e", "ɛ": "E", "a": "a", "ɑ": "A",
    "ɔ": "O", "o": "o", "u": "u", "y": "y",
    "ø": "2", "œ": "9", "ə": "@",
    "ɑ̃": "a~", "ɛ̃": "e~", "ɔ̃": "o~", "œ̃": "9~",
    "j": "j", "w": "w", "ɥ": "H",
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k",
    "ɡ": "g", "g": "g",
    "f": "f", "v": "v", "s": "s", "z": "z", "ʃ": "S", "ʒ": "Z",
    "m": "m", "n": "n", "ɲ": ["n", "j"], "ŋ": "N",
    "l": "l", "ʁ": "R", "r": "R",
}

_VOYELLES_SET = {"i", "e", "E", "a", "A", "O", "o", "u", "y", "2", "9", "@"}
_NASALES_SET = {"a~", "e~", "o~", "9~"}
_OCCLUSIVES_SET = {"p", "b", "t", "d", "k", "g"}
_FRICATIVES_SET = {"f", "v", "s", "z", "S", "Z"}
_NASALES_CONS_SET = {"m", "n", "N"}
_LIQUIDES_SET = {"l", "R"}
_SEMI_VOYELLES_SET = {"j", "w", "H"}


def _default_duration(sampa: str) -> int:
    if sampa in _NASALES_SET:
        return _DUR_VOYELLE_NASALE
    if sampa == "@":
        return _DUR_SCHWA
    if sampa in _VOYELLES_SET:
        return _DUR_VOYELLE
    if sampa in _OCCLUSIVES_SET:
        return _DUR_CONSONNE_OCCLUSIVE
    if sampa in _FRICATIVES_SET:
        return _DUR_CONSONNE_FRICATIVE
    if sampa in _NASALES_CONS_SET:
        return _DUR_CONSONNE_NASALE
    if sampa in _LIQUIDES_SET:
        return _DUR_CONSONNE_LIQUIDE
    if sampa in _SEMI_VOYELLES_SET:
        return _DUR_SEMI_VOYELLE
    return _DUR_DEFAULT


def _ipa_to_pho(ipa: str, pitch_hz: float = 180.0) -> str:
    """Convertit une chaîne IPA en format PHO MBROLA."""
    phonemes = iter_phonemes(ipa)
    lines: list[str] = []

    for ph in phonemes:
        mapped = _IPA_TO_SAMPA.get(ph)
        if mapped is None:
            continue
        if isinstance(mapped, list):
            for sub in mapped:
                dur = _default_duration(sub)
                if sub in _VOYELLES_SET or sub in _NASALES_SET:
                    lines.append(f"{sub} {dur} 50 {pitch_hz:.0f}")
                else:
                    lines.append(f"{sub} {dur}")
        else:
            dur = _default_duration(mapped)
            if mapped in _VOYELLES_SET or mapped in _NASALES_SET:
                lines.append(f"{mapped} {dur} 50 {pitch_hz:.0f}")
            else:
                lines.append(f"{mapped} {dur}")

    return "\n".join(lines) + "\n"


def _find_voice_path(voice: str) -> Path | None:
    for data_dir in _MBROLA_DATA_DIRS:
        candidate = data_dir / voice / voice
        if candidate.exists():
            return candidate
        candidate = data_dir / voice
        if candidate.is_file():
            return candidate
    return None


def _parse_au(data: bytes) -> tuple[np.ndarray, int]:
    """Parse un fichier .au (Sun audio) → (samples_float32, sample_rate)."""
    if len(data) < 24:
        return np.array([], dtype=np.float32), _MBROLA_SAMPLE_RATE

    magic = data[:4]
    if magic != b".snd":
        samples_int16 = np.frombuffer(data, dtype=np.int16)
        return samples_int16.astype(np.float32) / 32768.0, _MBROLA_SAMPLE_RATE

    data_offset = struct.unpack(">I", data[4:8])[0]
    sample_rate = struct.unpack(">I", data[16:20])[0]
    pcm_data = data[data_offset:]
    samples_int16 = np.frombuffer(pcm_data, dtype=">i2")
    return samples_int16.astype(np.float32) / 32768.0, sample_rate


class MbrolaTTSEngine:
    """Implémentation TTSEngine avec MBROLA (diphone, entrée PHO directe)."""

    def __init__(
        self,
        voice: str = _DEFAULT_VOICE,
        pitch_hz: float = 180.0,
        duration_scale: float = 1.0,
    ) -> None:
        self._voice = voice
        self._pitch_hz = pitch_hz
        self._duration_scale = max(0.3, min(5.0, duration_scale))
        self._voice_path: Path | None = None

    def _ensure_voice(self) -> Path:
        if self._voice_path is not None:
            return self._voice_path
        path = _find_voice_path(self._voice)
        if path is None:
            raise FileNotFoundError(
                f"Voix MBROLA '{self._voice}' introuvable. "
                f"Installer avec : sudo apt install mbrola-{self._voice}"
            )
        self._voice_path = path
        return path

    def _run_mbrola(self, pho: str) -> tuple[np.ndarray, int]:
        voice_path = self._ensure_voice()
        proc = subprocess.run(
            ["mbrola", str(voice_path), "-", "-.au"],
            input=pho.encode("utf-8"),
            capture_output=True,
            timeout=10,
        )
        if proc.returncode != 0 and not proc.stdout:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"MBROLA erreur : {stderr}")
        if not proc.stdout:
            return np.array([], dtype=np.float32), _MBROLA_SAMPLE_RATE
        return _parse_au(proc.stdout)

    def _scale_pho_durations(self, pho: str) -> str:
        lines: list[str] = []
        for line in pho.splitlines():
            line = line.strip()
            if not line or line.startswith(";"):
                lines.append(line)
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    dur = int(float(parts[1]) * self._duration_scale)
                    parts[1] = str(dur)
                except ValueError:
                    pass
            lines.append(" ".join(parts))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _pho_to_timings(pho: str) -> list[PhonemeTiming]:
        timings: list[PhonemeTiming] = []
        cursor_ms = 0.0
        for line in pho.splitlines():
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            phoneme = parts[0]
            try:
                duration = float(parts[1])
            except ValueError:
                continue
            if phoneme != "_":
                timings.append(PhonemeTiming(ipa=phoneme, start_ms=cursor_ms,
                                             end_ms=cursor_ms + duration))
            cursor_ms += duration
        return timings

    def synthesize(self, text: str) -> TTSResult:
        try:
            proc = subprocess.run(
                ["espeak-ng", "-v", f"mb-{self._voice}", "--pho", "-q", text],
                capture_output=True, timeout=10,
            )
            pho = proc.stdout.decode("utf-8", errors="replace")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pho = _ipa_to_pho(text, pitch_hz=self._pitch_hz)

        if not pho.strip():
            return TTSResult(samples=np.array([], dtype=np.float32),
                             sample_rate=_MBROLA_SAMPLE_RATE, phoneme_timings=[])

        if self._duration_scale != 1.0:
            pho = self._scale_pho_durations(pho)

        samples, sr = self._run_mbrola(pho)
        timings = self._pho_to_timings(pho)
        return TTSResult(samples=samples, sample_rate=sr, phoneme_timings=timings)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        pho = _ipa_to_pho(phonemes_ipa, pitch_hz=self._pitch_hz)
        if not pho.strip():
            return TTSResult(samples=np.array([], dtype=np.float32),
                             sample_rate=_MBROLA_SAMPLE_RATE, phoneme_timings=[])

        if self._duration_scale != 1.0:
            pho = self._scale_pho_durations(pho)

        pho = f"_ 20\n{pho}_ 20\n"
        samples, sr = self._run_mbrola(pho)
        timings = self._pho_to_timings(pho)
        return TTSResult(samples=samples, sample_rate=sr, phoneme_timings=timings)


# ── Auto-enregistrement ──

def _check() -> bool:
    return shutil.which("mbrola") is not None


register(EngineInfo(
    key="mbrola",
    name="MBROLA",
    description="Synthèse diphone avec contrôle précis des durées et du pitch.",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=False,
    requires_api_key=False,
    install_instructions="sudo apt install mbrola mbrola-fr1 mbrola-fr4",
    check_available=_check,
    factory=lambda p: MbrolaTTSEngine(**p),
    params=[
        EngineParam("voice", "Voix", "choice", _DEFAULT_VOICE,
                     choices=["fr1", "fr2", "fr4"], role="voice"),
        EngineParam("duration_scale", "Vitesse", "float", 1.0,
                     min_val=0.3, max_val=5.0, role="speed"),
        EngineParam("pitch_hz", "Pitch (Hz)", "float", 180.0, min_val=80, max_val=350,
                     role="pitch"),
    ],
    category="builtin",
    license_notice=(
        "MBROLA est sous licence AGPL-3.0. Les voix françaises (fr1-fr7) "
        "sont distribuées sous licence non-commerciale."
    ),
))
