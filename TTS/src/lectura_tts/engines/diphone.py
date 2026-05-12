"""Moteur TTS Lectura Diphone — concatenation WORLD.

Synthese par diphones dans le domaine WORLD via lectura-tts-diphone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lectura_tts.models import PhonemeTiming, TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

log = logging.getLogger(__name__)


class DiphoneTTSEngine:
    """Wrapper lectura_tts pour le moteur TTS diphone WORLD."""

    def __init__(
        self,
        models_dir: str = "",
        synth_mode: str = "FLUIDE",
        duration_scale: float = 1.0,
        pause_scale: float = 1.0,
        f0_hz: float = 190.0,
        prosody_style: str = "auto",
        macro_expressivity: float = 1.0,
        micro_expressivity: float = 1.0,
        seed: str = "",
        determinism: float = 0.5,
        spectral_contrast: float = 1.3,
        ap_cleanup: float = 1.5,
        formant_sharpening: float = 1.3,
        vtln_alpha: float = 1.0,
    ) -> None:
        self._models_dir = models_dir or None
        self._synth_mode = synth_mode
        self._duration_scale = duration_scale
        self._pause_scale = pause_scale
        self._f0_hz = f0_hz
        self._prosody_style = prosody_style
        self._macro_expressivity = macro_expressivity
        self._micro_expressivity = micro_expressivity
        self._seed: int | None = int(seed) if seed.strip() else None
        self._determinism = float(determinism)
        self._spectral_contrast = spectral_contrast
        self._ap_cleanup = ap_cleanup
        self._formant_sharpening = formant_sharpening
        self._vtln_alpha = vtln_alpha
        self._engine = None

    @staticmethod
    def _pad_tail(audio: np.ndarray, sr: int, ms: int = 50) -> np.ndarray:
        """Ajoute un court silence en fin d'audio pour eviter la coupure."""
        pad = np.zeros(int(sr * ms / 1000), dtype=np.float32)
        return np.concatenate([audio, pad])

    def _resolve_seed(self) -> int | None:
        """Resout la graine selon le niveau de determinisme."""
        import random
        if self._seed is not None:
            return self._seed
        if self._determinism >= 1.0:
            return 42  # seed fixe par defaut
        if self._determinism <= 0.0:
            return random.randint(0, 2**31)
        # Entre 0 et 1 : seed fixe si >= 0.5
        if self._determinism >= 0.5:
            return 42
        return random.randint(0, 2**31)

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from lectura_tts_diphone import creer_engine
        self._engine = creer_engine(mode="local", models_dir=self._models_dir)

    def synthesize(self, text: str) -> TTSResult:
        """Synthetise du texte (necessite G2P)."""
        self._ensure_loaded()
        groups = self._engine._g2p_backend.phonemize(text)
        if not groups:
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=44100,
            )
        audio = self._engine.synthesize_groups(
            groups,
            mode=self._synth_mode,
            duration_scale=self._duration_scale,
            pause_scale=self._pause_scale,
            prosody={"f0_hz": self._f0_hz},
            macro_expressivity=self._macro_expressivity,
            micro_expressivity=self._micro_expressivity,
            seed=self._resolve_seed(),
            prosody_style=self._prosody_style,
            spectral_contrast=self._spectral_contrast,
            ap_cleanup=self._ap_cleanup,
            formant_sharpening=self._formant_sharpening,
            vtln_alpha=self._vtln_alpha,
        )
        audio = self._pad_tail(audio, 44100)
        return TTSResult(samples=audio, sample_rate=44100)

    # Ponctuation → boundary type (ordre important : "..." avant ".")
    _PUNCT_MARKS = [("...", "suspensive"), (",", "comma"),
                    (".", "period"), ("?", "question"), ("!", "exclamation")]

    def _ipa_to_groups(self, phonemes_ipa: str) -> list[dict]:
        """Decoupe un texte IPA en groupes prosodiques via la ponctuation."""
        import re
        from lectura_tts_diphone.phonemes import ipa_to_phones

        text = phonemes_ipa.strip()
        if not text:
            return []

        # Split sur la ponctuation en gardant le separateur
        parts = re.split(r'(\.\.\.|[.,?!])', text)

        groups: list[dict] = []
        for i in range(0, len(parts), 2):
            chunk = parts[i].strip()
            if not chunk:
                continue
            phones = ipa_to_phones(chunk)
            if not phones:
                continue
            # Ponctuation qui suit ce chunk
            boundary = "none"
            if i + 1 < len(parts):
                punct = parts[i + 1]
                for mark, btype in self._PUNCT_MARKS:
                    if punct == mark:
                        boundary = btype
                        break
            groups.append({"phones": phones, "boundary": boundary})

        # Si aucune ponctuation detectee, un seul groupe "period"
        if len(groups) == 1 and groups[0]["boundary"] == "none":
            groups[0]["boundary"] = "period"

        return groups

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        """Synthetise des phonemes IPA."""
        self._ensure_loaded()

        groups = self._ipa_to_groups(phonemes_ipa)
        if not groups:
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=44100,
            )
        audio = self._engine.synthesize_groups(
            groups,
            mode=self._synth_mode,
            duration_scale=self._duration_scale,
            pause_scale=self._pause_scale,
            prosody={"f0_hz": self._f0_hz},
            macro_expressivity=self._macro_expressivity,
            micro_expressivity=self._micro_expressivity,
            seed=self._resolve_seed(),
            prosody_style=self._prosody_style,
            spectral_contrast=self._spectral_contrast,
            ap_cleanup=self._ap_cleanup,
            formant_sharpening=self._formant_sharpening,
            vtln_alpha=self._vtln_alpha,
        )
        audio = self._pad_tail(audio, 44100)
        return TTSResult(samples=audio, sample_rate=44100)


# ── Auto-enregistrement ──

def _check() -> bool:
    """Verifie si lectura-tts-diphone + pyworld sont disponibles."""
    try:
        import lectura_tts_diphone  # noqa: F401
        import pyworld  # noqa: F401
        from lectura_tts_diphone._chargeur import find_models_dir
        return find_models_dir() is not None
    except ImportError:
        return False


register(EngineInfo(
    key="lectura-diphone",
    name="Lectura Diphone",
    description="Synthese par concatenation de diphones WORLD — francais 44.1 kHz.",
    supports_phonemes=True,
    supports_text=True,
    requires_internet=False,
    requires_api_key=False,
    install_instructions="pip install lectura-tts-diphone[all]",
    check_available=_check,
    factory=lambda p: DiphoneTTSEngine(**p),
    params=[
        # Primaires (role non-vide → visibles dans la barre principale)
        EngineParam("prosody_style", "Mode prosodique", "choice", "auto",
                    choices=["auto", "declaratif", "question",
                             "exclamation", "suspensif", "neutre"],
                    role="voice"),
        EngineParam("macro_expressivity", "Expressivite macro", "float", 1.0,
                    min_val=0.0, max_val=4.0, role="expressivity"),
        EngineParam("micro_expressivity", "Expressivite micro", "float", 1.0,
                    min_val=0.0, max_val=4.0, role="expressivity_micro"),
        EngineParam("duration_scale", "Vitesse", "float", 1.0,
                    min_val=0.5, max_val=3.0, role="speed"),
        EngineParam("f0_hz", "Pitch", "float", 190.0,
                    min_val=120.0, max_val=300.0, role="pitch"),
        EngineParam("determinism", "Determinisme", "float", 0.5,
                    min_val=0.0, max_val=1.0, role="determinism"),
        EngineParam("seed", "Graine", "str", "", role="seed"),
        # Avances (role vide → panneau avance)
        EngineParam("synth_mode", "Mode synthese", "choice", "FLUIDE",
                    choices=["FLUIDE", "MOT_A_MOT", "SYLLABES"], role=""),
        EngineParam("pause_scale", "Pauses", "float", 1.0,
                    min_val=0.5, max_val=3.0, role=""),
        EngineParam("spectral_contrast", "Contraste spectral", "float", 1.3,
                    min_val=1.0, max_val=3.0, role=""),
        EngineParam("ap_cleanup", "Clarte voix", "float", 1.5,
                    min_val=1.0, max_val=3.0, role=""),
        EngineParam("formant_sharpening", "Nettete formants", "float", 1.3,
                    min_val=1.0, max_val=2.0, role=""),
        EngineParam("vtln_alpha", "Chaleur voix", "float", 1.0,
                    min_val=0.8, max_val=1.2, role="voice"),
    ],
    category="builtin",
    pip_packages=["lectura-tts-diphone[all]"],
    check_modules=["lectura_tts_diphone", "pyworld"],
))
