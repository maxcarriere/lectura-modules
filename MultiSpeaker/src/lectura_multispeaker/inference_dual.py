"""Engine TTS dual FastPitch/Conformer — routage par longueur de sequence.

Le Conformer 384 produit le meilleur audio sur les phrases moyennes/longues
et pour siwis sur toutes les longueurs, mais degrade les sequences courtes
(syllabes, mots isoles) des non-siwis.

Le FastPitch 256 (production actuelle) gere correctement les courtes grace
a ses heuristiques et son kernel=9.

DualTTSEngine instancie les deux moteurs et route selon :
  - speaker == siwis (toutes longueurs) -> Conformer
  - n_phones > threshold (tous speakers) -> Conformer
  - n_phones <= threshold ET speaker != siwis -> FastPitch
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lectura_multispeaker.inference_onnx import OnnxTTSEngine, TTSResult

log = logging.getLogger(__name__)


class DualTTSEngine:
    """Wrapper dual FastPitch + Conformer avec routage automatique.

    Parameters
    ----------
    models_dir : Path
        Repertoire contenant dual_config.json + sous-repertoires
        fastpitch/ et conformer/ + hifigan.onnx partage.
    speaker : str
        Nom du speaker initial (default: "siwis")
    """

    def __init__(self, models_dir: Path, speaker: str = "siwis") -> None:
        self._models_dir = Path(models_dir)
        self._speaker = speaker

        # Charger la config de routage
        config_path = self._models_dir / "dual_config.json"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        routing = config["routing"]
        self._threshold = routing["threshold"]
        self._always_conformer = set(routing.get("always_conformer_speakers", []))
        self._short_model_key = routing.get("short_model", "fastpitch")
        self._long_model_key = routing.get("long_model", "conformer")

        models_cfg = config["models"]
        conformer_dir = self._models_dir / models_cfg["conformer"]["dir"]
        fastpitch_dir = self._models_dir / models_cfg["fastpitch"]["dir"]

        self._conformer = OnnxTTSEngine(conformer_dir, speaker=speaker)
        self._fastpitch = OnnxTTSEngine(fastpitch_dir, speaker=speaker)

        log.info(
            "DualTTSEngine charge depuis %s (threshold=%d, always_conformer=%s)",
            models_dir, self._threshold, self._always_conformer,
        )

    @property
    def speaker(self) -> str:
        """Speaker actuellement selectionne."""
        return self._speaker

    @property
    def sample_rate(self) -> int:
        """Frequence d'echantillonnage (identique pour les deux engines)."""
        return 22050

    @property
    def model_type(self) -> str:
        """Type de modele."""
        return "dual"

    def set_speaker(self, speaker: str) -> None:
        """Change le speaker actif sur les deux engines."""
        self._speaker = speaker
        self._conformer.set_speaker(speaker)
        self._fastpitch.set_speaker(speaker)

    def _choose_engine(self, phonemes_ipa: str) -> tuple[OnnxTTSEngine, str]:
        """Choisit l'engine selon le speaker et la longueur de la sequence.

        Returns
        -------
        tuple[OnnxTTSEngine, str]
            (engine, nom du modele choisi)
        """
        from lectura_multispeaker.phonemes import ipa_to_phones

        phones = ipa_to_phones(phonemes_ipa)
        n_phones = len(phones) + 2  # +2 SIL markers (#...#)

        use_conformer = (
            self._speaker in self._always_conformer
            or n_phones > self._threshold
        )

        if use_conformer:
            return self._conformer, self._long_model_key
        return self._fastpitch, self._short_model_key

    def synthesize_phonemes(
        self,
        phonemes_ipa: str,
        phrase_type: int = 0,
        duration_scale: float | None = None,
        pitch_shift: float | None = None,
        pitch_range: float | None = None,
        energy_scale: float | None = None,
        pause_scale: float | None = None,
        variability: bool = False,
        style: str | None = None,
        style_vector: list[float] | None = None,
        n_ode_steps: int | None = None,
        duration_noise: float | None = None,
    ) -> TTSResult:
        """Synthetise une sequence de phonemes IPA via l'engine appropriate.

        Route vers Conformer ou FastPitch selon le speaker et n_phones.
        """
        engine, model_name = self._choose_engine(phonemes_ipa)

        log.debug(
            "Routage dual: speaker=%s, ipa=%s... -> %s",
            self._speaker, phonemes_ipa[:20], model_name,
        )

        return engine.synthesize_phonemes(
            phonemes_ipa,
            phrase_type=phrase_type,
            duration_scale=duration_scale,
            pitch_shift=pitch_shift,
            pitch_range=pitch_range,
            energy_scale=energy_scale,
            pause_scale=pause_scale,
            variability=variability,
            style=style,
            style_vector=style_vector,
            n_ode_steps=n_ode_steps,
            duration_noise=duration_noise,
        )

    def synthesize(
        self,
        text: str,
        phrase_type: int | None = None,
        duration_scale: float | None = None,
        pitch_shift: float | None = None,
        pitch_range: float | None = None,
        energy_scale: float | None = None,
        pause_scale: float | None = None,
        variability: bool = False,
        style: str | None = None,
        style_vector: list[float] | None = None,
        n_ode_steps: int | None = None,
        duration_noise: float | None = None,
    ) -> TTSResult:
        """Synthetise du texte (necessite lectura-g2p).

        Chaque phrase du texte est routee individuellement selon sa longueur.
        """
        try:
            from lectura_g2p import creer_engine as creer_g2p
        except ImportError:
            raise ImportError(
                "lectura-g2p requis pour synthesize(text). "
                "Installer avec : pip install lectura-multispeaker[g2p]"
            )

        # Reutiliser le G2P du conformer (ou en creer un)
        if self._conformer._g2p is None:
            self._conformer._g2p = creer_g2p(mode="auto")
        g2p = self._conformer._g2p

        from lectura_multispeaker.inference_onnx import _text_to_sentences, PhonemeTiming

        sentences = _text_to_sentences(text, g2p)

        if not sentences:
            return TTSResult(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self.sample_rate,
                phoneme_timings=[],
            )

        prosody = dict(
            duration_scale=duration_scale, pitch_shift=pitch_shift,
            pitch_range=pitch_range, energy_scale=energy_scale,
            pause_scale=pause_scale, variability=variability,
            style=style, style_vector=style_vector,
            n_ode_steps=n_ode_steps, duration_noise=duration_noise,
        )

        if len(sentences) == 1:
            ipa, auto_pt = sentences[0]
            return self.synthesize_phonemes(
                ipa,
                phrase_type=phrase_type if phrase_type is not None else auto_pt,
                **prosody,
            )

        all_samples: list[np.ndarray] = []
        all_timings: list[PhonemeTiming] = []
        time_offset_ms = 0.0

        for ipa, auto_pt in sentences:
            result = self.synthesize_phonemes(
                ipa,
                phrase_type=phrase_type if phrase_type is not None else auto_pt,
                **prosody,
            )

            for t in result.phoneme_timings:
                all_timings.append(PhonemeTiming(
                    ipa=t.ipa,
                    start_ms=t.start_ms + time_offset_ms,
                    end_ms=t.end_ms + time_offset_ms,
                ))

            all_samples.append(result.samples)
            time_offset_ms += len(result.samples) / self.sample_rate * 1000

        combined = np.concatenate(all_samples)
        return TTSResult(
            samples=combined,
            sample_rate=self.sample_rate,
            phoneme_timings=all_timings,
        )
