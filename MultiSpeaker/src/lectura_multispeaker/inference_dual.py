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

Les phrases tres longues (>_MAX_PHONES_CHUNK phones) sont automatiquement
decoupees a la ponctuation (virgules, points-virgules) pour eviter la
degradation de l'attention du Conformer sur les sequences >100 phones.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from lectura_multispeaker.inference_onnx import OnnxTTSEngine, TTSResult, PhonemeTiming

log = logging.getLogger(__name__)

# Seuil de phones au-dela duquel on decoupe les phrases longues
# a la ponctuation pour le Conformer (evite la degradation d'attention).
_MAX_PHONES_CHUNK = 80

# Ponctuation IPA sur laquelle on peut couper
_SPLIT_PUNCT = re.compile(r"(?<=[,;:])")


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

        # Facteur de duree par modele (1.0 = pas de correction)
        duration_factors = config.get("duration_factors", {})
        self._duration_factor = {
            self._short_model_key: duration_factors.get(self._short_model_key, 1.0),
            self._long_model_key: duration_factors.get(self._long_model_key, 1.0),
        }

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

    def _split_long_ipa(self, phonemes_ipa: str) -> list[str]:
        """Decoupe une sequence IPA trop longue a la ponctuation.

        Decoupe aux virgules/points-virgules pour que chaque chunk
        reste sous _MAX_PHONES_CHUNK phones. Si aucune ponctuation
        n'est presente, retourne la sequence entiere (pas de split force).
        """
        from lectura_multispeaker.phonemes import ipa_to_phones

        phones = ipa_to_phones(phonemes_ipa)
        if len(phones) + 2 <= _MAX_PHONES_CHUNK:
            return [phonemes_ipa]

        # Decouper aux virgules/points-virgules
        parts = _SPLIT_PUNCT.split(phonemes_ipa)
        if len(parts) <= 1:
            return [phonemes_ipa]

        # Fusionner les parties pour rester sous le seuil
        chunks: list[str] = []
        current = parts[0]

        for part in parts[1:]:
            candidate = current + part
            n = len(ipa_to_phones(candidate)) + 2
            if n > _MAX_PHONES_CHUNK and current.strip():
                chunks.append(current.strip())
                current = part
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        log.debug(
            "Split IPA long (%d phones) en %d chunks: %s",
            len(phones), len(chunks),
            [len(ipa_to_phones(c)) for c in chunks],
        )
        return chunks

    def _resolve_engine(
        self, phonemes_ipa: str, model: str | None = None,
    ) -> tuple[OnnxTTSEngine, str]:
        """Choisit l'engine en tenant compte du forçage eventuel.

        Parameters
        ----------
        phonemes_ipa : str
            Sequence IPA (pour le routage auto).
        model : str | None
            "conformer" ou "fastpitch" pour forcer, None = routage auto.
        """
        if model == self._long_model_key or model == "conformer":
            return self._conformer, self._long_model_key
        if model == self._short_model_key or model == "fastpitch" or model == "fft":
            return self._fastpitch, self._short_model_key
        return self._choose_engine(phonemes_ipa)

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
        model: str | None = None,
    ) -> TTSResult:
        """Synthetise une sequence de phonemes IPA via l'engine appropriate.

        Route vers Conformer ou FastPitch selon le speaker et n_phones.
        Les sequences tres longues sont automatiquement decoupees a la
        ponctuation pour eviter la degradation du Conformer.

        Parameters
        ----------
        model : str | None
            Forcer un modele : "conformer", "fastpitch"/"fft", ou None (auto).
        """
        engine, model_name = self._resolve_engine(phonemes_ipa, model)

        log.debug(
            "Routage dual: speaker=%s, model=%s, ipa=%s... -> %s",
            self._speaker, model or "auto", phonemes_ipa[:20], model_name,
        )

        # Appliquer le facteur de duree specifique au modele
        factor = self._duration_factor.get(model_name, 1.0)
        if factor != 1.0:
            from lectura_multispeaker.inference_onnx import _PROSODY_DEFAULTS
            base = duration_scale if duration_scale is not None else _PROSODY_DEFAULTS["duration_scale"]
            duration_scale = base * factor

        prosody = dict(
            duration_scale=duration_scale, pitch_shift=pitch_shift,
            pitch_range=pitch_range, energy_scale=energy_scale,
            pause_scale=pause_scale, variability=variability,
            style=style, style_vector=style_vector,
            n_ode_steps=n_ode_steps, duration_noise=duration_noise,
        )

        # Decouper les sequences longues envoyees au Conformer
        if model_name == self._long_model_key:
            chunks = self._split_long_ipa(phonemes_ipa)
            if len(chunks) > 1:
                return self._synthesize_chunks(engine, chunks, phrase_type, prosody)

        return engine.synthesize_phonemes(
            phonemes_ipa, phrase_type=phrase_type, **prosody,
        )

    def _synthesize_chunks(
        self,
        engine: OnnxTTSEngine,
        chunks: list[str],
        phrase_type: int,
        prosody: dict,
    ) -> TTSResult:
        """Synthetise une liste de chunks IPA et concatene les resultats."""
        all_samples: list[np.ndarray] = []
        all_timings: list[PhonemeTiming] = []
        time_offset_ms = 0.0

        for i, chunk_ipa in enumerate(chunks):
            # Dernier chunk garde le phrase_type original,
            # les intermediaires sont declaratifs (0) pour eviter
            # l'intonation finale.
            pt = phrase_type if i == len(chunks) - 1 else 0

            result = engine.synthesize_phonemes(
                chunk_ipa, phrase_type=pt, **prosody,
            )

            for t in result.phoneme_timings:
                all_timings.append(PhonemeTiming(
                    ipa=t.ipa,
                    start_ms=t.start_ms + time_offset_ms,
                    end_ms=t.end_ms + time_offset_ms,
                ))

            all_samples.append(result.samples)
            time_offset_ms += len(result.samples) / self.sample_rate * 1000

        combined = np.concatenate(all_samples) if all_samples else np.zeros(0, dtype=np.float32)
        return TTSResult(
            samples=combined,
            sample_rate=self.sample_rate,
            phoneme_timings=all_timings,
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
        model: str | None = None,
    ) -> TTSResult:
        """Synthetise du texte (necessite lectura-g2p).

        Chaque phrase du texte est routee individuellement selon sa longueur.

        Parameters
        ----------
        model : str | None
            Forcer un modele : "conformer", "fastpitch"/"fft", ou None (auto).
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

        from lectura_multispeaker.inference_onnx import _text_to_sentences

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
            model=model,
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
