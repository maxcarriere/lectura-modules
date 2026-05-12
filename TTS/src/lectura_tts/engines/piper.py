"""Moteur TTS Piper — inférence VITS ONNX directe (sans le package piper-tts GPL).

Charge un modèle Piper (.onnx + .onnx.json) et fait l'inférence
directement via onnxruntime (MIT). Aucune dépendance GPL.

Pré-requis : onnxruntime (déjà disponible si Kokoro est installé)
Modèles : téléchargés depuis https://huggingface.co/rhasspy/piper-voices
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from lectura_tts.models import PhonemeTiming, TTSResult
from lectura_tts.registry import EngineInfo, EngineParam, register

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "fr_FR-siwis-medium"
_MODELS_DIR = Path.home() / ".local/share/piper"

# Tokens spéciaux VITS
_BOS = "^"
_EOS = "$"
_PAD = "_"


class PiperTTSEngine:
    """Inférence VITS ONNX directe — zéro dépendance GPL."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        models_dir: str = "",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        self._model_name = model_name
        self._models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        self._length_scale = max(0.5, min(3.0, length_scale))
        self._noise_scale = noise_scale
        self._noise_w = noise_w
        # Lazy-loaded
        self._session = None
        self._phoneme_id_map: dict[str, list[int]] = {}
        self._sample_rate: int = 22050
        self._hop_length: int = 256
        self._num_speakers: int = 1

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import onnxruntime

        model_path = self._models_dir / f"{self._model_name}.onnx"
        config_path = self._models_dir / f"{self._model_name}.onnx.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle Piper introuvable : {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config Piper introuvable : {config_path}")

        # Charger la config JSON
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        self._phoneme_id_map = cfg.get("phoneme_id_map", {})
        audio_cfg = cfg.get("audio", {})
        self._sample_rate = audio_cfg.get("sample_rate", 22050)
        self._hop_length = audio_cfg.get("hop_length", 256)
        self._num_speakers = cfg.get("num_speakers", 1)

        # Créer la session ONNX
        opts = onnxruntime.SessionOptions()
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    # ── Conversion IPA → IDs ──

    def _phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """Convertit une liste de phonèmes IPA en séquence d'IDs pour le modèle.

        Piper décompose les diacritiques combinants (ex: ɔ̃ → ɔ + ̃).
        Si un phonème n'est pas dans le id_map, on tente caractère par caractère.
        """
        id_map = self._phoneme_id_map
        ids: list[int] = []

        # BOS + PAD
        ids.extend(id_map.get(_BOS, [1]))
        ids.extend(id_map.get(_PAD, [0]))

        for phoneme in phonemes:
            if phoneme in id_map:
                ids.extend(id_map[phoneme])
            elif len(phoneme) > 1:
                # Décomposer : base + diacritiques combinants
                for char in phoneme:
                    if char in id_map:
                        ids.extend(id_map[char])
                    else:
                        log.warning("Caractère absent du id_map : %r (de %s)", char, phoneme)
            else:
                log.warning("Phonème absent du id_map : %s", phoneme)
                continue
            ids.extend(id_map.get(_PAD, [0]))

        # EOS
        ids.extend(id_map.get(_EOS, [2]))

        return ids

    # ── Inférence ONNX ──

    def _run_inference(
        self, phoneme_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Exécute le modèle VITS et retourne (audio, durations_par_id | None)."""
        assert self._session is not None

        ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        ids_lengths = np.array([ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [self._noise_scale, self._length_scale, self._noise_w],
            dtype=np.float32,
        )

        args = {
            "input": ids_array,
            "input_lengths": ids_lengths,
            "scales": scales,
        }

        if self._num_speakers > 1:
            args["sid"] = np.array([0], dtype=np.int64)

        result = self._session.run(None, args)
        audio = result[0].squeeze()

        # Le modèle retourne optionnellement les durées par phonème ID
        phoneme_id_samples = None
        if len(result) > 1:
            phoneme_id_samples = (result[1].squeeze() * self._hop_length).astype(
                np.int64
            )

        return audio, phoneme_id_samples

    # ── Alignement phonème → timing ──

    def _build_timings(
        self,
        phonemes: list[str],
        phoneme_ids: list[int],
        phoneme_id_samples: np.ndarray,
    ) -> list[PhonemeTiming]:
        """Reconstruit les timings phonème à partir des durées par ID."""
        id_map = self._phoneme_id_map
        pad_ids = id_map.get(_PAD, [0])
        sr = self._sample_rate
        timings: list[PhonemeTiming] = []

        # Séquence attendue : BOS PAD ph1 PAD ph2 PAD ... EOS
        all_symbols = [_BOS] + list(phonemes) + [_EOS]
        id_idx = 0

        for symbol in all_symbols:
            expected_ids = id_map.get(symbol, [])
            if symbol != _EOS:
                expected_ids = list(expected_ids) + list(pad_ids)

            start_idx = id_idx
            ok = True
            for eid in expected_ids:
                if id_idx >= len(phoneme_ids) or eid != phoneme_ids[id_idx]:
                    ok = False
                    break
                id_idx += 1

            if not ok:
                log.debug("Alignement échoué au symbole %s", symbol)
                return timings

            # Sommer les samples pour ce symbole
            num_samples = int(phoneme_id_samples[start_idx:id_idx].sum())

            if symbol not in (_BOS, _EOS, _PAD):
                cursor = int(phoneme_id_samples[:start_idx].sum())
                start_ms = cursor / sr * 1000
                end_ms = (cursor + num_samples) / sr * 1000
                timings.append(PhonemeTiming(
                    ipa=symbol, start_ms=start_ms, end_ms=end_ms,
                ))

        return timings

    # ── API publique ──

    def synthesize(self, text: str) -> TTSResult:
        """Synthétise du texte (utilise eSpeak via subprocess pour la phonémisation)."""
        # Fallback : pour le texte brut, on passe par synthesize_phonemes
        # après phonémisation externe (le pipeline Lectura fournit déjà les IPA)
        return self.synthesize_phonemes(text)

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        """Synthétise une séquence de phonèmes IPA."""
        self._ensure_loaded()

        # Séparer les phonèmes (caractères IPA individuels)
        from lectura_tts.ipa import iter_phonemes
        phoneme_list = list(iter_phonemes(phonemes_ipa))

        if not phoneme_list:
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=self._sample_rate,
                phoneme_timings=[],
            )

        phoneme_ids = self._phonemes_to_ids(phoneme_list)
        audio, phoneme_id_samples = self._run_inference(phoneme_ids)

        # Normaliser
        max_val = np.max(np.abs(audio))
        if max_val > 1e-8:
            audio = audio / max_val
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        # Construire les timings si le modèle les fournit
        timings: list[PhonemeTiming] = []
        if phoneme_id_samples is not None and len(phoneme_id_samples) == len(phoneme_ids):
            timings = self._build_timings(phoneme_list, phoneme_ids, phoneme_id_samples)

        return TTSResult(
            samples=audio,
            sample_rate=self._sample_rate,
            phoneme_timings=timings,
        )


# ── Auto-enregistrement ──

def _check() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR"

register(EngineInfo(
    key="piper",
    name="Piper",
    description="Synthèse neuronale VITS — inférence ONNX directe (MIT).",
    supports_phonemes=True,
    supports_text=False,
    requires_internet=False,
    requires_api_key=False,
    install_instructions="Modèles téléchargés automatiquement.",
    check_available=_check,
    factory=lambda p: PiperTTSEngine(**p),
    params=[
        EngineParam("model_name", "Modèle", "choice", _DEFAULT_MODEL,
                     choices=["fr_FR-siwis-medium", "fr_FR-siwis-low",
                              "fr_FR-mls-medium", "fr_FR-gilles-low"],
                     role="voice"),
        EngineParam("length_scale", "Vitesse", "float", 1.0,
                     min_val=0.5, max_val=3.0, role="speed"),
    ],
    category="builtin",
    model_urls=[
        (
            f"{_HF_BASE}/siwis/medium/fr_FR-siwis-medium.onnx",
            str(_MODELS_DIR / "fr_FR-siwis-medium.onnx"),
        ),
        (
            f"{_HF_BASE}/siwis/medium/fr_FR-siwis-medium.onnx.json",
            str(_MODELS_DIR / "fr_FR-siwis-medium.onnx.json"),
        ),
    ],
))
