"""Engine TTS ONNX local multi-speaker — inference via onnxruntime.

Supports two model types :
- matcha-conformer (v2) : matcha_encoder_{speaker}.onnx + matcha_unet.onnx + hifigan.onnx
- fastpitch (v1 legacy) : encoder_{speaker}.onnx + decoder.onnx + hifigan.onnx

For each model type, two encoder layouts :
- Unified : encoder.onnx (single file, dynamic speaker_id + style_vector)
- Per-speaker : encoder_{speaker}.onnx / matcha_encoder_{speaker}.onnx (baked speaker_id)

Glue numpy entre encoder et decoder/unet (length regulation + embeddings).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SEMITONE = 0.0577622  # log(2) / 12

# Prosodic defaults when no style preset is active
# duration_scale=0.85 : acceleration par defaut de 15% (comme monospeaker)
_PROSODY_DEFAULTS = {
    "duration_scale": 0.85,
    "pitch_shift": 0.0,
    "pitch_range": 1.0,
    "energy_scale": 1.0,
    "pause_scale": 1.0,
}

# Style presets: style_vector [expressiveness, energy_level, speaking_rate,
#                              final_intonation, is_dialogue]
#                + prosodic overrides (duration_scale, pitch_range, etc.)
STYLE_PRESETS = {
    "neutre": {
        "style_vector": [0.0, 0.0, 0.0, 0.0, 0.0],
        "pitch_range": 1.0, "energy_scale": 1.0,
        "duration_scale": 0.85, "pause_scale": 1.0,
        "duration_noise": 0.0,
    },
    "narratif": {
        "style_vector": [0.0, -0.2, -0.2, 0.0, 0.0],
        "pitch_range": 0.9, "energy_scale": 0.95,
        "duration_scale": 0.90, "pause_scale": 1.2,
        "duration_noise": 0.15,
    },
    "dialogue": {
        "style_vector": [0.3, 0.2, 0.2, 0.0, 1.0],
        "pitch_range": 1.15, "energy_scale": 1.0,
        "duration_scale": 0.80, "pause_scale": 0.8,
        "duration_noise": 0.12,
    },
    "expressif": {
        "style_vector": [1.0, 1.0, 0.0, 0.0, 0.0],
        "pitch_range": 1.1, "energy_scale": 1.3,
        "duration_scale": 0.88, "pause_scale": 1.0,
        "duration_noise": 0.18,
    },
    "meditatif": {
        "style_vector": [-1.5, -0.8, -1.0, -0.5, 0.0],
        "pitch_range": 0.7, "energy_scale": 0.8,
        "duration_scale": 1.15, "pause_scale": 1.8,
        "duration_noise": 0.08,
    },
    "rapide": {
        "style_vector": [0.0, 0.3, 2.0, 0.0, 0.0],
        "pitch_range": 1.1, "energy_scale": 1.0,
        "duration_scale": 0.65, "pause_scale": 0.6,
        "duration_noise": 0.05,
    },
    "lent": {
        "style_vector": [0.0, -0.2, -2.0, 0.0, 0.0],
        "pitch_range": 1.0, "energy_scale": 0.9,
        "duration_scale": 1.25, "pause_scale": 1.5,
        "duration_noise": 0.10,
    },
}

# Ponctuation reconnue par le modele TTS
_PUNCT_MAP = {",": ",", ";": ",", ":": ",", ".": ".", "!": "!", "?": "?",
              "\u2026": "\u2026", "...": "\u2026"}

# Durees minimales en frames pour la ponctuation (1 frame ~ 11.6 ms)
_PUNCT_MIN_FRAMES = {",": 10, ".": 20, "?": 15, "!": 15, "\u2026": 20}

# Phones correspondant a des silences
_SILENCE_PHONES = {"#", ",", ".", "?", "!", "\u2026"}


def _zero_silence_regions(
    audio: np.ndarray,
    phones: list[str],
    durations: np.ndarray,
    hop_length: int,
    fade_samples: int = 128,
) -> np.ndarray:
    """Remplace les zones SIL/ponctuation par du vrai silence."""
    if len(audio) == 0:
        return audio

    all_phones = ["#"] + list(phones) + ["#"]
    mask = np.ones(len(audio), dtype=np.float32)

    offset = 0
    for i, phone in enumerate(all_phones):
        dur_samples = int(durations[i]) * hop_length
        if phone in _SILENCE_PHONES:
            s = max(0, offset)
            e = min(len(audio), offset + dur_samples)
            mask[s:e] = 0.0
        offset += dur_samples

    fade = fade_samples
    diff = np.diff(mask, prepend=mask[0])

    for idx in np.where(diff < -0.5)[0]:
        e = min(len(audio), idx + fade)
        n = e - idx
        if n > 0:
            mask[idx:e] = np.linspace(1.0, 0.0, n)

    for idx in np.where(diff > 0.5)[0]:
        s = max(0, idx - fade)
        n = idx - s
        if n > 0:
            mask[s:idx] = np.linspace(0.0, 1.0, n)

    return audio * mask


def _text_to_sentences(text: str, g2p) -> list[tuple[str, int]]:
    """Decoupe le texte en phrases, chacune avec IPA + phrase_type.

    Delegue au pipeline unifie lectura_g2p.texte_vers_phrases_ipa().
    """
    try:
        from lectura_g2p import texte_vers_phrases_ipa
        return texte_vers_phrases_ipa(text, engine=g2p)
    except ImportError:
        # Fallback minimal sans le pipeline complet
        log.warning("lectura-g2p non installe — liaisons et formules ignorees")
        words = text.split()
        if not words:
            return []
        result = g2p.analyser(words)
        return [("".join(result.get("g2p", [])), 0)]


@dataclass
class PhonemeTiming:
    """Timing d'un phoneme dans l'audio synthetise."""
    ipa: str
    start_ms: float
    end_ms: float


@dataclass
class TTSResult:
    """Resultat d'une synthese TTS."""
    samples: np.ndarray  # float32, mono
    sample_rate: int
    phoneme_timings: list[PhonemeTiming] = field(default_factory=list)


class OnnxTTSEngine:
    """Engine TTS ONNX local multi-speaker.

    Supports two model types :
    - matcha-conformer : matcha_encoder_{speaker}.onnx + matcha_unet.onnx + hifigan.onnx
    - fastpitch (legacy) : encoder_{speaker}.onnx + decoder.onnx + hifigan.onnx

    Parameters
    ----------
    models_dir : Path
        Repertoire contenant les ONNX + config.json + phoneme_vocab.json
    speaker : str
        Nom du speaker initial (default: "siwis")
    """

    def __init__(self, models_dir: Path, speaker: str = "siwis") -> None:
        self._models_dir = models_dir
        self._speaker = speaker
        self._speaker_id = 0
        self._encoder = None
        self._decoder = None
        self._unet = None
        self._hifigan = None
        self._config: dict[str, Any] | None = None
        self._phone2id: dict[str, int] | None = None
        self._sample_rate = 22050
        self._unified = False
        self._model_type = "fastpitch"  # or "matcha-conformer"
        self._n_style_dims = 0
        self._speakers_map: dict[str, int] = {}
        self._g2p = None
        self._unet_has_spk = False

    @property
    def speaker(self) -> str:
        """Speaker actuellement selectionne."""
        return self._speaker

    def set_speaker(self, speaker: str) -> None:
        """Change le speaker actif.

        En mode unifie, change uniquement le speaker_id (pas de reload).
        En mode per-speaker, recharge l'encodeur ONNX correspondant.
        """
        if speaker == self._speaker and self._encoder is not None:
            return
        self._speaker = speaker
        if self._unified:
            self._speaker_id = self._speakers_map.get(speaker, 0)
        elif self._hifigan is not None:
            self._load_encoder(speaker)

    def _ensure_loaded(self) -> None:
        """Charge les sessions ONNX (lazy)."""
        if self._hifigan is not None and self._encoder is not None:
            return

        import onnxruntime as ort

        from lectura_multispeaker._chargeur import load_model_bytes

        # Charger config
        config_path = self._models_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            self._config = json.load(f)

        # Charger vocabulaire
        vocab_path = self._models_dir / "phoneme_vocab.json"
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._phone2id = vocab_data["phone2id"]

        # Build speakers map from speakers.json or config
        self._build_speakers_map()
        self._speaker_id = self._speakers_map.get(self._speaker, 0)
        self._n_style_dims = self._config.get("model", {}).get("n_style_dims", 0)

        # Detect model type from config
        model_type = self._config.get("model", {}).get("type", "fastpitch")
        self._model_type = model_type

        # Options ONNX
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        providers = ["CPUExecutionProvider"]

        # Charger hifigan (partage pour les deux types)
        hifi_bytes = load_model_bytes(self._models_dir, "hifigan.onnx")
        if hifi_bytes is None:
            raise FileNotFoundError(
                f"hifigan.onnx introuvable dans {self._models_dir}"
            )
        self._hifigan = ort.InferenceSession(hifi_bytes, sess_options=opts, providers=providers)

        if model_type == "matcha-conformer":
            # Matcha: matcha_unet.onnx (partage) + matcha_encoder_{speaker}.onnx (per-speaker)
            unet_bytes = load_model_bytes(self._models_dir, "matcha_unet.onnx")
            if unet_bytes is None:
                raise FileNotFoundError(
                    f"matcha_unet.onnx introuvable dans {self._models_dir}"
                )
            self._unet = ort.InferenceSession(unet_bytes, sess_options=opts, providers=providers)
            # Check if UNet expects spk_emb (speaker FiLM conditioning)
            unet_input_names = {inp.name for inp in self._unet.get_inputs()}
            self._unet_has_spk = "spk_emb" in unet_input_names

            # Detect unified vs per-speaker encoder
            unified_bytes = load_model_bytes(self._models_dir, "matcha_encoder.onnx")
            self._unified = unified_bytes is not None
            if self._unified:
                self._encoder = ort.InferenceSession(
                    unified_bytes, sess_options=opts, providers=providers)
            else:
                self._load_encoder(self._speaker)
        else:
            # FastPitch legacy: decoder.onnx + encoder_{speaker}.onnx
            dec_bytes = load_model_bytes(self._models_dir, "decoder.onnx")
            if dec_bytes is None:
                raise FileNotFoundError(
                    f"decoder.onnx introuvable dans {self._models_dir}"
                )
            self._decoder = ort.InferenceSession(dec_bytes, sess_options=opts, providers=providers)

            # Detect unified vs per-speaker encoder
            unified_bytes = load_model_bytes(self._models_dir, "encoder.onnx")
            self._unified = unified_bytes is not None
            if self._unified:
                self._encoder = ort.InferenceSession(
                    unified_bytes, sess_options=opts, providers=providers)
            else:
                self._load_encoder(self._speaker)

        self._sample_rate = self._config.get("audio", {}).get("sample_rate", 22050)
        log.info("OnnxTTSEngine multi-speaker charge depuis %s (type=%s, speaker=%s)",
                 self._models_dir, model_type, self._speaker)

    def _build_speakers_map(self) -> None:
        """Build name -> id mapping from speakers.json or config."""
        speakers_path = self._models_dir / "speakers.json"
        if speakers_path.exists():
            with open(speakers_path, encoding="utf-8") as f:
                data = json.load(f)
            for s in data.get("speakers", []):
                self._speakers_map[s["name"]] = s["id"]
        else:
            # Fallback: hardcoded
            for name, sid in [("siwis", 0), ("ezwa", 1), ("nadine", 2),
                              ("bernard", 3), ("gilles", 4), ("zeckou", 5)]:
                self._speakers_map[name] = sid

    def _load_encoder(self, speaker: str) -> None:
        """Charge l'encodeur ONNX pour un speaker specifique (per-speaker layout)."""
        import onnxruntime as ort

        from lectura_multispeaker._chargeur import load_model_bytes

        # Matcha: matcha_encoder_{speaker}.onnx, FastPitch: encoder_{speaker}.onnx
        if self._model_type == "matcha-conformer":
            encoder_name = f"matcha_encoder_{speaker}.onnx"
        else:
            encoder_name = f"encoder_{speaker}.onnx"

        enc_bytes = load_model_bytes(self._models_dir, encoder_name)

        if enc_bytes is None:
            raise FileNotFoundError(
                f"Encodeur introuvable pour speaker '{speaker}' dans {self._models_dir}"
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        self._encoder = ort.InferenceSession(
            enc_bytes, sess_options=opts, providers=["CPUExecutionProvider"]
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

        Args:
            style: Nom d'un preset de style (ex: "narratif", "dialogue")
            style_vector: Vecteur de style explicite [n_style_dims]
                         (prioritaire sur style)
            n_ode_steps: Nombre de pas ODE (Matcha uniquement, defaut: config)
            duration_noise: Bruit de duree lisse (0.0=off, 0.1=subtil, 0.2=prononce)
        """
        try:
            from lectura_g2p import creer_engine as creer_g2p
        except ImportError:
            raise ImportError(
                "lectura-g2p requis pour synthesize(text). "
                "Installer avec : pip install lectura-tts-multispeaker[g2p]"
            )

        if self._g2p is None:
            self._g2p = creer_g2p(mode="auto")

        sentences = _text_to_sentences(text, self._g2p)

        if not sentences:
            return TTSResult(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                phoneme_timings=[],
            )

        prosody = dict(duration_scale=duration_scale, pitch_shift=pitch_shift,
                       pitch_range=pitch_range, energy_scale=energy_scale,
                       pause_scale=pause_scale, variability=variability,
                       style=style, style_vector=style_vector,
                       n_ode_steps=n_ode_steps, duration_noise=duration_noise)

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
            time_offset_ms += len(result.samples) / self._sample_rate * 1000

        combined = np.concatenate(all_samples)
        return TTSResult(
            samples=combined,
            sample_rate=self._sample_rate,
            phoneme_timings=all_timings,
        )

    def _resolve_style(
        self,
        style: str | None = None,
        style_vector: list[float] | None = None,
    ) -> tuple[list[float], dict[str, float]]:
        """Resolve style to (vector, prosody_overrides).

        Priority: explicit style_vector > named preset > neutral default.
        When style_vector is given directly, no prosodic overrides are applied.
        """
        if style_vector is not None:
            return list(style_vector), {}
        if style is not None and style in STYLE_PRESETS:
            preset = STYLE_PRESETS[style]
            sv = list(preset["style_vector"])
            prosody = {k: v for k, v in preset.items() if k != "style_vector"}
            return sv, prosody
        n = self._n_style_dims if self._n_style_dims > 0 else 5
        return [0.0] * n, {}

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
        """Synthetise une sequence de phonemes IPA.

        Args:
            phonemes_ipa: Chaine IPA (ex: "bɔ̃ʒuʁ")
            phrase_type: 0=decl, 1=inter, 2=excl, 3=susp
            duration_scale: Multiplicateur de duree globale (None = preset ou 1.0)
            pitch_shift: Decalage F0 en demi-tons (None = preset ou 0.0)
            pitch_range: Echelle de variation F0 (None = preset ou 1.0)
            energy_scale: Multiplicateur d'energie (None = preset ou 1.0)
            pause_scale: Multiplicateur pour les pauses (None = preset ou 1.0)
            style: Nom d'un preset de style (ex: "narratif", "dialogue")
            style_vector: Vecteur de style explicite [n_style_dims]
            n_ode_steps: Nombre de pas ODE (Matcha uniquement, defaut: config)
            duration_noise: Bruit de duree lisse (0.0=off, 0.1=subtil, 0.2=prononce)
        """
        self._ensure_loaded()

        # Resolve style preset → style_vector + prosodic overrides
        sv, prosody_overrides = self._resolve_style(style, style_vector)

        # Apply prosodic values: explicit param > preset override > global default
        duration_scale = duration_scale if duration_scale is not None else prosody_overrides.get("duration_scale", _PROSODY_DEFAULTS["duration_scale"])
        pitch_shift = pitch_shift if pitch_shift is not None else prosody_overrides.get("pitch_shift", _PROSODY_DEFAULTS["pitch_shift"])
        pitch_range = pitch_range if pitch_range is not None else prosody_overrides.get("pitch_range", _PROSODY_DEFAULTS["pitch_range"])
        energy_scale = energy_scale if energy_scale is not None else prosody_overrides.get("energy_scale", _PROSODY_DEFAULTS["energy_scale"])
        pause_scale = pause_scale if pause_scale is not None else prosody_overrides.get("pause_scale", _PROSODY_DEFAULTS["pause_scale"])
        if duration_noise is None:
            duration_noise = prosody_overrides.get("duration_noise", 0.0)

        from lectura_multispeaker.phonemes import (
            ipa_to_phones, get_phone_min_frames, _PHONE_FALLBACKS,
        )
        from lectura_multispeaker._enhance import enhance_mel, noise_gate, fade_out

        # Convertir IPA -> phone IDs
        phones = ipa_to_phones(phonemes_ipa)

        # Ajouter un point final si la phrase ne finit pas par une ponctuation,
        # sinon le modele coupe brutalement (pas de release prosodique).
        if phones and phones[-1] not in _SILENCE_PHONES:
            phones.append(".")

        # Frontieres de mots
        space_after: list[int] = []
        segments = phonemes_ipa.split(" ")
        phone_count = 0
        for seg_idx, segment in enumerate(segments):
            if segment:
                phone_count += len(ipa_to_phones(segment))
            if seg_idx < len(segments) - 1 and phone_count > 0:
                space_after.append(phone_count - 1)

        # Inserer un micro-silence entre voyelles consecutives (hiatus)
        # UNIQUEMENT au sein d'un meme mot (pas entre mots).
        from lectura_multispeaker.phonemes import _VOWELS
        word_boundaries = set(space_after)
        patched: list[str] = [phones[0]] if phones else []
        inserted = 0
        new_space_after: list[int] = []
        for k in range(1, len(phones)):
            if (phones[k] in _VOWELS and phones[k - 1] in _VOWELS
                    and (k - 1) not in word_boundaries):
                patched.append("#")
                inserted += 1
            if (k - 1) in word_boundaries:
                new_space_after.append(k - 1 + inserted)
            patched.append(phones[k])
        # Dernier indice si c'etait une frontiere
        if (len(phones) - 1) in word_boundaries:
            new_space_after.append(len(phones) - 1 + inserted)
        phones = patched
        space_after = new_space_after

        sil_id = self._phone2id["#"]
        unk_id = self._phone2id.get("<UNK>", 1)

        def _resolve(p: str) -> int:
            pid = self._phone2id.get(p)
            if pid is not None:
                return pid
            fb = _PHONE_FALLBACKS.get(p)
            if fb is not None:
                return self._phone2id.get(fb, unk_id)
            return unk_id

        phone_ids = [sil_id] + [_resolve(p) for p in phones] + [sil_id]
        phone_ids_np = np.array([phone_ids], dtype=np.int64)
        phrase_type_np = np.array([phrase_type], dtype=np.int64)

        # 1. Encoder
        spk_emb = None
        if self._unified:
            speaker_id_np = np.array([self._speaker_id], dtype=np.int64)
            style_np = np.array([sv], dtype=np.float32)
            enc_outputs = self._encoder.run(None, {
                "phone_ids": phone_ids_np,
                "phrase_type": phrase_type_np,
                "speaker_id": speaker_id_np,
                "style_vector": style_np,
            })
            enc_out, dur_pred, pitch_pred, energy_pred = enc_outputs[:4]
            if len(enc_outputs) > 4:
                spk_emb = enc_outputs[4]
        elif self._model_type == "matcha-conformer":
            # Per-speaker Matcha encoder: style_vector is dynamic (not baked)
            style_np = np.array([sv], dtype=np.float32)
            enc_outputs = self._encoder.run(None, {
                "phone_ids": phone_ids_np,
                "phrase_type": phrase_type_np,
                "style_vector": style_np,
            })
            enc_out, dur_pred, pitch_pred, energy_pred = enc_outputs[:4]
            if len(enc_outputs) > 4:
                spk_emb = enc_outputs[4]
        else:
            # Legacy FastPitch per-speaker encoder (no style_vector input)
            enc_out, dur_pred, pitch_pred, energy_pred = self._encoder.run(None, {
                "phone_ids": phone_ids_np,
                "phrase_type": phrase_type_np,
            })

        # 2. Process predictions (numpy glue)
        dur_raw = np.exp(dur_pred[0])

        # Apply pause_scale aux SIL + ponctuation
        punct_ids = {self._phone2id.get(p, -1) for p in _PUNCT_MIN_FRAMES}
        pause_mask = np.zeros(len(phone_ids), dtype=bool)
        for idx, pid in enumerate(phone_ids):
            if pid == sil_id or pid in punct_ids:
                pause_mask[idx] = True
        if pause_scale != 1.0:
            dur_raw[pause_mask] *= pause_scale

        durations = np.maximum(1, np.round(dur_raw * duration_scale)).astype(np.int64)

        if variability:
            rng = np.random.default_rng()
            dur_noise_var = rng.normal(1.0, 0.10, size=durations.shape)
            dur_noise_var[pause_mask] = 1.0
            durations = np.maximum(1, np.round(durations * dur_noise_var)).astype(np.int64)

        # Bruit de duree lisse — variation de rythme naturelle
        if duration_noise and duration_noise > 0 and len(durations) > 2:
            rng = np.random.default_rng()
            raw = rng.normal(0.0, duration_noise, size=durations.shape)
            # Lissage 3-taps : les phones voisins accelerent/ralentissent ensemble
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
            smoothed = np.convolve(raw, kernel, mode="same")
            # Facteur multiplicatif (exp pour rester > 0, centre sur 1.0)
            dur_factor = np.exp(smoothed).astype(np.float64)
            # Ne pas toucher aux pauses/ponctuation
            dur_factor[pause_mask] = 1.0
            durations = np.maximum(
                1, np.round(durations * dur_factor)).astype(np.int64)

        # Durees minimales pour la ponctuation
        for idx, phone in enumerate(phones):
            min_frames = _PUNCT_MIN_FRAMES.get(phone)
            if min_frames is not None:
                durations[idx + 1] = max(durations[idx + 1], min_frames)

        # Duration floor par type de phone (evite les phones tronques)
        for idx, phone in enumerate(phones):
            frame_idx = idx + 1  # +1 pour le SIL initial
            min_dur = get_phone_min_frames(phone)
            durations[frame_idx] = max(durations[frame_idx], min_dur)

        # Duree minimale du dernier phone parle (position finale de mot).
        # Le modele multi sous-predit les consonnes finales sur mots isoles.
        _MIN_FINAL_PHONE_FRAMES = 8  # ~93 ms
        # Chercher le dernier phone parle (ni silence, ni ponctuation)
        all_phones_with_sil = ["#"] + list(phones) + ["#"]
        last_phone_idx = None
        for _idx in range(len(all_phones_with_sil) - 1, -1, -1):
            if all_phones_with_sil[_idx] not in _SILENCE_PHONES:
                last_phone_idx = _idx
                break
        if last_phone_idx is not None:
            durations[last_phone_idx] = max(durations[last_phone_idx], _MIN_FINAL_PHONE_FRAMES)

        # SIL final minimum (evite la coupure en fin de mot)
        _MIN_FINAL_SIL_FRAMES = 12  # ~139 ms a 22050 Hz / 256 hop
        durations[-1] = max(durations[-1], _MIN_FINAL_SIL_FRAMES)

        # Detection sequence courte (mots isoles / 2-3 mots)
        _spoken_mask = np.array(
            [False]  # SIL initial
            + [p not in _SILENCE_PHONES for p in phones]
            + [False],  # SIL final
            dtype=bool,
        )
        _n_spoken = int(_spoken_mask.sum())

        # Rallonger les phones parles avant les pauses/ponctuations — le
        # vocoder coupe souvent la fin des segments.
        _PRE_PAUSE_SCALE = 1.2
        all_phones = ["#"] + list(phones) + ["#"]

        # Avant chaque ponctuation interne
        for _j, _ph in enumerate(all_phones):
            if _ph in _PUNCT_MIN_FRAMES and _j > 1:
                # Dernier phone parle avant cette ponctuation
                _k = _j - 1
                if all_phones[_k] not in _SILENCE_PHONES:
                    durations[_k] = max(
                        durations[_k],
                        int(round(durations[_k] * _PRE_PAUSE_SCALE)),
                    )

        # Deux derniers phones parles (fin de phrase)
        _last_spoken = []
        for _j in range(len(all_phones) - 1, -1, -1):
            if all_phones[_j] not in _SILENCE_PHONES and all_phones[_j] not in _PUNCT_MIN_FRAMES:
                _last_spoken.append(_j)
                if len(_last_spoken) == 2:
                    break
        for _j in _last_spoken:
            durations[_j] = max(
                durations[_j],
                int(round(durations[_j] * _PRE_PAUSE_SCALE)),
            )

        _is_short_sequence = _n_spoken <= 15

        # Ralentissement des phones parles sur sequences courtes.
        # Facteur dependant de la longueur — siwis exclue (pas besoin).
        if _is_short_sequence and self._speaker != "siwis":
            if _n_spoken <= 3:
                _SLOW_FACTOR = 2.0
            elif _n_spoken <= 10:
                _SLOW_FACTOR = 1.3
            else:  # 11-15
                _SLOW_FACTOR = 1.15
            _spoken_indices = np.where(_spoken_mask)[0]
            for _si in _spoken_indices:
                durations[_si] = max(1, int(round(durations[_si] * _SLOW_FACTOR)))

        # Pitch avec shift et range
        pitch_mean = pitch_pred[0].mean()
        pitch_values = (
            pitch_mean
            + (pitch_pred[0] - pitch_mean) * pitch_range
            + pitch_shift * SEMITONE
        )

        # Clamper les valeurs aberrantes (phones non-SIL)
        # Le modele multi produit parfois des extremes sur les mots courts.
        _PITCH_CLAMP_SIGMA = 2.0
        speech_mask = ~pause_mask
        if speech_mask.any():
            sp = pitch_values[speech_mask]
            sp_mean = sp.mean()
            sp_std = max(sp.std(), 0.1)
            lo = sp_mean - _PITCH_CLAMP_SIGMA * sp_std
            hi = sp_mean + _PITCH_CLAMP_SIGMA * sp_std
            pitch_values = np.clip(pitch_values, lo, hi)

        # Contour prosodique pour sequences courtes (mots isoles / 2-3 mots)
        if _is_short_sequence:
            _CONTOUR_AMP = 0.5
            _spoken_indices = np.where(_spoken_mask)[0]
            _positions = np.linspace(0.0, 1.0, _n_spoken)
            # Demi-sinusoide asymetrique : pic a 60% de la sequence
            _stretched = np.where(
                _positions <= 0.6,
                _positions / 0.6 * 0.5,       # [0, 0.6] -> [0, 0.5]
                0.5 + (_positions - 0.6) / 0.4 * 0.5,  # [0.6, 1] -> [0.5, 1]
            )
            _contour = _CONTOUR_AMP * np.sin(np.pi * _stretched)
            _contour -= _contour.mean()  # centrer : montee puis descente sous baseline
            for _ci, _si in enumerate(_spoken_indices):
                pitch_values[_si] += _contour[_ci]

        if variability:
            pitch_values *= rng.normal(1.0, 0.03, size=pitch_values.shape)

        # Energy
        energy_values = energy_pred[0] * energy_scale

        if variability:
            energy_values *= rng.normal(1.0, 0.02, size=energy_values.shape)

        # 3. Embeddings (matmul simple — poids Conv1d(1, D, 1))
        emb = self._config["embeddings"]
        pitch_w = np.array(emb["pitch_emb_weight"], dtype=np.float32)
        pitch_b = np.array(emb["pitch_emb_bias"], dtype=np.float32)
        energy_w = np.array(emb["energy_emb_weight"], dtype=np.float32)
        energy_b = np.array(emb["energy_emb_bias"], dtype=np.float32)

        pitch_emb = np.outer(pitch_values, pitch_w) + pitch_b[np.newaxis, :]
        energy_emb = np.outer(energy_values, energy_w) + energy_b[np.newaxis, :]

        # 4. Enrichir + expand
        enriched = enc_out[0] + pitch_emb + energy_emb  # [T, D]
        expanded = np.repeat(enriched, durations, axis=0)  # [T_mel, D]

        # 5. Decoder / ODE sampling
        if self._model_type == "matcha-conformer":
            # Matcha: conditioning is (1, D, T_mel), then ODE sampling
            cond = expanded.T[np.newaxis].astype(np.float32)  # [1, D, T_mel]
            T_mel = cond.shape[2]
            mask = np.ones((1, 1, T_mel), dtype=np.float32)

            steps = n_ode_steps or self._config.get("model", {}).get(
                "n_ode_steps", 4)
            mel = self._matcha_ode_sample(cond, mask, n_steps=steps,
                                          spk_emb=spk_emb)
            mel_np = mel[0]  # [80, T_mel]
        else:
            # FastPitch legacy: decoder takes expanded embeddings
            expanded_batch = expanded[np.newaxis].astype(np.float32)  # [1, T_mel, D]
            mel = self._decoder.run(None, {"decoder_in": expanded_batch})[0]  # [1, 80, T_mel]
            mel_np = mel[0]  # [80, T_mel]

        # 6. Enhancement
        enh = self._config.get("enhance", {})
        mel_np = enhance_mel(
            mel_np,
            spectral_alpha=enh.get("spectral_alpha", 0.20),
            temporal_alpha=enh.get("temporal_alpha", 0.20),
        )
        mel_np = noise_gate(
            mel_np,
            threshold=enh.get("noise_gate_threshold", -8.0),
            silence_val=enh.get("silence_val", -11.5),
        )
        mel_np = fade_out(
            mel_np,
            n_frames=enh.get("fade_frames", 5),
            silence_val=enh.get("silence_val", -11.5),
        )

        # 7. Vocoder
        mel_input = mel_np[np.newaxis].astype(np.float32)
        audio = self._hifigan.run(None, {"mel": mel_input})[0]  # [1, 1, T_audio]
        audio = audio.squeeze()

        # 8. Post-traitement audio
        hop_length = self._config.get("audio", {}).get("hop_length", 256)
        audio = _zero_silence_regions(audio, phones, durations, hop_length)

        # Normaliser
        max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
        audio = np.clip(audio / max_val, -1.0, 1.0).astype(np.float32)

        # 9. Construire les timings phonemes
        timings = self._build_timings(phones, durations, hop_length, space_after)

        return TTSResult(
            samples=audio,
            sample_rate=self._sample_rate,
            phoneme_timings=timings,
        )

    def _matcha_ode_sample(
        self,
        cond: np.ndarray,
        mask: np.ndarray,
        n_steps: int = 4,
        temperature: float = 1.0,
        spk_emb: np.ndarray | None = None,
    ) -> np.ndarray:
        """Boucle ODE Euler en numpy, appelle le UNet ONNX N fois.

        Args:
            cond: [1, D, T_mel] conditioning encoder
            mask: [1, 1, T_mel] binary mask
            n_steps: nombre de pas ODE
            temperature: temperature du bruit initial
            spk_emb: [1, D] speaker embedding for FiLM conditioning (optional)

        Returns:
            mel: [1, 80, T_mel] mel-spectrogram genere
        """
        T_mel = cond.shape[2]

        # Bruit initial
        rng = np.random.default_rng()
        x = (rng.standard_normal((1, 80, T_mel)) * temperature).astype(np.float32)

        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = np.array([i / n_steps], dtype=np.float32)

            # Appel UNet ONNX
            unet_inputs = {
                "x_t": x,
                "t": t_val,
                "cond": cond,
                "mask": mask,
            }
            if spk_emb is not None and self._unet_has_spk:
                unet_inputs["spk_emb"] = spk_emb
            v = self._unet.run(None, unet_inputs)[0]

            x = x + v * dt

        # Appliquer le masque
        x = x * mask
        return x

    def _build_timings(
        self,
        phones: list[str],
        durations: np.ndarray,
        hop_length: int,
        space_after: list[int] | None = None,
    ) -> list[PhonemeTiming]:
        """Construit les timings phonemes depuis les durees predites."""
        timings: list[PhonemeTiming] = []

        offset = int(durations[0]) * hop_length

        for i, phone in enumerate(phones):
            dur_frames = int(durations[i + 1])
            dur_samples = dur_frames * hop_length
            start_ms = offset / self._sample_rate * 1000
            end_ms = (offset + dur_samples) / self._sample_rate * 1000
            timings.append(PhonemeTiming(ipa=phone, start_ms=start_ms, end_ms=end_ms))
            offset += dur_samples

        if space_after:
            sa = set(space_after)
            for idx in sorted(sa, reverse=True):
                if 0 <= idx < len(timings):
                    boundary_ms = timings[idx].end_ms
                    timings.insert(idx + 1, PhonemeTiming(
                        ipa=" ", start_ms=boundary_ms, end_ms=boundary_ms,
                    ))

        return timings
