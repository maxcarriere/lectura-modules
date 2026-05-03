"""Engine TTS ONNX local — inference FastPitch + HiFi-GAN via onnxruntime.

Charge 3 sessions ONNX (lazy) :
- fastpitch_encoder.onnx
- fastpitch_decoder.onnx
- hifigan.onnx

Glue numpy entre encoder et decoder (length regulation + embeddings).
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

# Ponctuation reconnue par le modele TTS
_PUNCT_MAP = {",": ",", ";": ",", ":": ",", ".": ".", "!": "!", "?": "?",
              "\u2026": "\u2026", "...": "\u2026"}


def _text_to_ipa(text: str, g2p) -> tuple[str, int]:
    """Convertit du texte en IPA avec ponctuation.

    Pipeline : Tokeniseur → G2P (mots seulement) → IPA avec ponctuation.

    Args:
        text: Texte francais.
        g2p: Engine G2P (lectura_nlp).

    Returns:
        (ipa_string, phrase_type) — phrase_type deduit de la ponctuation finale.
    """
    try:
        from lectura_tokeniseur import tokenise
    except ImportError:
        log.warning("lectura-tokeniseur non installe — ponctuation ignoree")
        tokens = text.split()
        if not tokens:
            return "", 0
        result = g2p.analyser(tokens)
        return "".join(result.get("g2p", [])), 0

    tokens = tokenise(text)

    words = [t.text for t in tokens if t.type.name == "MOT"]
    if not words:
        return "", 0

    g2p_result = g2p.analyser(words)
    ipa_list = g2p_result.get("g2p", [])

    ipa_parts: list[str] = []
    word_idx = 0
    for token in tokens:
        if token.type.name == "MOT":
            if word_idx < len(ipa_list):
                ipa_parts.append(ipa_list[word_idx])
                word_idx += 1
        elif token.type.name == "PONCTUATION":
            punct = _PUNCT_MAP.get(token.text.strip())
            if punct:
                ipa_parts.append(punct)

    ipa = "".join(ipa_parts)

    # Deduire phrase_type de la ponctuation finale
    phrase_type = 0  # declaratif
    stripped = text.rstrip()
    if stripped.endswith("?"):
        phrase_type = 1
    elif stripped.endswith("!"):
        phrase_type = 2
    elif stripped.endswith("\u2026") or stripped.endswith("..."):
        phrase_type = 3

    return ipa, phrase_type


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
    """Engine TTS ONNX local.

    Parameters
    ----------
    models_dir : Path
        Repertoire contenant les 3 ONNX + config.json + phoneme_vocab.json
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._encoder = None
        self._decoder = None
        self._hifigan = None
        self._config: dict[str, Any] | None = None
        self._phone2id: dict[str, int] | None = None
        self._sample_rate = 22050
        self._g2p = None

    def _ensure_loaded(self) -> None:
        """Charge les sessions ONNX (lazy)."""
        if self._encoder is not None:
            return

        import onnxruntime as ort

        from lectura_tts_monospeaker._chargeur import load_model_bytes

        # Charger config
        config_path = self._models_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            self._config = json.load(f)

        # Charger vocabulaire
        vocab_path = self._models_dir / "phoneme_vocab.json"
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)
        self._phone2id = vocab_data["phone2id"]

        # Options ONNX
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        providers = ["CPUExecutionProvider"]

        # Charger les 3 modeles (supporte .onnx ou .enc)
        enc_bytes = load_model_bytes(self._models_dir, "fastpitch_encoder.onnx")
        dec_bytes = load_model_bytes(self._models_dir, "fastpitch_decoder.onnx")
        hifi_bytes = load_model_bytes(self._models_dir, "hifigan.onnx")

        if enc_bytes is None or dec_bytes is None or hifi_bytes is None:
            raise FileNotFoundError(
                f"Modeles ONNX incomplets dans {self._models_dir}"
            )

        self._encoder = ort.InferenceSession(enc_bytes, sess_options=opts, providers=providers)
        self._decoder = ort.InferenceSession(dec_bytes, sess_options=opts, providers=providers)
        self._hifigan = ort.InferenceSession(hifi_bytes, sess_options=opts, providers=providers)

        self._sample_rate = self._config.get("audio", {}).get("sample_rate", 22050)
        log.info("OnnxTTSEngine charge depuis %s", self._models_dir)

    def synthesize(
        self,
        text: str,
        phrase_type: int | None = None,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
    ) -> TTSResult:
        """Synthetise du texte (necessite lectura-g2p).

        Pipeline : Tokeniseur → G2P → IPA avec ponctuation → synthesize_phonemes.

        Args:
            text: Texte francais a synthetiser.
            phrase_type: 0=decl, 1=inter, 2=excl, 3=susp (None=auto-detect).
            duration_scale: Multiplicateur de duree globale.
            pitch_shift: Decalage F0 en demi-tons.
            pitch_range: Echelle de variation F0.
            energy_scale: Multiplicateur d'energie.
            pause_scale: Multiplicateur pour les pauses.
        """
        try:
            from lectura_nlp import creer_engine as creer_g2p
        except ImportError:
            raise ImportError(
                "lectura-g2p requis pour synthesize(text). "
                "Installer avec : pip install lectura-tts-monospeaker[g2p]"
            )

        if self._g2p is None:
            self._g2p = creer_g2p(mode="auto")

        ipa, auto_phrase_type = _text_to_ipa(text, self._g2p)

        if not ipa:
            return TTSResult(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                phoneme_timings=[],
            )

        return self.synthesize_phonemes(
            ipa,
            phrase_type=phrase_type if phrase_type is not None else auto_phrase_type,
            duration_scale=duration_scale,
            pitch_shift=pitch_shift,
            pitch_range=pitch_range,
            energy_scale=energy_scale,
            pause_scale=pause_scale,
        )

    def synthesize_phonemes(
        self,
        phonemes_ipa: str,
        phrase_type: int = 0,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
    ) -> TTSResult:
        """Synthetise une sequence de phonemes IPA.

        Args:
            phonemes_ipa: Chaine IPA (ex: "bɔ̃ʒuʁ")
            phrase_type: 0=decl, 1=inter, 2=excl, 3=susp
            duration_scale: Multiplicateur de duree globale
            pitch_shift: Decalage F0 en demi-tons
            pitch_range: Echelle de variation F0
            energy_scale: Multiplicateur d'energie
            pause_scale: Multiplicateur pour les pauses

        Returns:
            TTSResult avec samples (float32), sample_rate, phoneme_timings
        """
        self._ensure_loaded()

        from lectura_tts_monospeaker.phonemes import ipa_to_phones
        from lectura_tts_monospeaker._enhance import enhance_mel, noise_gate, fade_out

        # Convertir IPA → phone IDs
        phones = ipa_to_phones(phonemes_ipa)
        sil_id = self._phone2id["#"]
        unk_id = self._phone2id.get("<UNK>", 1)

        phone_ids = [sil_id] + [self._phone2id.get(p, unk_id) for p in phones] + [sil_id]
        phone_ids_np = np.array([phone_ids], dtype=np.int64)
        phrase_type_np = np.array([phrase_type], dtype=np.int64)

        # 1. Encoder
        enc_out, dur_pred, pitch_pred, energy_pred = self._encoder.run(None, {
            "phone_ids": phone_ids_np,
            "phrase_type": phrase_type_np,
        })

        # 2. Process predictions (numpy glue)
        dur_raw = np.exp(dur_pred[0])

        # Apply pause_scale to SIL tokens
        if pause_scale != 1.0:
            sil_mask = np.array(phone_ids) == sil_id
            dur_raw[sil_mask] *= pause_scale

        durations = np.maximum(1, np.round(dur_raw * duration_scale)).astype(np.int64)

        # Pitch avec shift et range
        pitch_mean = pitch_pred[0].mean()
        pitch_values = (
            pitch_mean
            + (pitch_pred[0] - pitch_mean) * pitch_range
            + pitch_shift * SEMITONE
        )

        # Energy
        energy_values = energy_pred[0] * energy_scale

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
        expanded = expanded[np.newaxis].astype(np.float32)  # [1, T_mel, D]

        # 5. Decoder
        mel = self._decoder.run(None, {"decoder_in": expanded})[0]  # [1, 80, T_mel]
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

        # Normaliser
        max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
        audio = np.clip(audio / max_val, -1.0, 1.0).astype(np.float32)

        # 8. Construire les timings phonemes
        hop_length = self._config.get("audio", {}).get("hop_length", 256)
        timings = self._build_timings(phones, durations, hop_length)

        return TTSResult(
            samples=audio,
            sample_rate=self._sample_rate,
            phoneme_timings=timings,
        )

    def _build_timings(
        self,
        phones: list[str],
        durations: np.ndarray,
        hop_length: int,
    ) -> list[PhonemeTiming]:
        """Construit les timings phonemes depuis les durees predites."""
        timings: list[PhonemeTiming] = []
        sample_pos = 0

        # durations correspond a [SIL, ...phones..., SIL]
        # on skip le premier SIL
        offset = int(durations[0]) * hop_length

        for i, phone in enumerate(phones):
            dur_frames = int(durations[i + 1])  # +1 pour skip SIL initial
            dur_samples = dur_frames * hop_length
            start_ms = offset / self._sample_rate * 1000
            end_ms = (offset + dur_samples) / self._sample_rate * 1000
            timings.append(PhonemeTiming(ipa=phone, start_ms=start_ms, end_ms=end_ms))
            offset += dur_samples

        return timings
