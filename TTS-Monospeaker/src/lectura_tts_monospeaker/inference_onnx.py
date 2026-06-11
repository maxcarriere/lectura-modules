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

# Durees minimales en frames pour la ponctuation (1 frame ≈ 11.6 ms)
_PUNCT_MIN_FRAMES = {",": 10, ".": 20, "?": 15, "!": 15, "\u2026": 25}

# Phones correspondant a des silences (zeroes dans l'audio)
_SILENCE_PHONES = {"#", ",", ".", "?", "!", "\u2026"}


def _zero_silence_regions(
    audio: np.ndarray,
    phones: list[str],
    durations: np.ndarray,
    hop_length: int,
    fade_samples: int = 128,
) -> np.ndarray:
    """Remplace les zones SIL/ponctuation par du vrai silence.

    Utilise les durees predites pour identifier precisement les regions
    de silence et de ponctuation, puis les met a zero avec un court
    fondu aux transitions pour eviter les clics.

    Args:
        audio: Signal float32 mono.
        phones: Liste des phones (sans les SIL aux extremites).
        durations: Durees en frames [SIL, phone1, ..., phoneN, SIL].
        hop_length: Nombre d'echantillons par frame.
        fade_samples: Longueur du fondu aux transitions.
    """
    if len(audio) == 0:
        return audio

    # Construire le masque : 1.0 = parole, 0.0 = silence
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

    # Lisser les transitions (fondu dans la zone de silence)
    fade = fade_samples
    diff = np.diff(mask, prepend=mask[0])

    # Transitions 1→0 (parole → silence) : fondu au debut du silence
    for idx in np.where(diff < -0.5)[0]:
        e = min(len(audio), idx + fade)
        n = e - idx
        if n > 0:
            mask[idx:e] = np.linspace(1.0, 0.0, n)

    # Transitions 0→1 (silence → parole) : fondu a la fin du silence
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

    _retimbre_engine = None  # RetimbreEngine (lazy, instance-level)

    def synthesize(
        self,
        text: str,
        phrase_type: int | None = None,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
        variability: bool = False,
        voix: str | Path | list[str] | dict[str, float] | None = None,
        voix_variante: float = 0.0,
        voix_tau: float = 0.3,
        vc_models_dir: str | Path | None = None,
    ) -> TTSResult:
        """Synthetise du texte (necessite lectura-g2p).

        Pipeline : Tokeniseur → decoupage en phrases → G2P → IPA → synthese
        par phrase avec phrase_type individuel → concatenation.

        Args:
            text: Texte francais a synthetiser.
            phrase_type: 0=decl, 1=inter, 2=excl, 3=susp (None=auto-detect).
            duration_scale: Multiplicateur de duree globale.
            pitch_shift: Decalage F0 en demi-tons.
            pitch_range: Echelle de variation F0.
            energy_scale: Multiplicateur d'energie.
            pause_scale: Multiplicateur pour les pauses.
            voix: Voix cible pour retimbre OpenVoice. Polymorphe :
                - str : nom de preset ("siwis") ou chemin fichier audio.
                - list[str] : plusieurs references (poids egaux).
                - dict[str, float] : blend pondere.
                None = pas de retimbre.
                Requires: pip install 'lectura-tts-monospeaker[vc]'
            voix_variante: Curseur de variante vocale (-1 a +1).
                -1 = grave/masculin, 0 = neutre, +1 = aigu/enfant.
            voix_tau: Parametre tau d'OpenVoice (0 = deterministe).
            vc_models_dir: Repertoire des modeles VC (defaut: auto-detection).
        """
        try:
            from lectura_g2p import creer_engine as creer_g2p
        except ImportError:
            raise ImportError(
                "lectura-g2p requis pour synthesize(text). "
                "Installer avec : pip install lectura-tts-monospeaker[g2p]"
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
                       pause_scale=pause_scale, variability=variability)

        # Phrase unique — pas de concatenation
        if len(sentences) == 1:
            ipa, auto_pt = sentences[0]
            result = self.synthesize_phonemes(
                ipa,
                phrase_type=phrase_type if phrase_type is not None else auto_pt,
                **prosody,
            )
        else:
            # Multi-phrases : synthetiser chacune et concatener
            all_samples: list[np.ndarray] = []
            all_timings: list[PhonemeTiming] = []
            time_offset_ms = 0.0

            for i, (ipa, auto_pt) in enumerate(sentences):
                r = self.synthesize_phonemes(
                    ipa,
                    phrase_type=phrase_type if phrase_type is not None else auto_pt,
                    **prosody,
                )

                # Decaler les timings
                for t in r.phoneme_timings:
                    all_timings.append(PhonemeTiming(
                        ipa=t.ipa,
                        start_ms=t.start_ms + time_offset_ms,
                        end_ms=t.end_ms + time_offset_ms,
                    ))

                all_samples.append(r.samples)
                time_offset_ms += len(r.samples) / self._sample_rate * 1000

            combined = np.concatenate(all_samples)
            result = TTSResult(
                samples=combined,
                sample_rate=self._sample_rate,
                phoneme_timings=all_timings,
            )

        # Retimbre optionnel (OpenVoice zero-shot)
        if voix is not None:
            result = self._apply_retimbre(
                result, voix, voix_variante, voix_tau, vc_models_dir,
            )

        return result

    def synthesize_phonemes(
        self,
        phonemes_ipa: str,
        phrase_type: int = 0,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.3,
        energy_scale: float = 1.0,
        pause_scale: float = 1.0,
        variability: bool = False,
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

        from lectura_tts_monospeaker.phonemes import (
            ipa_to_phones, get_phone_min_frames, _PHONE_FALLBACKS,
        )
        from lectura_tts_monospeaker._enhance import enhance_mel, noise_gate, fade_out

        # Convertir IPA → phone IDs
        phones = ipa_to_phones(phonemes_ipa)

        # Reperer les frontieres de mots (espaces dans l'IPA) pour les timings
        space_after: list[int] = []
        segments = phonemes_ipa.split(" ")
        phone_count = 0
        for seg_idx, segment in enumerate(segments):
            if segment:
                phone_count += len(ipa_to_phones(segment))
            if seg_idx < len(segments) - 1 and phone_count > 0:
                space_after.append(phone_count - 1)

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
            dur_noise = rng.normal(1.0, 0.10, size=durations.shape)
            dur_noise[pause_mask] = 1.0
            durations = np.maximum(1, np.round(durations * dur_noise)).astype(np.int64)

        # Durees minimales pour la ponctuation
        for idx, phone in enumerate(phones):
            min_frames = _PUNCT_MIN_FRAMES.get(phone)
            if min_frames is not None:
                durations[idx + 1] = max(durations[idx + 1], min_frames)  # +1 pour SIL

        # Duration floor par type de phone (evite les phones tronques)
        for idx, phone in enumerate(phones):
            frame_idx = idx + 1  # +1 pour le SIL initial
            min_dur = get_phone_min_frames(phone)
            durations[frame_idx] = max(durations[frame_idx], min_dur)

        # Pitch avec shift et range
        pitch_mean = pitch_pred[0].mean()
        pitch_values = (
            pitch_mean
            + (pitch_pred[0] - pitch_mean) * pitch_range
            + pitch_shift * SEMITONE
        )

        # Contour prosodique pour sequences courtes (mots isoles / 2-3 mots)
        # Detecte les phones parles (hors SIL et ponctuation)
        _spoken_mask = np.array(
            [False]  # SIL initial
            + [p not in _SILENCE_PHONES for p in phones]
            + [False],  # SIL final
            dtype=bool,
        )
        _n_spoken = int(_spoken_mask.sum())
        _has_punct = any(p in _PUNCT_MIN_FRAMES for p in phones)
        if _n_spoken <= 10 and not _has_punct and _n_spoken >= 2:
            # Demi-sinusoide asymetrique : pic a 60% de la sequence
            _CONTOUR_AMP = 0.3
            _spoken_indices = np.where(_spoken_mask)[0]
            _positions = np.linspace(0.0, 1.0, _n_spoken)
            # Etirer pour que sin(pi * x_stretched) atteigne son max a 60%
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

        # 8. Post-traitement audio
        hop_length = self._config.get("audio", {}).get("hop_length", 256)

        # Zeroing des silences/ponctuation (base sur les durees predites)
        audio = _zero_silence_regions(audio, phones, durations, hop_length)

        # Normaliser
        max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
        audio = np.clip(audio / max_val, -1.0, 1.0).astype(np.float32)

        # 9. Construire les timings phonemes (avec espaces aux frontieres de mots)
        timings = self._build_timings(phones, durations, hop_length, space_after)

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
        space_after: list[int] | None = None,
    ) -> list[PhonemeTiming]:
        """Construit les timings phonemes depuis les durees predites.

        Si space_after est fourni, insere des PhonemeTiming(ipa=" ")
        aux frontieres de mots pour le DTW d'alignement.
        """
        timings: list[PhonemeTiming] = []

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

        # Inserer les espaces aux frontieres de mots (du dernier au premier)
        if space_after:
            sa = set(space_after)
            for idx in sorted(sa, reverse=True):
                if 0 <= idx < len(timings):
                    boundary_ms = timings[idx].end_ms
                    timings.insert(idx + 1, PhonemeTiming(
                        ipa=" ", start_ms=boundary_ms, end_ms=boundary_ms,
                    ))

        return timings

    def _apply_retimbre(
        self,
        result: TTSResult,
        voix: str | Path | list[str] | dict[str, float],
        voix_variante: float,
        voix_tau: float,
        vc_models_dir: str | Path | None,
    ) -> TTSResult:
        """Applique le retimbre OpenVoice sur le resultat TTS."""
        from lectura_tts_monospeaker._retimbre import RetimbreEngine

        if self._retimbre_engine is None:
            self._retimbre_engine = RetimbreEngine(vc_models_dir=vc_models_dir)

        audio, sr = self._retimbre_engine.retimbre(
            result.samples, self._sample_rate, voix,
            variante=voix_variante, tau=voix_tau,
        )

        # Normaliser
        peak = np.max(np.abs(audio))
        if peak > 0.01:
            audio = (audio * 0.9 / peak).astype(np.float32)

        return TTSResult(
            samples=audio,
            sample_rate=sr,
            phoneme_timings=result.phoneme_timings,
        )
