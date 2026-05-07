"""RVC Voice Conversion via ONNX Runtime (no PyTorch).

Pipeline: HuBERT (content) + RMVPE (pitch) + Synthesizer (conversion).
Adapted from rvc_onnx_infer.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort

from lectura_vc._chargeur import load_model_bytes, get_model_path

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SR_16K = 16000
SR_48K = 48000

# Synthesizer ONNX: fixed T due to attention reshape constraints
SYNTH_FIXED_T = 1000

# RMVPE mel spectrogram parameters
MEL_N_FFT = 1024
MEL_HOP = 160
MEL_WIN = 1024
MEL_N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SR_16K // 2  # 8000

# RMVPE pitch decoding
N_CLASS = 360
MAGIC_CONST = 1997.3794084376191
CENTS_MAPPING = np.pad(20 * np.arange(N_CLASS) + MAGIC_CONST, (4, 4))  # 368

# F0 binning
F0_BIN = 256
F0_MIN = 50.0
F0_MAX = 1100.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


# ── Mel spectrogram ──────────────────────────────────────────────────────────

_mel_basis_cache: np.ndarray | None = None


def _get_mel_basis() -> np.ndarray:
    """Compute and cache the mel filterbank (128 x 513)."""
    global _mel_basis_cache
    if _mel_basis_cache is None:
        _mel_basis_cache = librosa.filters.mel(
            sr=SR_16K, n_fft=MEL_N_FFT, n_mels=MEL_N_MELS,
            fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True,
        ).astype(np.float32)
    return _mel_basis_cache


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram matching RMVPE.

    Returns shape (1, 128, n_frames) float32.
    """
    S = librosa.stft(
        audio.astype(np.float32),
        n_fft=MEL_N_FFT, hop_length=MEL_HOP, win_length=MEL_WIN,
        center=True,
    )
    magnitude = np.abs(S)
    mel_basis = _get_mel_basis()
    mel = mel_basis @ magnitude
    log_mel = np.log(np.clip(mel, 1e-5, None))
    return log_mel[np.newaxis, :, :].astype(np.float32)


# ── RMVPE pitch decoding ─────────────────────────────────────────────────────

def decode_f0(hidden: np.ndarray, thred: float = 0.03) -> np.ndarray:
    """Decode salience map to F0 in Hz.

    Parameters
    ----------
    hidden : shape (n_frames, 360)
    thred : voicing threshold

    Returns
    -------
    F0 in Hz, shape (n_frames,). 0 = unvoiced.
    """
    center = np.argmax(hidden, axis=1)
    salience = np.pad(hidden, ((0, 0), (4, 4)))
    center += 4

    todo_salience = []
    todo_cents = []
    starts = center - 4
    ends = center + 5
    for idx in range(salience.shape[0]):
        todo_salience.append(salience[idx, starts[idx]:ends[idx]])
        todo_cents.append(CENTS_MAPPING[starts[idx]:ends[idx]])

    todo_salience = np.array(todo_salience)
    todo_cents = np.array(todo_cents)

    product_sum = np.sum(todo_salience * todo_cents, axis=1)
    weight_sum = np.sum(todo_salience, axis=1)
    cents_pred = product_sum / weight_sum

    maxx = np.max(hidden, axis=1)
    cents_pred[maxx <= thred] = 0

    f0 = 10 * (2 ** (cents_pred / 1200))
    f0[f0 == 10] = 0
    return f0


def calculate_f0_bins(f0nsf: np.ndarray) -> np.ndarray:
    """Quantize F0 (Hz) to mel-scale bins [1, 255]."""
    f0_mel = 1127 * np.log(1 + f0nsf / 700)
    f0_mel[f0_mel > 0] = (
        (f0_mel[f0_mel > 0] - F0_MEL_MIN) * (F0_BIN - 2)
        / (F0_MEL_MAX - F0_MEL_MIN) + 1
    )
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > F0_BIN - 1] = F0_BIN - 1
    return np.rint(f0_mel).astype(np.int64)


# ── HuBERT pre-processing ────────────────────────────────────────────────────

def normalize_audio_for_hubert(audio: np.ndarray) -> np.ndarray:
    """Wav2Vec2FeatureExtractor normalization: zero-mean, unit-variance."""
    audio = audio.astype(np.float32)
    mean = np.mean(audio)
    var = np.var(audio)
    return (audio - mean) / np.sqrt(var + 1e-7)


# ── RVC Converter ─────────────────────────────────────────────────────────────

class RVCConverter:
    """RVC voice conversion using ONNX Runtime."""

    def __init__(
        self,
        models_dir: Path,
        speaker: str,
        *,
        segment_size: float = 7.0,
    ):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]

        # Load HuBERT
        hubert_data = load_model_bytes(models_dir, "hubert.onnx")
        if hubert_data is None:
            raise FileNotFoundError(f"hubert.onnx introuvable dans {models_dir}")
        logger.info("Chargement HuBERT ONNX...")
        self.hubert_sess = ort.InferenceSession(hubert_data, opts, providers=providers)

        # Load RMVPE
        rmvpe_data = load_model_bytes(models_dir, "rmvpe.onnx")
        if rmvpe_data is None:
            raise FileNotFoundError(f"rmvpe.onnx introuvable dans {models_dir}")
        logger.info("Chargement RMVPE ONNX...")
        self.rmvpe_sess = ort.InferenceSession(rmvpe_data, opts, providers=providers)

        # Load Synthesizer
        synth_name = f"synthesizer_{speaker}.onnx"
        synth_data = load_model_bytes(models_dir, synth_name)
        if synth_data is None:
            raise FileNotFoundError(f"{synth_name} introuvable dans {models_dir}")
        logger.info("Chargement Synthesizer ONNX (%s)...", speaker)
        self.synth_sess = ort.InferenceSession(synth_data, opts, providers=providers)

        self.speaker = speaker
        self.sr = SR_48K
        self.segment_size = segment_size

    # ── Feature extraction ────────────────────────────────────────────────

    def extract_hubert_features(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract HuBERT content features. Returns (n_frames, 768)."""
        input_values = normalize_audio_for_hubert(audio_16k)
        input_values = input_values[np.newaxis, :]

        outputs = self.hubert_sess.run(
            ["last_hidden_state"],
            {"input_values": input_values.astype(np.float32)},
        )
        feats = outputs[0][0]
        return np.nan_to_num(feats).astype(np.float32)

    def extract_f0(self, audio_16k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract F0 using RMVPE. Returns (f0_hz, f0_bins)."""
        mel = compute_mel(audio_16k)
        n_frames = mel.shape[2]

        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, n_pad)), mode="constant")

        hidden = self.rmvpe_sess.run(
            ["hidden"],
            {"mel": mel},
        )[0]

        hidden = hidden[0, :n_frames, :]
        f0nsf = decode_f0(hidden, thred=0.03)
        f0 = calculate_f0_bins(f0nsf)
        return f0nsf, f0

    # ── Synthesis ─────────────────────────────────────────────────────────

    def convert_from_features(
        self,
        phone: np.ndarray,
        pitchf: np.ndarray,
        pitch: np.ndarray,
        protect: float = 0.5,
        pitch_modification: float = 0.0,
    ) -> np.ndarray:
        """Convert features to audio at 48kHz."""
        use_protect = protect < 0.5

        if not np.isclose(pitch_modification, 0.0):
            pitchf = pitchf * pow(2, pitch_modification / 12)
            pitch = calculate_f0_bins(pitchf)

        phone = phone[np.newaxis, :, :]
        pitchf = pitchf[np.newaxis, :]
        pitch = pitch[np.newaxis, :]

        # 2x interpolation (nearest, matching PyTorch F.interpolate)
        feats = np.repeat(phone, 2, axis=1)

        if use_protect:
            feats0 = feats.copy()

        phone_len = feats.shape[1]
        pitch = pitch[:, :phone_len]
        pitchf = pitchf[:, :phone_len]

        if use_protect:
            pitchff = pitchf.copy()
            pitchff[pitchf > 0] = 1.0
            pitchff[pitchf < 1] = protect
            pitchff = pitchff[:, :, np.newaxis]
            feats = feats * pitchff + feats0 * (1 - pitchff)

        actual_phone_len = phone_len
        if phone_len > SYNTH_FIXED_T:
            raise ValueError(
                f"Phone length {phone_len} > ONNX fixed T={SYNTH_FIXED_T}. "
                f"Decoupez en segments plus courts."
            )

        pad_len = SYNTH_FIXED_T - phone_len
        if pad_len > 0:
            feats = np.pad(feats, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
            pitch = np.pad(pitch, ((0, 0), (0, pad_len)), mode="constant", constant_values=0)
            pitchf = np.pad(pitchf, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0)

        phone_lengths = np.array([actual_phone_len], dtype=np.int64)
        sid = np.array([0], dtype=np.int64)

        audio = self.synth_sess.run(
            ["audio"],
            {
                "phone": feats.astype(np.float32),
                "phone_lengths": phone_lengths,
                "pitch": pitch.astype(np.int64),
                "pitchf": pitchf.astype(np.float32),
                "sid": sid,
            },
        )[0]

        # Upsample factor: 12*10*2*2 = 480
        expected_audio_len = actual_phone_len * 480
        audio = audio[0, 0, :expected_audio_len].astype(np.float32)
        return audio

    # ── Full pipeline ─────────────────────────────────────────────────────

    def convert(
        self,
        audio: np.ndarray,
        sr: int = SR_16K,
        protect: float = 0.5,
        pitch_modification: float = 0.0,
    ) -> tuple[np.ndarray, int]:
        """Convert audio to target voice.

        Parameters
        ----------
        audio : 1-D float32 audio.
        sr : sample rate of input audio.
        protect : voice protection (<0.5 blends source+target in unvoiced).
        pitch_modification : semitone shift.

        Returns
        -------
        (converted_audio_48kHz, 48000)
        """
        # Resample to 16kHz if needed
        if sr != SR_16K:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR_16K)

        ret = []
        segment_size = int(self.segment_size * SR_16K)

        for i in range(0, len(audio), segment_size):
            segment = audio[i: i + segment_size]
            segment = np.pad(segment, (SR_16K, SR_16K), mode="reflect")

            logger.debug("Segment %d: %d samples", i // segment_size, len(segment))

            pitchf, pitch = self.extract_f0(segment)
            phone = self.extract_hubert_features(segment)

            seg_audio = self.convert_from_features(
                phone, pitchf, pitch, protect, pitch_modification
            )

            # Remove 1s padding on each side
            seg_audio = seg_audio[self.sr: -self.sr]
            ret.append(seg_audio)

        return np.concatenate(ret), self.sr
