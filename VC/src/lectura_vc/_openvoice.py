"""OpenVoice v2 Voice Conversion via ONNX Runtime (no PyTorch).

Zero-shot voice conversion using pre-exported ONNX models:
  - openvoice_se.onnx   : Speaker Embedding extractor (ref_enc)
  - openvoice_vc.onnx   : Voice Converter (enc_q + flow + dec)

Audio is processed at 22050 Hz.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort

from lectura_vc._chargeur import load_model_bytes

logger = logging.getLogger(__name__)

# OpenVoice v2 audio parameters (from config.json)
OV_SR = 22050
OV_N_FFT = 1024
OV_HOP = 256
OV_WIN = 1024
OV_N_FREQ = OV_N_FFT // 2 + 1  # 513


# ── Spectrogram (numpy, matching OpenVoice's spectrogram_torch) ──────────────

def compute_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute magnitude spectrogram matching OpenVoice.

    Parameters
    ----------
    audio : 1-D float32 at 22050 Hz.

    Returns
    -------
    np.ndarray shape (1, 513, T) float32.
    """
    # Manual reflect padding (same as OpenVoice spectrogram_torch)
    pad = (OV_N_FFT - OV_HOP) // 2  # 384
    audio_padded = np.pad(audio.astype(np.float32), (pad, pad), mode="reflect")

    S = librosa.stft(
        audio_padded,
        n_fft=OV_N_FFT,
        hop_length=OV_HOP,
        win_length=OV_WIN,
        center=False,
    )
    # sqrt(real^2 + imag^2 + 1e-6) matching PyTorch implementation
    magnitude = np.sqrt(np.abs(S) ** 2 + 1e-6).astype(np.float32)
    return magnitude[np.newaxis, :, :]  # (1, 513, T)


# ── OpenVoice Converter ──────────────────────────────────────────────────────

class OpenVoiceConverter:
    """Zero-shot voice conversion using OpenVoice v2 ONNX models."""

    def __init__(self, models_dir: Path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]

        # Speaker embedding extractor
        se_data = load_model_bytes(models_dir, "openvoice_se.onnx")
        if se_data is None:
            raise FileNotFoundError(f"openvoice_se.onnx introuvable dans {models_dir}")
        logger.info("Chargement OpenVoice SE ONNX...")
        self.se_sess = ort.InferenceSession(se_data, opts, providers=providers)

        # Voice converter
        vc_data = load_model_bytes(models_dir, "openvoice_vc.onnx")
        if vc_data is None:
            raise FileNotFoundError(f"openvoice_vc.onnx introuvable dans {models_dir}")
        logger.info("Chargement OpenVoice VC ONNX...")
        self.vc_sess = ort.InferenceSession(vc_data, opts, providers=providers)

        self.sr = OV_SR

    def extract_se(
        self,
        audio: np.ndarray | str | Path,
        sr: int | None = None,
    ) -> np.ndarray:
        """Extract speaker embedding from reference audio.

        Parameters
        ----------
        audio : 1-D float32 array or path to audio file.
        sr : sample rate (auto-detected if audio is a file path).

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        audio_np = self._load_audio(audio, sr)
        spec = compute_spectrogram(audio_np)  # (1, 513, T)

        # ref_enc expects (1, T, 513)
        spec_t = spec.transpose(0, 2, 1).astype(np.float32)

        se = self.se_sess.run(
            None,
            {"spec": spec_t},
        )[0]  # (1, 256) or (1, 256, 1)

        if se.ndim == 2:
            se = se[:, :, np.newaxis]

        return se.astype(np.float32)

    def extract_se_multi(
        self,
        audio_list: list[np.ndarray | str | Path],
        sr: int | None = None,
    ) -> np.ndarray:
        """Extract and average speaker embeddings from multiple references.

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        ses = []
        for audio in audio_list:
            se = self.extract_se(audio, sr=sr)
            ses.append(se)
        return np.mean(ses, axis=0).astype(np.float32)

    def convert(
        self,
        audio: np.ndarray | str | Path,
        src_se: np.ndarray,
        tgt_se: np.ndarray,
        sr: int | None = None,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Convert voice from source to target speaker.

        Parameters
        ----------
        audio : source audio (1-D float32 or path).
        src_se : source speaker embedding (1, 256, 1).
        tgt_se : target speaker embedding (1, 256, 1).
        sr : sample rate of source (auto if file path).
        tau : conversion strength (0 = deterministic, higher = more variation).

        Returns
        -------
        (converted_audio, 22050)
        """
        audio_np = self._load_audio(audio, sr)
        spec = compute_spectrogram(audio_np)  # (1, 513, T)
        spec_lengths = np.array([spec.shape[2]], dtype=np.int64)

        # Ensure SE shapes
        src_se = src_se.astype(np.float32).reshape(1, 256, 1)
        tgt_se = tgt_se.astype(np.float32).reshape(1, 256, 1)

        result = self.vc_sess.run(
            None,
            {
                "spec": spec,
                "spec_lengths": spec_lengths,
                "sid_src": src_se,
                "sid_tgt": tgt_se,
            },
        )[0]  # (1, 1, T_audio)

        audio_out = result[0, 0].astype(np.float32)
        return audio_out, self.sr

    def _load_audio(self, audio: np.ndarray | str | Path, sr: int | None) -> np.ndarray:
        """Load and resample audio to 22050 Hz."""
        if isinstance(audio, (str, Path)):
            audio_np, _ = librosa.load(str(audio), sr=self.sr)
        else:
            audio_np = audio.astype(np.float32)
            if sr is not None and sr != self.sr:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=self.sr)
        return audio_np
