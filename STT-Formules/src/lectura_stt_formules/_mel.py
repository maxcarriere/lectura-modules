"""Extraction mel spectrogram (numpy only, sans torch/torchaudio).

Utilise uniquement numpy + soundfile pour l'inference legere.
Parametrage identique au training (dataset.py) :
  n_fft=512, hop=160, win=400, n_mels=80, fmin=0, fmax=8000, power=2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# Constantes mel (identiques au training)
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 80
FMIN = 0.0
FMAX = 8000.0


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Charge un WAV, resample en 16 kHz mono float32.

    Args:
        path: chemin vers le fichier audio (WAV, FLAC, OGG, etc.)
        sample_rate: frequence cible (defaut 16000)

    Returns:
        ndarray float32 1-D de shape (N,)
    """
    data, sr = sf.read(str(path), dtype="float32")

    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample si necessaire (interpolation lineaire simple)
    if sr != sample_rate:
        duration = len(data) / sr
        n_target = int(duration * sample_rate)
        indices = np.linspace(0, len(data) - 1, n_target)
        idx_low = np.floor(indices).astype(int)
        idx_high = np.minimum(idx_low + 1, len(data) - 1)
        frac = indices - idx_low
        data = data[idx_low] * (1 - frac) + data[idx_high] * frac
        data = data.astype(np.float32)

    return data


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Construit un banc de filtres mel triangulaires.

    Returns:
        ndarray float32 de shape (n_mels, n_fft // 2 + 1)
    """
    n_freqs = n_fft // 2 + 1

    # Hz -> Mel (formule HTK)
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    # Indices de bins frequentiels
    bin_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        f_left = hz_points[i]
        f_center = hz_points[i + 1]
        f_right = hz_points[i + 2]

        # Rampe montante
        up = (bin_freqs - f_left) / max(f_center - f_left, 1e-10)
        # Rampe descendante
        down = (f_right - bin_freqs) / max(f_right - f_center, 1e-10)

        filterbank[i] = np.maximum(0, np.minimum(up, down))

    return filterbank


def compute_log_mel(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Calcule le log-mel spectrogram.

    Meme parametrage que le training (torchaudio MelSpectrogram + log).

    Args:
        waveform: signal 1-D float32 de shape (N,)
        sample_rate: frequence d'echantillonnage
        n_fft: taille de la FFT
        hop_length: pas de la fenetre glissante
        win_length: taille de la fenetre
        n_mels: nombre de bins mel
        fmin: frequence minimale
        fmax: frequence maximale

    Returns:
        ndarray float32 de shape (1, n_mels, T)
    """
    # Fenetre de Hann
    window = np.hanning(win_length).astype(np.float32)

    # Padding pour centrer (identique a torch.stft center=True par defaut)
    pad_len = n_fft // 2
    waveform = np.pad(waveform, (pad_len, pad_len), mode="reflect")

    # Nombre de frames
    n_frames = 1 + (len(waveform) - n_fft) // hop_length

    # STFT
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        start = t * hop_length
        frame = waveform[start : start + n_fft]
        # Appliquer la fenetre (zero-pad si win_length < n_fft)
        windowed = np.zeros(n_fft, dtype=np.float32)
        windowed[:win_length] = frame[:win_length] * window
        spectrum = np.fft.rfft(windowed)
        stft[:, t] = spectrum

    # Power spectrogram (magnitude au carre, power=2.0)
    power_spec = np.abs(stft) ** 2

    # Banc de filtres mel
    mel_fb = _mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)

    # Mel spectrogram : (n_mels, T)
    mel_spec = mel_fb @ power_spec

    # Log mel
    log_mel = np.log(mel_spec + 1e-8)

    # Shape (1, n_mels, T)
    return log_mel[np.newaxis, :, :].astype(np.float32)
