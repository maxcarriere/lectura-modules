"""Extraction mel-spectrogram en numpy pur.

Parametres alignes sur l'entrainement (dataset.py) :
  sample_rate=16000, n_fft=512, hop_length=160, win_length=400
  n_mels=80, fmin=0, fmax=8000, power=2.0, log(mel + 1e-8)

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np

# ── Constantes (alignees sur l'entrainement) ─────────────────────────
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 80
FMIN = 0
FMAX = 8000


def _hz_to_mel(hz: float) -> float:
    """Conversion Hz → mel (formule HTK)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Conversion mel → Hz (formule HTK)."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _creer_filtres_mel(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Cree la matrice de filtres mel triangulaires (n_freqs, n_mels).

    Reproduit exactement l'algorithme de torchaudio.functional.melscale_fbanks
    avec norm="slaney" et mel_scale="htk".
    """
    n_freqs = n_fft // 2 + 1

    # Grille de frequences STFT (Hz) — linspace continu, pas d'arrondi en bins
    all_freqs = np.linspace(0, sr // 2, n_freqs, dtype=np.float64)

    # Points mel espaces lineairement
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    m_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    f_pts = np.array([_mel_to_hz(m) for m in m_pts])

    # Differences entre points mel consecutifs
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1,)

    # Pentes : distance de chaque freq STFT a chaque point mel
    slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]  # (n_freqs, n_mels+2)

    # Filtres triangulaires (identique a torchaudio)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]                # (n_freqs, n_mels)
    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))  # (n_freqs, n_mels)

    # Normalisation slaney
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    fb *= enorm[np.newaxis, :]

    return fb.T.astype(np.float32)  # (n_mels, n_freqs)


# Cache du filtre mel (calcule une seule fois)
_FILTRE_MEL_CACHE: np.ndarray | None = None


def _get_filtre_mel() -> np.ndarray:
    global _FILTRE_MEL_CACHE
    if _FILTRE_MEL_CACHE is None:
        _FILTRE_MEL_CACHE = _creer_filtres_mel()
    return _FILTRE_MEL_CACHE


def mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Calcule le log mel-spectrogram d'un signal audio.

    Parameters
    ----------
    audio : np.ndarray
        Signal PCM float32 mono, shape (N,).
    sr : int
        Sample rate (doit etre 16000).

    Returns
    -------
    np.ndarray
        Log mel-spectrogram, shape (1, 1, 80, T).
        Pret pour l'inference ONNX.
    """
    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate attendu : {SAMPLE_RATE}, recu : {sr}")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Audio doit etre 1D, shape recu : {audio.shape}")

    # Fenetre de Hann periodique (comme torch.hann_window(periodic=True))
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / WIN_LENGTH)).astype(np.float32)

    # Padding pour centrer les frames (comme torchaudio center=True)
    pad_len = N_FFT // 2
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="reflect")

    # STFT
    n_frames = 1 + (len(audio_padded) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio_padded,
        shape=(n_frames, N_FFT),
        strides=(audio_padded.strides[0] * HOP_LENGTH, audio_padded.strides[0]),
    )

    # Appliquer la fenetre (zero-pad si win_length < n_fft)
    pad_left = (N_FFT - WIN_LENGTH) // 2
    windowed = np.zeros_like(frames)
    windowed[:, pad_left : pad_left + WIN_LENGTH] = frames[:, pad_left : pad_left + WIN_LENGTH] * window

    # FFT
    spectrum = np.fft.rfft(windowed, n=N_FFT)  # (n_frames, n_fft//2+1)

    # Power spectrum
    power = np.abs(spectrum) ** 2  # (n_frames, n_fft//2+1)

    # Filtre mel
    mel_fb = _get_filtre_mel()  # (n_mels, n_fft//2+1)
    mel = mel_fb @ power.T  # (n_mels, n_frames)

    # Log mel
    log_mel = np.log(mel + 1e-8)

    # Shape (1, 1, 80, T) pour ONNX
    return log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
