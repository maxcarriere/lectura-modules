"""Enhancement mel-spectrogramme — numpy pur (pas de scipy).

Compense le lissage L1 du FastPitch :
- Contraste spectral (inter-bande par frame)
- Sharpening temporel (unsharp mask avec kernel gaussien numpy)
- Noise gate (frames silencieuses)
- Fade-out (anti-pop fin d'utterance)
"""

from __future__ import annotations

import numpy as np


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Cree un kernel gaussien 1D normalise."""
    ksize = int(sigma * 4) | 1  # taille impaire
    x = np.arange(ksize) - ksize // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _smooth_temporal(mel: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Lissage temporel par convolution gaussienne, bande par bande."""
    kernel = _gaussian_kernel_1d(sigma)
    smoothed = np.zeros_like(mel)
    for i in range(mel.shape[0]):
        smoothed[i] = np.convolve(mel[i], kernel, mode="same")
    return smoothed


def enhance_mel(
    mel: np.ndarray,
    spectral_alpha: float = 0.20,
    temporal_alpha: float = 0.20,
    clip_min: float = -11.5,
    clip_max: float = 2.0,
) -> np.ndarray:
    """Enhancement spectral + temporel du mel-spectrogramme.

    Args:
        mel: [n_mels, T] mel-spectrogram
        spectral_alpha: force du boost de contraste inter-bandes
        temporal_alpha: force du sharpening temporel
        clip_min: borne basse (log(1e-5) ≈ -11.5)
        clip_max: borne haute

    Returns:
        mel enhance, memes dimensions
    """
    # 1. Contraste spectral : eloigner les bandes de la moyenne par frame
    frame_mean = mel.mean(axis=0, keepdims=True)  # [1, T]
    spectral_detail = mel - frame_mean
    mel = frame_mean + spectral_detail * (1.0 + spectral_alpha)

    # 2. Sharpening temporel : unsharp mask
    smoothed = _smooth_temporal(mel, sigma=3.0)
    temporal_detail = mel - smoothed
    mel = mel + temporal_detail * temporal_alpha

    return np.clip(mel, clip_min, clip_max)


def noise_gate(
    mel: np.ndarray,
    threshold: float = -8.0,
    silence_val: float = -11.5,
) -> np.ndarray:
    """Gate les frames dont l'energie moyenne est sous le seuil.

    Applique un fondu progressif vers silence_val.
    """
    frame_mean = mel.mean(axis=0)  # [T]
    gate_strength = np.clip(
        (threshold - frame_mean) / (threshold - silence_val), 0, 1
    )
    return mel * (1 - gate_strength) + silence_val * gate_strength


def fade_out(
    mel: np.ndarray,
    n_frames: int = 5,
    silence_val: float = -11.5,
) -> np.ndarray:
    """Fade-out lineaire sur les derniers frames."""
    n = min(n_frames, mel.shape[1])
    if n > 0:
        fade = np.linspace(1.0, 0.0, n)
        mel[:, -n:] = mel[:, -n:] * fade + silence_val * (1.0 - fade)
    return mel
