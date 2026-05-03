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


def waveform_silence_gate(
    audio: np.ndarray,
    sample_rate: int = 22050,
    window_ms: float = 10.0,
    threshold_db: float = -40.0,
    fade_samples: int = 64,
) -> np.ndarray:
    """Gate les zones silencieuses de la forme d'onde (post-vocoder).

    Calcule le RMS local par fenetre. La ou le RMS est sous le seuil,
    le signal est attenue vers zero avec un fondu pour eviter les clics.

    Args:
        audio: Signal float32 mono.
        sample_rate: Frequence d'echantillonnage.
        window_ms: Taille de la fenetre d'analyse en ms.
        threshold_db: Seuil en dB sous lequel on considere le silence.
        fade_samples: Nombre d'echantillons pour le fondu entree/sortie.
    """
    if len(audio) == 0:
        return audio

    win_size = max(1, int(sample_rate * window_ms / 1000))
    threshold_lin = 10 ** (threshold_db / 20)

    # Calculer le RMS par fenetre
    n_windows = len(audio) // win_size
    if n_windows == 0:
        return audio

    # Masque de gate : 1.0 = garder, 0.0 = silence
    gate = np.ones(len(audio), dtype=np.float32)

    for i in range(n_windows):
        start = i * win_size
        end = start + win_size
        rms = np.sqrt(np.mean(audio[start:end] ** 2))
        if rms < threshold_lin:
            gate[start:end] = 0.0

    # Traiter le reste
    remainder = len(audio) - n_windows * win_size
    if remainder > 0:
        start = n_windows * win_size
        rms = np.sqrt(np.mean(audio[start:] ** 2))
        if rms < threshold_lin:
            gate[start:] = 0.0

    # Lisser les transitions avec un fondu
    diff = np.diff(gate, prepend=gate[0])
    # Transitions 0→1 (ouverture)
    opens = np.where(diff > 0.5)[0]
    for idx in opens:
        start = max(0, idx - fade_samples)
        length = idx - start
        if length > 0:
            gate[start:idx] = np.linspace(0.0, 1.0, length)

    # Transitions 1→0 (fermeture)
    closes = np.where(diff < -0.5)[0]
    for idx in closes:
        end = min(len(audio), idx + fade_samples)
        length = end - idx
        if length > 0:
            gate[idx:end] = np.linspace(1.0, 0.0, length)

    return audio * gate
