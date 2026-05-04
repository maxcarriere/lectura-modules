"""Fonctions WORLD pour la synthese diphone.

Operations sur les parametres WORLD (F0, spectral envelope, aperiodicity) :
  - stretch_params : time-warp par interpolation
  - concat_diphones : concatenation avec overlap en domaine log
  - synthesize : appel pw.synthesize()
  - ensure_full_spectrum : pad si bins tronques
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

SIWIS_SR = 44100
FRAME_PERIOD = 5.0
OVERLAP_FRAMES = 4  # ~20ms overlap entre diphones adjacents


def ensure_full_spectrum(sp: np.ndarray, ap: np.ndarray, sr: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Pad spectral data to full FFT size if truncated."""
    import pyworld as pw

    fft_size = pw.get_cheaptrick_fft_size(sr)
    expected_bins = fft_size // 2 + 1

    if sp.shape[1] >= expected_bins:
        return sp, ap

    n_frames = sp.shape[0]

    sp_full = np.zeros((n_frames, expected_bins), dtype=np.float64)
    sp_full[:, :sp.shape[1]] = sp
    edge = sp[:, -1:]
    for j in range(sp.shape[1], expected_bins):
        decay = max(1e-10, 1.0 - (j - sp.shape[1]) / (expected_bins - sp.shape[1]))
        sp_full[:, j] = edge.flatten() * decay
    sp_full = np.maximum(sp_full, 1e-10)

    ap_full = np.ones((n_frames, expected_bins), dtype=np.float64)
    ap_full[:, :ap.shape[1]] = ap

    return sp_full, ap_full


def stretch_params(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray,
                   n_target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-stretch WORLD parameters to target number of frames."""
    n_orig = len(f0)
    if n_orig == n_target:
        return f0.copy(), sp.copy(), ap.copy()
    if n_orig < 2 or n_target < 2:
        return (np.full(n_target, f0[0] if len(f0) > 0 else 0.0),
                np.tile(sp[0:1], (n_target, 1)) if len(sp) > 0 else sp,
                np.tile(ap[0:1], (n_target, 1)) if len(ap) > 0 else ap)

    x_old = np.linspace(0, 1, n_orig)
    x_new = np.linspace(0, 1, n_target)

    # F0 with voicing preservation
    voiced = f0 > 0
    f0_interp = np.interp(x_new, x_old, f0)
    voicing = np.interp(x_new, x_old, voiced.astype(np.float64))
    f0_out = np.where(voicing > 0.5, f0_interp, 0.0)

    # SP in log domain
    log_sp = np.log(np.maximum(sp, 1e-10))
    sp_out = np.exp(interp1d(x_old, log_sp, axis=0, kind="linear",
                              fill_value="extrapolate")(x_new))

    # AP linear
    ap_out = np.clip(interp1d(x_old, ap, axis=0, kind="linear",
                               fill_value="extrapolate")(x_new), 0.0, 1.0)

    return f0_out, sp_out, ap_out


def concat_diphones(
    segments: list[dict],
    overlap: int = OVERLAP_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate diphone WORLD params with smooth overlap blending."""
    if len(segments) == 1:
        return segments[0]["f0"], segments[0]["sp"], segments[0]["ap"]

    # Calculate total frames
    total = sum(len(s["f0"]) for s in segments)
    total -= overlap * (len(segments) - 1)
    total = max(total, 1)

    n_bins = segments[0]["sp"].shape[1]
    f0_out = np.zeros(total, dtype=np.float64)
    sp_out = np.full((total, n_bins), 1e-10, dtype=np.float64)
    ap_out = np.ones((total, n_bins), dtype=np.float64)

    pos = 0
    for i, seg in enumerate(segments):
        n = len(seg["f0"])
        actual_overlap = min(overlap, n // 2)

        if i == 0:
            f0_out[pos:pos + n] = seg["f0"]
            sp_out[pos:pos + n] = seg["sp"]
            ap_out[pos:pos + n] = seg["ap"]
            pos += n - actual_overlap
        else:
            # Blend overlap zone
            blend_len = min(actual_overlap, total - pos, n)
            if blend_len > 0:
                w = np.linspace(0, 1, blend_len)
                for j in range(blend_len):
                    alpha = w[j]
                    # F0
                    f0_a = f0_out[pos + j]
                    f0_b = seg["f0"][j]
                    if f0_a > 0 and f0_b > 0:
                        f0_out[pos + j] = f0_a * (1 - alpha) + f0_b * alpha
                    elif f0_b > 0:
                        f0_out[pos + j] = f0_b

                    # SP: log-domain blend
                    log_a = np.log(np.maximum(sp_out[pos + j], 1e-10))
                    log_b = np.log(np.maximum(seg["sp"][j], 1e-10))
                    sp_out[pos + j] = np.exp(log_a * (1 - alpha) + log_b * alpha)

                    # AP
                    ap_out[pos + j] = ap_out[pos + j] * (1 - alpha) + seg["ap"][j] * alpha

            # Write rest
            rest_start = blend_len
            if rest_start < n:
                write_len = min(n - rest_start, total - (pos + rest_start))
                if write_len > 0:
                    sl = slice(pos + rest_start, pos + rest_start + write_len)
                    f0_out[sl] = seg["f0"][rest_start:rest_start + write_len]
                    sp_out[sl] = seg["sp"][rest_start:rest_start + write_len]
                    ap_out[sl] = seg["ap"][rest_start:rest_start + write_len]

            pos += n - actual_overlap

    return f0_out[:total], sp_out[:total], ap_out[:total]


def synthesize(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray,
               sr: int = SIWIS_SR, frame_period: float = FRAME_PERIOD
               ) -> np.ndarray:
    """Synthesize audio from WORLD parameters via pw.synthesize()."""
    import pyworld as pw

    sp = np.maximum(sp, 1e-10).astype(np.float64)
    ap = np.clip(ap, 0.0, 1.0).astype(np.float64)
    f0 = f0.astype(np.float64)

    audio = pw.synthesize(f0, sp, ap, sr, frame_period)
    return audio.astype(np.float32)
