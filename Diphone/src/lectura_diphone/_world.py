"""Fonctions WORLD pour la synthese diphone.

Operations sur les parametres WORLD (F0, spectral envelope, aperiodicity) :
  - stretch_params : time-warp par interpolation
  - concat_diphones : concatenation avec overlap en domaine log
  - synthesize : appel pw.synthesize()
  - ensure_full_spectrum : pad si bins tronques
  - extract_timbre_signature : signature cepstrale moyenne
  - apply_timbre : transfert de timbre par domaine cepstral
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Concatenate diphone WORLD params with smooth overlap blending.

    Returns:
        (f0, sp, ap, boundaries) where boundaries is a list of
        (start_frame, end_frame) tuples, one per segment.
    """
    if len(segments) == 1:
        n = len(segments[0]["f0"])
        return segments[0]["f0"], segments[0]["sp"], segments[0]["ap"], [(0, n)]

    # Calculate total frames
    total = sum(len(s["f0"]) for s in segments)
    total -= overlap * (len(segments) - 1)
    total = max(total, 1)

    n_bins = segments[0]["sp"].shape[1]
    f0_out = np.zeros(total, dtype=np.float64)
    sp_out = np.full((total, n_bins), 1e-10, dtype=np.float64)
    ap_out = np.ones((total, n_bins), dtype=np.float64)
    boundaries: list[tuple[int, int]] = []

    pos = 0
    for i, seg in enumerate(segments):
        n = len(seg["f0"])
        actual_overlap = min(overlap, n // 2)

        if i == 0:
            f0_out[pos:pos + n] = seg["f0"]
            sp_out[pos:pos + n] = seg["sp"]
            ap_out[pos:pos + n] = seg["ap"]
            boundaries.append((pos, pos + n))
            pos += n - actual_overlap
        else:
            seg_start = pos
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

            seg_end = min(pos + n - actual_overlap + actual_overlap, total)
            boundaries.append((seg_start, seg_end))
            pos += n - actual_overlap

    return f0_out[:total], sp_out[:total], ap_out[:total], boundaries


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


def compress_aperiodicity(ap: np.ndarray, gamma: float = 1.5,
                          sr: int = SIWIS_SR) -> np.ndarray:
    """Reduire le plancher AP par compression power-law frequentielle.

    gamma > 1 pousse les petites valeurs d'AP vers 0 (= plus periodique).
    Bandes de frequence :
      0-2 kHz : gamma * 1.3 (nettoyage fort, region F1/F2)
      2-5 kHz : gamma * 1.0 (modere, region F3/F4)
      5+ kHz  : gamma * 0.7 (doux, preserve fricatives/sibilantes)
    """
    if gamma <= 1.0:
        return ap
    n_bins = ap.shape[1]
    freq_per_bin = (sr / 2) / max(1, n_bins - 1)
    bin_2k = int(2000 / freq_per_bin)
    bin_5k = int(5000 / freq_per_bin)
    gammas = np.ones(n_bins, dtype=np.float64)
    gammas[:bin_2k] = gamma * 1.3
    gammas[bin_2k:bin_5k] = gamma * 1.0
    gammas[bin_5k:] = gamma * 0.7
    return np.clip(np.power(np.clip(ap, 0.0, 1.0), gammas[np.newaxis, :]),
                   0.0, 1.0)


def sharpen_formants(sp: np.ndarray, gain: float = 1.3,
                     n_ceps: int = 30) -> np.ndarray:
    """Affuter les pics formantiques par liftering cepstral.

    Amplifie les coefficients cepstraux 1..n_ceps (structure formantique)
    avec une rampe lineaire de gain → 1.0 pour eviter le ringing.
    Le coefficient 0 (energie globale) est preserve.
    """
    if gain <= 1.0:
        return sp
    from scipy.fft import dct, idct
    log_sp = np.log(np.maximum(sp, 1e-10))
    cepstrum = dct(log_sp, type=2, axis=1, norm='ortho')
    n_bins = cepstrum.shape[1]
    lifter = np.ones(n_bins, dtype=np.float64)
    upper = min(n_ceps + 1, n_bins)
    lifter[1:upper] = np.linspace(gain, 1.0, upper - 1)
    cepstrum *= lifter[np.newaxis, :]
    return np.exp(idct(cepstrum, type=2, axis=1, norm='ortho'))


def warp_vtln(sp: np.ndarray, alpha: float = 1.0,
              sr: int = SIWIS_SR) -> np.ndarray:
    """Warping VTLN de l'enveloppe spectrale.

    Warping lineaire par morceaux de l'axe frequentiel :
      - Sous 3/4 Nyquist : f_new = f * alpha
      - Au-dessus : interpolation lineaire vers Nyquist
    alpha > 1 : tract plus court (brillant), alpha < 1 : tract plus long (sombre).
    """
    if abs(alpha - 1.0) < 0.001:
        return sp
    n_bins = sp.shape[1]
    nyquist = sr / 2
    freqs = np.linspace(0, nyquist, n_bins)
    f_pivot = nyquist * 0.75
    warped = np.where(
        freqs <= f_pivot,
        freqs * alpha,
        f_pivot * alpha + (freqs - f_pivot) / (nyquist - f_pivot)
                          * (nyquist - f_pivot * alpha),
    )
    warped = np.clip(warped, 0, nyquist)
    log_sp = np.log(np.maximum(sp, 1e-10))
    idx = np.clip(np.searchsorted(warped, freqs) - 1, 0, n_bins - 2)
    denom = np.maximum(warped[idx + 1] - warped[idx], 1e-10)
    w = np.clip((freqs - warped[idx]) / denom, 0, 1)
    sp_out = np.exp(log_sp[:, idx] * (1 - w[np.newaxis, :])
                    + log_sp[:, idx + 1] * w[np.newaxis, :])
    return np.ascontiguousarray(np.maximum(sp_out, 1e-10))


def extract_timbre_signature(sp: np.ndarray) -> np.ndarray:
    """Extraire la signature cepstrale moyenne d'une enveloppe spectrale.

    Calcule le cepstre moyen sur toutes les frames.
    Retourne un vecteur 1D de dimension n_bins.
    """
    from scipy.fft import dct

    log_sp = np.log(np.maximum(sp, 1e-10))
    cepstrum = dct(log_sp, type=2, axis=1, norm='ortho')
    return np.mean(cepstrum, axis=0)


def apply_timbre(sp: np.ndarray, signature: np.ndarray,
                 blend: float = 0.8, texture: float = 0.5,
                 formant_low: int = 3, formant_high: int = 16,
                 ) -> np.ndarray:
    """Appliquer une signature de timbre sur une enveloppe spectrale.

    Preserve les coefficients cepstraux formant_low..formant_high
    (contenu phonetique). Transfere les coefficients d'identite
    (0..formant_low) et de texture (formant_high..) depuis la signature.

    Args:
        sp: (n_frames, n_bins) enveloppe spectrale lineaire
        signature: (n_bins,) signature cepstrale cible
        blend: force du transfert pour les coefficients d'identite (0..formant_low)
        texture: force du transfert pour les coefficients de texture (formant_high..)
        formant_low: debut de la zone formantique preservee
        formant_high: fin de la zone formantique preservee

    Returns:
        sp modifiee avec le timbre cible applique
    """
    from scipy.fft import dct, idct

    log_sp = np.log(np.maximum(sp, 1e-10))
    ceps = dct(log_sp, type=2, axis=1, norm='ortho')
    n_bins = ceps.shape[1]

    # Signature cepstrale du signal courant (moyenne par frame)
    ceps_mean = np.mean(ceps, axis=0)

    # Coefficients d'identite (tilt spectral, energie)
    fl = min(formant_low, n_bins)
    if fl > 0 and blend > 0:
        # Deplacer chaque frame vers la signature cible
        delta = signature[:fl] - ceps_mean[:fl]
        ceps[:, :fl] += delta[np.newaxis, :] * blend

    # Zone formantique (formant_low..formant_high) : preservee intacte

    # Coefficients de texture (detail spectral fin)
    fh = min(formant_high, n_bins)
    if fh < n_bins and texture > 0:
        delta = signature[fh:] - ceps_mean[fh:]
        ceps[:, fh:] += delta[np.newaxis, :] * texture

    return np.maximum(np.exp(idct(ceps, type=2, axis=1, norm='ortho')), 1e-10)
