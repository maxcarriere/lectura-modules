"""Compression/decompression des diphones WORLD pour deploiement.

Format .dpk.gz :
  - SP : log(sp + 1e-10) en float16
  - AP : (ap * 255) en uint8
  - F0 : float32
  - Metadata : sr, frame_period, n_frames
  - pickle + gzip compresslevel=6

Reduction : ~480 Mo float64 → ~43 Mo compresse, sans perte audible.
"""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def save_compressed(data: dict, path: Path, max_freq: int = 8000,
                    sr: int = 44100) -> None:
    """Compresse et sauvegarde les diphones.

    Args:
        data: dict de diphones {key: {"f0", "sp", "ap", "sr", ...}}
        path: chemin de sortie (.dpk.gz)
        max_freq: frequence max a conserver (Hz)
        sr: sample rate pour le calcul des bins FFT
    """
    if not _HAS_NUMPY:
        raise ImportError("numpy requis pour la compression")

    # Determiner la troncature FFT
    first_key = next(iter(data))
    first_entry = data[first_key]
    n_fft_orig = first_entry["sp"].shape[1]
    freq_per_bin = (sr / 2) / (n_fft_orig - 1)
    n_fft_trunc = int(max_freq / freq_per_bin) + 1

    compressed = {}
    for di_key, entry in data.items():
        sp = entry["sp"][:, :n_fft_trunc] if n_fft_trunc < n_fft_orig else entry["sp"]
        ap = entry["ap"][:, :n_fft_trunc] if n_fft_trunc < n_fft_orig else entry["ap"]

        # SP → log float16
        log_sp = np.log(np.maximum(sp, 1e-10)).astype(np.float16)
        # AP → uint8
        ap_u8 = np.clip(ap * 255.0, 0, 255).astype(np.uint8)
        # F0 → float32
        f0 = entry["f0"].astype(np.float32)

        compressed[di_key] = {
            "f0": f0,
            "log_sp": log_sp,
            "ap_u8": ap_u8,
            "sr": entry.get("sr", sr),
            "frame_period": entry.get("frame_period", 5.0),
            "n_frames": len(f0),
        }

    metadata = {
        "format": "dpk_v1",
        "n_fft_truncated": n_fft_trunc,
        "n_fft_original": n_fft_orig,
        "max_freq_hz": max_freq,
        "n_diphones": len(compressed),
        "sr": sr,
    }

    payload = {"diphones": compressed, "metadata": metadata}

    path = Path(path)
    with gzip.open(path, "wb", compresslevel=6) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("Sauvegarde %d diphones → %s (%.1f Mo)",
             len(compressed), path, path.stat().st_size / 1e6)


def load_compressed(path: Path) -> dict:
    """Charge les diphones compresses et les decompresse en float64.

    Args:
        path: chemin du fichier .dpk.gz

    Returns:
        dict de diphones {key: {"f0", "sp", "ap", "sr", "frame_period", "n_frames"}}
    """
    if not _HAS_NUMPY:
        raise ImportError("numpy requis pour la decompression")

    path = Path(path)
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)

    compressed = payload["diphones"]
    metadata = payload["metadata"]

    diphones = {}
    for di_key, entry in compressed.items():
        # log_sp float16 → exp → float64
        sp = np.exp(entry["log_sp"].astype(np.float64))
        # ap_u8 → float64
        ap = entry["ap_u8"].astype(np.float64) / 255.0
        # f0 float32 → float64
        f0 = entry["f0"].astype(np.float64)

        diphones[di_key] = {
            "f0": f0,
            "sp": sp,
            "ap": ap,
            "sr": entry.get("sr", metadata.get("sr", 44100)),
            "frame_period": entry.get("frame_period", 5.0),
            "n_frames": entry.get("n_frames", len(f0)),
        }

    log.info("Charge %d diphones depuis %s (trunc=%d bins)",
             len(diphones), path.name, metadata.get("n_fft_truncated", 0))

    return diphones
