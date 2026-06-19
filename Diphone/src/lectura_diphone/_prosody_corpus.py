"""Prosodie corpus — clusters SIWIS par groupe prosodique.

Style 'corpus' : contours F0 + dur_ratio + energie extraits du corpus SIWIS
(9750 phrases), regroupes par mode x n_syl_bucket x role du groupe.
Chaque combinaison a 5 clusters K-Means avec co-variation naturelle preservee.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from random import Random

import numpy as np

_BANK: dict | None = None


def load_bank() -> dict:
    """Charge et cache la banque de contours groupe."""
    global _BANK
    if _BANK is not None:
        return _BANK
    path = Path(__file__).parent / "_group_contour_bank.json"
    with open(path, encoding="utf-8") as f:
        _BANK = json.load(f)
    return _BANK


def _syl_bucket(n_syl: int) -> str:
    """Map syllable count to contour bank key suffix.

    2..10 exact, 11-13, 14-17, 18+
    """
    if n_syl <= 1:
        return "2"
    if n_syl <= 10:
        return str(n_syl)
    if n_syl <= 13:
        return "11-13"
    if n_syl <= 17:
        return "14-17"
    return "18+"


def _interp_contour(contour: list[float], n_syl: int) -> list[float]:
    """Interpole un contour a n_syl points si necessaire."""
    if len(contour) == n_syl:
        return list(contour)
    x_src = np.linspace(0, 1, len(contour))
    x_dst = np.linspace(0, 1, n_syl)
    return list(np.interp(x_dst, x_src, contour))


def select_group_contour(
    mode: str, n_syl: int, group_role: str, rng: Random,
) -> dict:
    """Selectionne un cluster {f0, dur_ratio, energy} depuis la banque.

    Args:
        mode: "declaratif", "question", "exclamation", "suspensif"
        n_syl: nombre de syllabes dans le groupe
        group_role: "seul", "initial", "medial", "terminal"
            - initial/medial -> cle "continuation_{bucket}"
            - terminal/seul  -> cle "{mode}_{bucket}"
        rng: generateur aleatoire deterministe

    Returns:
        dict avec f0, dur_ratio, energy interpoles a n_syl points.
    """
    bank = load_bank()
    bucket = _syl_bucket(n_syl)

    if group_role in ("initial", "medial"):
        key = f"continuation_{bucket}"
    else:
        key = f"{mode}_{bucket}"

    clusters = bank.get(key)
    if not clusters:
        clusters = bank.get(f"declaratif_{bucket}")
    if not clusters:
        clusters = bank.get(f"continuation_{bucket}")
    if not clusters:
        return {
            "f0": [0.0] * max(1, n_syl),
            "dur_ratio": [1.0] * max(1, n_syl),
            "energy": [1.0] * max(1, n_syl),
        }

    # Tirage pondere par n (nombre de groupes dans le cluster)
    weights = [c["n"] for c in clusters]
    total = sum(weights)
    r = rng.random() * total
    cumul = 0.0
    selected = clusters[-1]
    for c in clusters:
        cumul += c["n"]
        if r <= cumul:
            selected = c
            break

    dur_ratio = selected.get("dur_ratio", [1.0] * len(selected.get("f0", [])))
    return {
        "f0": _interp_contour(selected["f0"], n_syl),
        "dur_ratio": _interp_contour(dur_ratio, n_syl),
        "energy": _interp_contour(selected["energy"], n_syl),
    }


def generate_corpus_prosody(
    n_syl: int,
    mode: str,
    rng: Random,
    base_f0: float = 200.0,
    group_role: str = "seul",
    expressivity: float = 1.0,
    jitter_st: float = 0.5,
) -> list[dict]:
    """Genere {f0_hz, f0_rel_st, dur_ratio, energy} par syllabe.

    Args:
        n_syl: nombre de syllabes dans le groupe
        mode: "declaratif", "question", "exclamation", "suspensif"
        rng: generateur aleatoire deterministe
        base_f0: F0 de base en Hz
        group_role: "seul", "initial", "medial", "terminal"
        expressivity: scale le contour F0 (0=plat, 1=normal, 2=exagere)
        jitter_st: ecart-type du jitter gaussien sur F0 (demi-tons)

    Returns:
        Liste de dicts, un par syllabe.
    """
    if n_syl <= 0:
        return []

    cluster = select_group_contour(mode, n_syl, group_role, rng)
    f0_contour = cluster["f0"]
    dur_ratio_contour = cluster["dur_ratio"]
    energy_contour = cluster["energy"]

    result = []
    for si in range(n_syl):
        f0_st = f0_contour[si] if si < len(f0_contour) else 0.0
        # Appliquer expressivite
        f0_st_scaled = f0_st * expressivity
        # Jitter gaussien (moins sur la derniere syllabe)
        if si == n_syl - 1:
            jitter = rng.gauss(0, jitter_st * 0.3)
        else:
            jitter = rng.gauss(0, jitter_st)
        f0_rel_st = max(-12.0, min(10.0, f0_st_scaled + jitter))
        f0_hz = base_f0 * (2.0 ** (f0_rel_st / 12.0))
        f0_hz = max(80.0, f0_hz)

        dr = dur_ratio_contour[si] if si < len(dur_ratio_contour) else 1.0
        energy = energy_contour[si] if si < len(energy_contour) else 1.0

        result.append({
            "f0_hz": f0_hz,
            "f0_rel_st": f0_rel_st,
            "dur_ratio": dr,
            "energy": energy,
        })

    return result
