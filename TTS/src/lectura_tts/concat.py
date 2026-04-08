"""Découpage et concaténation audio par granularité.

3 modes :
- FLUIDE : audio brut, pas de découpage
- MOT_A_MOT : découpe aux frontières de groupes, 300ms silence inter-groupes
- SYLLABES : découpe aux frontières syllabiques, 200ms inter-syllabes, 400ms inter-groupes
"""

from __future__ import annotations

import math

import numpy as np

from lectura_tts.models import Granularity, GroupeTiming, SyllabeTiming


def _trim_silence(
    samples: np.ndarray,
    threshold: float = 0.01,
    margin: int = 200,
) -> np.ndarray:
    """Coupe le silence aux extrémités d'un signal audio."""
    if len(samples) == 0:
        return samples
    above = np.abs(samples) > threshold
    indices = np.nonzero(above)[0]
    if len(indices) == 0:
        return samples
    start = max(0, indices[0] - margin)
    end = min(len(samples), indices[-1] + 1 + margin)
    return samples[start:end]


def _normalize_rms(samples: np.ndarray, target_rms: float) -> np.ndarray:
    """Normalise le volume d'un signal au RMS cible."""
    if len(samples) == 0:
        return samples
    current = float(np.sqrt(np.mean(samples * samples)))
    if current < 1e-6:
        return samples
    return samples * np.float32(target_rms / current)


def _cosine_window(length: int) -> np.ndarray:
    """Demi-fenêtre Hann (fade-in) de la longueur demandée."""
    if length <= 0:
        return np.array([], dtype=np.float32)
    t = np.arange(length, dtype=np.float32) / length
    return (0.5 * (1.0 - np.cos(math.pi * t))).astype(np.float32)


def process_audio(
    samples: np.ndarray,
    sample_rate: int,
    timings: list[GroupeTiming],
    granularity: Granularity,
) -> tuple[np.ndarray, list[GroupeTiming]]:
    """Découpe et recombine l'audio selon la granularité.

    Returns
    -------
    (audio_final, timings_ajustés)
    """
    if len(samples) == 0 or not timings:
        return samples, timings

    if granularity == Granularity.FLUIDE:
        return samples, timings

    if granularity == Granularity.MOT_A_MOT:
        return _process_mot_a_mot(samples, sample_rate, timings)

    return _process_syllabes(samples, sample_rate, timings)


def _process_mot_a_mot(
    samples: np.ndarray,
    sample_rate: int,
    timings: list[GroupeTiming],
) -> tuple[np.ndarray, list[GroupeTiming]]:
    """Découpe aux frontières de groupes avec 300ms silence inter-groupes."""
    inter_group_ms = 300
    inter_group_samples = int(inter_group_ms * sample_rate / 1000)
    silence = np.zeros(inter_group_samples, dtype=np.float32)

    parts: list[np.ndarray] = []
    new_timings: list[GroupeTiming] = []
    cursor = 0

    for gt_idx, gt in enumerate(timings):
        if gt_idx > 0:
            parts.append(silence)
            cursor += inter_group_samples

        if not gt.syllabe_timings:
            new_timings.append(GroupeTiming(group_index=gt.group_index))
            continue

        # Frontières du groupe dans l'audio original
        group_start_ms = gt.syllabe_timings[0].start_ms
        group_end_ms = gt.syllabe_timings[-1].end_ms

        start_sample = int(group_start_ms * sample_rate / 1000)
        end_sample = int(group_end_ms * sample_rate / 1000)
        start_sample = max(0, min(start_sample, len(samples)))
        end_sample = max(start_sample, min(end_sample, len(samples)))

        chunk = samples[start_sample:end_sample]
        if len(chunk) == 0:
            chunk = np.zeros(int(0.05 * sample_rate), dtype=np.float32)

        # Fades légers
        fade_len = min(int(0.005 * sample_rate), len(chunk) // 4)
        if fade_len > 0:
            chunk = chunk.copy()
            chunk[:fade_len] *= _cosine_window(fade_len)
            chunk[-fade_len:] *= _cosine_window(fade_len)[::-1]

        new_start_ms = cursor * 1000 / sample_rate
        parts.append(chunk)
        cursor += len(chunk)
        new_end_ms = cursor * 1000 / sample_rate

        # Recalculer les timings syllabiques proportionnellement
        orig_dur = group_end_ms - group_start_ms
        new_dur = new_end_ms - new_start_ms

        new_syll_timings: list[SyllabeTiming] = []
        for st in gt.syllabe_timings:
            if orig_dur > 0:
                rel_start = (st.start_ms - group_start_ms) / orig_dur
                rel_end = (st.end_ms - group_start_ms) / orig_dur
            else:
                rel_start = 0.0
                rel_end = 1.0
            new_syll_timings.append(SyllabeTiming(
                phone=st.phone, ortho=st.ortho,
                start_ms=new_start_ms + rel_start * new_dur,
                end_ms=new_start_ms + rel_end * new_dur,
            ))

        new_timings.append(GroupeTiming(
            group_index=gt.group_index, syllabe_timings=new_syll_timings,
        ))

    final = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return final, new_timings


def _process_syllabes(
    samples: np.ndarray,
    sample_rate: int,
    timings: list[GroupeTiming],
) -> tuple[np.ndarray, list[GroupeTiming]]:
    """Découpe aux frontières syllabiques, 200ms inter-syllabes, 400ms inter-groupes."""
    inter_syll_ms = 200
    inter_group_ms = 400
    inter_syll_samples = int(inter_syll_ms * sample_rate / 1000)
    inter_group_samples = int(inter_group_ms * sample_rate / 1000)
    silence_syll = np.zeros(inter_syll_samples, dtype=np.float32)
    silence_group = np.zeros(inter_group_samples, dtype=np.float32)

    parts: list[np.ndarray] = []
    new_timings: list[GroupeTiming] = []
    cursor = 0

    for gt_idx, gt in enumerate(timings):
        if gt_idx > 0:
            parts.append(silence_group)
            cursor += inter_group_samples

        new_syll_timings: list[SyllabeTiming] = []

        for st_idx, st in enumerate(gt.syllabe_timings):
            if st_idx > 0:
                parts.append(silence_syll)
                cursor += inter_syll_samples

            start_sample = int(st.start_ms * sample_rate / 1000)
            end_sample = int(st.end_ms * sample_rate / 1000)
            start_sample = max(0, min(start_sample, len(samples)))
            end_sample = max(start_sample, min(end_sample, len(samples)))

            chunk = samples[start_sample:end_sample]
            if len(chunk) == 0:
                chunk = np.zeros(int(0.05 * sample_rate), dtype=np.float32)

            fade_len = min(int(0.005 * sample_rate), len(chunk) // 4)
            if fade_len > 0:
                chunk = chunk.copy()
                chunk[:fade_len] *= _cosine_window(fade_len)
                chunk[-fade_len:] *= _cosine_window(fade_len)[::-1]

            new_start_ms = cursor * 1000 / sample_rate
            parts.append(chunk)
            cursor += len(chunk)
            new_end_ms = cursor * 1000 / sample_rate

            new_syll_timings.append(SyllabeTiming(
                phone=st.phone, ortho=st.ortho,
                start_ms=new_start_ms, end_ms=new_end_ms,
            ))

        new_timings.append(GroupeTiming(
            group_index=gt.group_index, syllabe_timings=new_syll_timings,
        ))

    final = np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    # Normalisation anti-clipping
    if len(final) > 0:
        peak = np.max(np.abs(final))
        if peak > 0.95:
            final = final * np.float32(0.95 / peak)

    return final, new_timings
