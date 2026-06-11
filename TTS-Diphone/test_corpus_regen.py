#!/usr/bin/env python3
"""Regeneration de phrases SIWIS avec la voix diphonique.

Utilise les durées absolues et F0 mesures du corpus SIWIS,
injectes directement dans le pipeline diphone. Ca permet de
valider que le pipeline de synthese est correct independamment
du modele prosodique.

Sortie : /tmp/tts_regen/
"""

import sys
import json
import math

import numpy as np
import numpy.core.numeric

# Compat numpy 1.x <- pickle genere sous numpy 2.x
if not hasattr(np, '_core'):
    class _core: pass
    nc = _core()
    nc.numeric = numpy.core.numeric
    sys.modules['numpy._core'] = nc
    sys.modules['numpy._core.numeric'] = numpy.core.numeric

import soundfile as sf
from pathlib import Path

from lectura_tts_diphone import creer_engine
from lectura_tts_diphone.engine import (
    DiphoneEngine, SynthMode, _VOWELS, _phones_to_syllables, _BASE_RATE,
    _PHONE_NORMALIZE,
)
from lectura_tts_diphone._world import (
    FRAME_PERIOD, OVERLAP_FRAMES, SIWIS_SR,
    compress_aperiodicity, concat_diphones, ensure_full_spectrum,
    sharpen_formants, stretch_params, synthesize, warp_vtln,
)

OUT = Path("/tmp/tts_regen")
OUT.mkdir(exist_ok=True)

# ── Charger les donnees SIWIS ────────────────────────────────────
SYL_PATH = Path("/data/work/projets/lectura/workspace/_En Cours/Voix/tts/world_siwis/siwis_entries_syllabified.json")
PROS_PATH = Path("/data/work/projets/lectura/workspace/_En Cours/Voix/tts/prosody_dataset/siwis_female.jsonl")

with open(SYL_PATH) as f:
    SYL_DATA = json.load(f)

with open(PROS_PATH) as f:
    PROS_DATA = [json.loads(l) for l in f]

# ── Setup ────────────────────────────────────────────────────────
engine = creer_engine()
print(f"Engine charge: {len(engine.diphones)} diphones\n")

# ── Phrases selectionnees ────────────────────────────────────────
SELECTED = {
    "short_decl":  159,   # "J'insiste sur ce point."  (5 syl)
    "medium_decl": 1,     # "Cette lutte se situe a deux niveaux."  (9 syl)
    "long_decl":   0,     # "Benoit Hamon, monsieur le ministre..."  (16 syl)
    "question":    15,    # "Le match est-il equitable ?"  (7 syl)
    "exclamation": 26,    # "C'est un peu nebuleux !"  (6 syl)
    "medium_2":    3,     # "Peu a peu, ils mobilisent des moyens."  (10 syl)
    "medium_3":    8,     # "Le chomage a atteint un niveau record." (11 syl)
}

# ── Parametres audio ─────────────────────────────────────────────
SPECTRAL_CONTRAST = 1.5
AP_CLEANUP = 1.5
FORMANT_SHARPENING = 1.3


def _extract_corpus_phones(entry_syl):
    """Extrait les phones avec timing et pauses depuis les donnees syllabifiees."""
    phones_raw = entry_syl["phones"]
    result = []
    for i, p in enumerate(phones_raw):
        dur_ms = (p["end"] - p["start"]) * 1000
        # Detecter les pauses (gap entre phones consecutifs)
        pause_after_ms = 0.0
        if i < len(phones_raw) - 1:
            gap = (phones_raw[i + 1]["start"] - p["end"]) * 1000
            if gap > 20:  # > 20ms = pause significative
                pause_after_ms = gap
        result.append({
            "ph": p["ph"],
            "dur_ms": dur_ms,
            "start": p["start"],
            "end": p["end"],
            "pause_after_ms": pause_after_ms,
        })
    return result


def _extract_corpus_f0(entry_pros):
    """Extrait les F0 par syllabe depuis les donnees prosodiques."""
    return [
        {
            "f0_mean": s["f0_mean"],
            "f0_start": s["f0_start"],
            "f0_end": s["f0_end"],
            "energy_rms": s["energy_rms"],
            "duration_ms": s["duration"] * 1000,
        }
        for s in entry_pros["syllables"]
    ]


def _map_corpus_phones_to_engine(corpus_phones):
    """Mappe les phones IPA du corpus vers le format engine.

    Retourne (phones_engine, durations_ms, pauses_after_ms).
    """
    phones = []
    durations = []
    pauses = []

    for cp in corpus_phones:
        ph = cp["ph"]
        # Normalisation IPA -> engine
        if _PHONE_NORMALIZE:
            ph = _PHONE_NORMALIZE.get(ph, ph)
        phones.append(ph)
        durations.append(cp["dur_ms"])
        pauses.append(cp["pause_after_ms"])

    return phones, durations, pauses


def _synthesize_with_corpus_prosody(
    phones, phone_durations_ms, pauses_after_ms, syl_f0,
    syl_durations_ms=None,
    spectral_contrast=1.3, ap_cleanup=1.5, formant_sharpening=1.3,
):
    """Synthese directe avec durées absolues et F0 du corpus.

    phones: liste de phones IPA
    phone_durations_ms: duree de chaque phone en ms
    pauses_after_ms: pause apres chaque phone en ms (0 = pas de pause)
    syl_f0: liste de dicts {f0_mean, f0_start, f0_end, energy_rms} par syllabe
    syl_durations_ms: duree cible par syllabe en ms (si fourni, corrige les
        durees de phones pour que chaque syllabe atteigne sa cible)
    """
    import pyworld as pw

    n = len(phones)
    chain = engine.build_diphone_chain(phones)

    # ── Correction syllabique des durées ──
    # Le mapping phone→diphone avec (dur_a+dur_b)/2 ecrase les phones courts.
    # On corrige en scalant les phones de chaque syllabe pour que le total
    # syllabique corresponde au corpus.
    vowel_positions = [j for j in range(n) if phones[j] and phones[j][0] in _VOWELS]
    n_syl = len(vowel_positions)

    corrected_phone_durs = list(phone_durations_ms)

    if syl_durations_ms and n_syl > 0:
        syllable_spans = _phones_to_syllables(phones, vowel_positions)
        for si, (syl_start, syl_end) in enumerate(syllable_spans):
            if si >= len(syl_durations_ms):
                break
            # Duree actuelle de la syllabe (somme des phones)
            current_syl_dur = sum(corrected_phone_durs[j] for j in range(syl_start, syl_end))
            if current_syl_dur <= 0:
                continue
            target_syl_dur = syl_durations_ms[si]
            scale = target_syl_dur / current_syl_dur
            for j in range(syl_start, syl_end):
                corrected_phone_durs[j] *= scale

    # ── Mapper les durées phones → durées diphones ──
    # Compensation de l'overlap : chaque jonction perd OVERLAP_FRAMES frames
    overlap_compensation = OVERLAP_FRAMES * FRAME_PERIOD  # 20 ms par jonction
    diphone_durations = []
    for di_idx, di_key in enumerate(chain):
        if di_key.startswith("#-"):
            diphone_durations.append(corrected_phone_durs[0] * 0.5 + overlap_compensation * 0.5)
        elif di_key.endswith("-#"):
            diphone_durations.append(corrected_phone_durs[-1] * 0.5 + overlap_compensation * 0.5)
        else:
            a_idx = di_idx - 1
            b_idx = di_idx
            dur_a = corrected_phone_durs[a_idx] if a_idx < n else 50.0
            dur_b = corrected_phone_durs[b_idx] if b_idx < n else 50.0
            diphone_durations.append((dur_a + dur_b) * 0.5 + overlap_compensation)

    # ── F0 par phone (depuis F0 syllabique) ──
    vowel_positions = [j for j in range(n) if phones[j] and phones[j][0] in _VOWELS]
    n_syl = len(vowel_positions)

    # Assigner F0 par syllabe
    f0_targets = [175.0] * n  # fallback

    if n_syl > 0 and syl_f0:
        syllable_spans = _phones_to_syllables(phones, vowel_positions)
        for si, (syl_start, syl_end) in enumerate(syllable_spans):
            if si < len(syl_f0):
                f0_hz = syl_f0[si]["f0_mean"]
                for j in range(syl_start, syl_end):
                    f0_targets[j] = f0_hz

    # Energie par syllabe
    if n_syl > 0 and syl_f0:
        # Calculer energie mediane pour normaliser
        energies = [s["energy_rms"] for s in syl_f0 if s["energy_rms"] > 0]
        median_energy = np.median(energies) if energies else 0.15
        energy_factors = [1.0] * n
        for si, (syl_start, syl_end) in enumerate(syllable_spans):
            if si < len(syl_f0):
                e_ratio = syl_f0[si]["energy_rms"] / median_energy if median_energy > 0 else 1.0
                e_ratio = max(0.3, min(2.0, e_ratio))
                for j in range(syl_start, syl_end):
                    energy_factors[j] = e_ratio
    else:
        energy_factors = [1.0] * n

    # ── Build diphone segments ──
    diphone_segments = []
    for di_idx, di_key in enumerate(chain):
        diphone = engine.diphones.get(di_key)

        if diphone is None:
            fft_size = pw.get_cheaptrick_fft_size(SIWIS_SR)
            n_bins = fft_size // 2 + 1
            n_silence = 4
            diphone_segments.append({
                "f0": np.zeros(n_silence, dtype=np.float64),
                "sp": np.full((n_silence, n_bins), 1e-10, dtype=np.float64),
                "ap": np.ones((n_silence, n_bins), dtype=np.float64),
                "key": di_key,
            })
            continue

        f0 = diphone["f0"].astype(np.float64)
        sp = diphone["sp"].astype(np.float64)
        ap = diphone["ap"].astype(np.float64)
        sr = diphone.get("sr", SIWIS_SR)
        sp, ap = ensure_full_spectrum(sp, ap, sr)

        target_ms = diphone_durations[di_idx]

        if di_key.startswith("#-"):
            target_f0 = f0_targets[0]
            e_factor = energy_factors[0]
        elif di_key.endswith("-#"):
            target_f0 = f0_targets[-1]
            e_factor = energy_factors[-1]
        else:
            a_idx = di_idx - 1
            b_idx = di_idx
            f0_a = f0_targets[a_idx] if a_idx < len(f0_targets) else 190.0
            f0_b = f0_targets[b_idx] if b_idx < len(f0_targets) else 190.0
            target_f0 = (f0_a + f0_b) / 2
            e_a = energy_factors[a_idx] if a_idx < len(energy_factors) else 1.0
            e_b = energy_factors[b_idx] if b_idx < len(energy_factors) else 1.0
            e_factor = (e_a + e_b) / 2

        n_target = max(4, int(target_ms / FRAME_PERIOD))
        f0_s, sp_s, ap_s = stretch_params(f0, sp, ap, n_target)

        # F0 scaling
        voiced = f0_s > 0
        if np.any(voiced) and target_f0 > 0:
            current_mean = np.mean(f0_s[voiced])
            if current_mean > 0:
                f0_s[voiced] *= target_f0 / current_mean

        # Energy scaling
        if abs(e_factor - 1.0) > 0.001:
            sp_s = sp_s * e_factor

        diphone_segments.append({
            "f0": f0_s,
            "sp": sp_s,
            "ap": ap_s,
            "key": di_key,
        })

    if not diphone_segments:
        return np.array([], dtype=np.float32)

    f0_cat, sp_cat, ap_cat, boundaries = concat_diphones(diphone_segments)

    # Pipeline audio
    if ap_cleanup > 1.0:
        ap_cat = compress_aperiodicity(ap_cat, gamma=ap_cleanup, sr=SIWIS_SR)
    if formant_sharpening > 1.0:
        sp_cat = sharpen_formants(sp_cat, gain=formant_sharpening)
    if spectral_contrast > 1.0:
        log_sp = np.log(np.maximum(sp_cat, 1e-10))
        mean_log = np.mean(log_sp, axis=0, keepdims=True)
        sp_cat = np.exp(mean_log + (log_sp - mean_log) * spectral_contrast)

    audio = synthesize(f0_cat, sp_cat, ap_cat, sr=SIWIS_SR)
    if len(audio) > 0:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.9 / peak)

    return audio.astype(np.float32)


def _synthesize_with_pauses(phones, phone_durations_ms, pauses_after_ms, syl_f0):
    """Synthese par segments avec insertion de pauses reelles."""
    pause_positions = []
    for i, pause in enumerate(pauses_after_ms):
        if pause > 0:
            pause_positions.append((i, pause))

    # Extraire durees syllabiques du corpus
    syl_durs = [s["duration_ms"] for s in syl_f0] if syl_f0 else None

    if not pause_positions:
        # Pas de pauses : synthese directe
        audio = _synthesize_with_corpus_prosody(
            phones, phone_durations_ms, pauses_after_ms, syl_f0,
            syl_durations_ms=syl_durs,
            spectral_contrast=SPECTRAL_CONTRAST,
            ap_cleanup=AP_CLEANUP,
            formant_sharpening=FORMANT_SHARPENING,
        )
        return audio

    # Decouper aux positions de pause
    seg_boundaries = []
    prev_end = 0
    for phone_idx, pause_ms in pause_positions:
        seg_boundaries.append((prev_end, phone_idx + 1, pause_ms))
        prev_end = phone_idx + 1
    # Dernier segment
    if prev_end < len(phones):
        seg_boundaries.append((prev_end, len(phones), 0))

    # Compter les syllabes par segment pour distribuer syl_f0 et syl_durs
    audio_parts = []
    syl_offset = 0

    for seg_start, seg_end, pause_ms in seg_boundaries:
        seg_phones = phones[seg_start:seg_end]
        seg_durs = phone_durations_ms[seg_start:seg_end]
        seg_pauses = pauses_after_ms[seg_start:seg_end]

        # Compter les voyelles dans ce segment pour syl_f0
        n_vowels_seg = sum(1 for p in seg_phones if p and p[0] in _VOWELS)
        seg_f0 = syl_f0[syl_offset:syl_offset + n_vowels_seg] if syl_f0 else []
        seg_syl_durs = syl_durs[syl_offset:syl_offset + n_vowels_seg] if syl_durs else None
        syl_offset += n_vowels_seg

        if not seg_phones:
            continue

        audio = _synthesize_with_corpus_prosody(
            seg_phones, seg_durs, seg_pauses, seg_f0,
            syl_durations_ms=seg_syl_durs,
            spectral_contrast=SPECTRAL_CONTRAST,
            ap_cleanup=AP_CLEANUP,
            formant_sharpening=FORMANT_SHARPENING,
        )
        audio_parts.append(audio)

        # Inserer la pause
        if pause_ms > 0:
            n_silence = int(pause_ms / 1000.0 * SIWIS_SR)
            audio_parts.append(np.zeros(n_silence, dtype=np.float32))

    if not audio_parts:
        return np.array([], dtype=np.float32)

    combined = np.concatenate(audio_parts)
    combined = np.clip(combined, -0.95, 0.95)
    return combined


# ── Generation ───────────────────────────────────────────────────
print("=" * 80)
print("  Regeneration SIWIS avec voix diphonique (durées absolues + F0 corpus)")
print("=" * 80)

for name, idx in SELECTED.items():
    entry_syl = SYL_DATA[idx]
    entry_pros = PROS_DATA[idx]
    text = entry_syl["text"]

    print(f"\n{'─' * 80}")
    print(f"[{name}] \"{text}\"")

    # Extraire donnees corpus
    corpus_phones = _extract_corpus_phones(entry_syl)
    syl_f0 = _extract_corpus_f0(entry_pros)

    # Mapper vers format engine
    phones, durations_ms, pauses_ms = _map_corpus_phones_to_engine(corpus_phones)

    # Stats
    total_speech_ms = sum(durations_ms)
    total_pause_ms = sum(pauses_ms)
    total_ms = total_speech_ms + total_pause_ms
    n_vowels = sum(1 for p in phones if p and p[0] in _VOWELS)
    pauses_info = [(i, p) for i, p in enumerate(pauses_ms) if p > 0]

    print(f"  Corpus: {len(phones)} phones, {n_vowels} voyelles, {len(syl_f0)} syl prosody")
    print(f"  Duree: parole={total_speech_ms:.0f}ms + pauses={total_pause_ms:.0f}ms = {total_ms:.0f}ms")
    if pauses_info:
        print(f"  Pauses: {', '.join(f'ph{i}({phones[i]})→{p:.0f}ms' for i, p in pauses_info)}")
    print(f"  Phones: {' '.join(phones)}")
    print(f"  Durees: {' '.join(f'{d:.0f}' for d in durations_ms)}")

    # F0 et duree par syllabe
    f0_str = " ".join(f"{s['f0_mean']:.0f}" for s in syl_f0)
    dur_str = " ".join(f"{s['duration_ms']:.0f}" for s in syl_f0)
    print(f"  F0/syl:  {f0_str} Hz")
    print(f"  Dur/syl: {dur_str} ms")

    # Verifier les diphones disponibles
    chain = engine.build_diphone_chain(phones)
    missing = [di for di in chain if di not in engine.diphones]
    if missing:
        print(f"  ⚠ Diphones manquants ({len(missing)}): {missing[:10]}")

    # ── Synthese avec prosodie corpus (avec pauses) ──
    audio_regen = _synthesize_with_pauses(phones, durations_ms, pauses_ms, syl_f0)
    sf.write(str(OUT / f"regen_{name}.wav"), audio_regen, 44100)

    # ── Synthese avec prosodie corpus SANS pauses ──
    syl_durs = [s["duration_ms"] for s in syl_f0] if syl_f0 else None
    audio_nopauses = _synthesize_with_corpus_prosody(
        phones, durations_ms, pauses_ms, syl_f0,
        syl_durations_ms=syl_durs,
        spectral_contrast=SPECTRAL_CONTRAST,
        ap_cleanup=AP_CLEANUP,
        formant_sharpening=FORMANT_SHARPENING,
    )
    sf.write(str(OUT / f"regen_nopauses_{name}.wav"), audio_nopauses, 44100)

    # ── Synthese modele (v11 normal) ──
    groups = engine._g2p_backend.phonemize(text)
    audio_model = engine.synthesize_groups(
        groups,
        mode=SynthMode.FLUIDE,
        duration_scale=1.0,
        pause_scale=1.0,
        macro_expressivity=1.0,
        micro_expressivity=1.0,
        seed=42,
        spectral_contrast=SPECTRAL_CONTRAST,
        ap_cleanup=AP_CLEANUP,
        formant_sharpening=FORMANT_SHARPENING,
    )
    sf.write(str(OUT / f"model_{name}.wav"), audio_model, 44100)

    d_regen = len(audio_regen) / 44100
    d_nopauses = len(audio_nopauses) / 44100
    d_model = len(audio_model) / 44100
    d_corpus = total_ms / 1000

    print(f"\n  Durees: corpus={d_corpus:.2f}s  regen={d_regen:.2f}s  "
          f"regen_nopauses={d_nopauses:.2f}s  model={d_model:.2f}s")
    print(f"  Ratio regen/corpus: {d_regen/d_corpus:.2f}x  "
          f"model/corpus: {d_model/d_corpus:.2f}x")

print(f"\n{'=' * 80}")
print(f"  {len(SELECTED) * 3} fichiers generes dans {OUT}/")
print(f"{'=' * 80}")
print(f"\nEcoute (comparer corpus-regen vs modele) :")
for name in SELECTED:
    print(f"  aplay {OUT}/regen_{name}.wav {OUT}/model_{name}.wav")
