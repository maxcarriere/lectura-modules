#!/usr/bin/env python3
"""Validation pipeline : synthese avec prosodie corpus (ground truth).

Prend des phrases SIWIS reelles, injecte leur prosodie mesuree
(F0, duree, energie, pauses) dans le moteur diphone, et compare
avec la prosodie generee par le modele phase 3.

Sortie : /tmp/tts_groundtruth/
"""

import sys
import json
import math
import random as _random

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
    DiphoneEngine, SynthMode, _VOWELS, _phones_to_syllables,
)
from lectura_tts_diphone._prosody_syl import (
    generate_syllable_prosody, syllable_prosody_to_f0, SylProsody,
)

OUT = Path("/tmp/tts_groundtruth")
OUT.mkdir(exist_ok=True)

# ── Charger les donnees SIWIS ────────────────────────────────────
PROSODY_PATH = Path("/data/work/projets/lectura/workspace/_En Cours/Voix/tts/prosody_dataset/siwis_female.jsonl")
with open(PROSODY_PATH) as f:
    ALL_ENTRIES = [json.loads(l) for l in f]

# ── Phrases selectionnees (indices dans le dataset) ──────────────
SELECTED = {
    "short_decl":  159,   # "J'insiste sur ce point."  (5 syl)
    "medium_decl": 1,     # "Cette lutte se situe a deux niveaux."  (9 syl)
    "long_decl":   0,     # "Benoit Hamon, monsieur le ministre..."  (16 syl)
    "question":    15,    # "Le match est-il equitable ?"  (7 syl)
    "exclamation": 26,    # "C'est un peu nebuleux !"  (6 syl)
    "medium_2":    3,     # "Peu a peu, ils mobilisent des moyens."  (10 syl)
    "long_2":      4,     # "S'il y a unanimite pour augmenter le volume..." (19 syl)
    "medium_3":    8,     # "Le chomage a atteint un niveau record." (11 syl)
}

# ── Setup ────────────────────────────────────────────────────────
engine = creer_engine()
print(f"Engine charge: {len(engine.diphones)} diphones\n")

COMMON_PARAMS = dict(
    mode=SynthMode.FLUIDE,
    duration_scale=1.0,
    pause_scale=1.0,
    macro_expressivity=1.0,
    micro_expressivity=1.0,
    seed=42,
    spectral_contrast=1.5,
    ap_cleanup=1.5,
    formant_sharpening=1.3,
)


def _compute_corpus_stats(entry):
    """Extrait les stats prosodiques du corpus SIWIS."""
    syls = entry["syllables"]
    n = len(syls)
    # F0 base = mediane des f0_mean
    f0_values = [s["f0_mean"] for s in syls]
    base_f0 = np.median(f0_values)

    # Duree mediane pour normaliser
    dur_values = [s["duration"] for s in syls]
    median_dur = np.median(dur_values)

    # Energie mediane pour normaliser
    energy_values = [s["energy_rms"] for s in syls]
    median_energy = np.median(energy_values)

    # Construire les SylProsody ground truth
    gt_prosody = []
    for s in syls:
        f0_rel_st = 12.0 * math.log2(s["f0_mean"] / base_f0) if base_f0 > 0 else 0.0
        f0_delta_st = 12.0 * math.log2(s["f0_end"] / s["f0_start"]) if s["f0_start"] > 0 else 0.0
        dur_ratio = s["duration"] / median_dur if median_dur > 0 else 1.0
        energy_ratio = s["energy_rms"] / median_energy if median_energy > 0 else 1.0
        gt_prosody.append(SylProsody(
            f0_rel_st=f0_rel_st,
            f0_delta_st=f0_delta_st,
            dur_ratio=dur_ratio,
            energy_ratio=energy_ratio,
            archetype="GT",
        ))

    pauses = [(i, s["pause_after_ms"]) for i, s in enumerate(syls) if s.get("pause_after_ms", 0) > 0]

    return {
        "base_f0": float(base_f0),
        "n_syl": n,
        "median_dur_ms": float(median_dur * 1000),
        "median_energy": float(median_energy),
        "f0_values": f0_values,
        "dur_values_ms": [d * 1000 for d in dur_values],
        "energy_values": energy_values,
        "pauses": pauses,
        "gt_prosody": gt_prosody,
        "total_dur_ms": sum(d * 1000 for d in dur_values) + sum(s.get("pause_after_ms", 0) for s in syls),
    }


def _synth_with_ground_truth(groups, gt_prosody, base_f0):
    """Synthese avec prosodie ground truth injectee."""
    original = DiphoneEngine._group_f0_contour

    def _gt_contour(phones, mode, info, word_boundaries=None):
        n = len(phones)
        vowel_positions = [j for j in range(n) if phones[j] and phones[j][0] in _VOWELS]
        n_syl = len(vowel_positions)
        if n_syl == 0:
            return [base_f0] * n

        # Utiliser la prosodie ground truth
        info["_syl_prosody"] = gt_prosody[:n_syl]
        syl_f0_hz = syllable_prosody_to_f0(gt_prosody[:n_syl], base_f0, 1.0)

        f0s = [base_f0] * n
        syllable_spans = _phones_to_syllables(phones, vowel_positions)
        for si, (syl_start, syl_end) in enumerate(syllable_spans):
            if si < len(syl_f0_hz):
                for j in range(syl_start, syl_end):
                    f0s[j] = syl_f0_hz[si]
        return f0s

    DiphoneEngine._group_f0_contour = classmethod(
        lambda cls, *a, **kw: _gt_contour(*a, **kw))
    try:
        return engine.synthesize_groups(groups, base_f0=base_f0, **COMMON_PARAMS)
    finally:
        DiphoneEngine._group_f0_contour = original


def _get_model_prosody(groups):
    """Recupere la prosodie generee par le modele (sans synthese)."""
    all_syl_prosody = []
    n_groups = len(groups)
    for gi, group in enumerate(groups):
        phones = group["phones"]
        boundary = group.get("boundary", "none")
        vowel_positions = [j for j in range(len(phones))
                           if phones[j] and phones[j][0] in _VOWELS]
        n_syl = len(vowel_positions)
        if n_syl == 0:
            continue

        mode_map = {"period": "declaratif", "question": "question",
                    "exclamation": "exclamation", "suspensive": "declaratif",
                    "comma": "declaratif", "none": "declaratif"}
        mode_str = mode_map.get(boundary, "declaratif")
        rng = _random.Random(hash((gi, n_groups, len(phones))))
        syl_prosody = generate_syllable_prosody(n_syl, mode_str, rng)
        all_syl_prosody.extend(syl_prosody)
    return all_syl_prosody


# ── Generation ───────────────────────────────────────────────────
print("=" * 80)
print("  Validation pipeline : ground truth SIWIS vs modele phase 3")
print("=" * 80)

for name, idx in SELECTED.items():
    entry = ALL_ENTRIES[idx]
    text = entry["text"]
    corpus_stats = _compute_corpus_stats(entry)

    print(f"\n{'─' * 80}")
    print(f"[{name}] \"{text}\"")
    print(f"  Corpus: {corpus_stats['n_syl']} syl, base_f0={corpus_stats['base_f0']:.0f} Hz, "
          f"dur_med={corpus_stats['median_dur_ms']:.0f} ms")

    # Pauses dans le corpus
    if corpus_stats["pauses"]:
        pause_str = ", ".join(f"syl{i}→{p}ms" for i, p in corpus_stats["pauses"])
        print(f"  Pauses corpus: {pause_str}")
    else:
        print(f"  Pauses corpus: aucune")

    # Duree totale corpus (parole + pauses)
    print(f"  Duree corpus totale: {corpus_stats['total_dur_ms']:.0f} ms")

    # Phonemiser avec notre G2P
    groups = engine._g2p_backend.phonemize(text)
    total_vowels = sum(1 for g in groups for p in g["phones"] if p and p[0] in _VOWELS)
    boundaries = [g.get("boundary", "?") for g in groups]
    print(f"  G2P: {len(groups)} gr, {total_vowels} syl (engine), boundaries={boundaries}")

    # Comparer le nombre de syllabes
    if total_vowels != corpus_stats["n_syl"]:
        print(f"  ⚠ Mismatch syllabes: corpus={corpus_stats['n_syl']} vs engine={total_vowels}")

    # Prosodie modele
    model_prosody = _get_model_prosody(groups)
    if model_prosody:
        model_f0_st = [sp.f0_rel_st for sp in model_prosody]
        model_dur = [sp.dur_ratio for sp in model_prosody]
        model_nrg = [sp.energy_ratio for sp in model_prosody]
        model_arch = [sp.archetype for sp in model_prosody]

        # Ground truth
        gt = corpus_stats["gt_prosody"]
        gt_f0_st = [sp.f0_rel_st for sp in gt]
        gt_dur = [sp.dur_ratio for sp in gt]
        gt_nrg = [sp.energy_ratio for sp in gt]

        # Tableau comparatif
        n_show = min(len(model_prosody), len(gt))
        print(f"\n  {'Syl':>3} │ {'F0 corpus':>10} │ {'F0 model':>10} │ {'Dur corpus':>10} │ {'Dur model':>10} │ {'Nrg corpus':>10} │ {'Nrg model':>10} │ {'Archetype':>10}")
        print(f"  {'─'*3}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")
        for si in range(n_show):
            cf0 = f"{gt_f0_st[si]:+.1f}st"
            mf0 = f"{model_f0_st[si]:+.1f}st"
            cdur = f"{gt_dur[si]:.2f}x"
            mdur = f"{model_dur[si]:.2f}x"
            cnrg = f"{gt_nrg[si]:.2f}x"
            mnrg = f"{model_nrg[si]:.2f}x"
            arch = model_arch[si]
            # Marquer les ecarts importants
            dur_flag = " !" if abs(gt_dur[si] - model_dur[si]) > 0.5 else ""
            print(f"  {si:>3} │ {cf0:>10} │ {mf0:>10} │ {cdur:>10} │ {mdur:>10}{dur_flag} │ {cnrg:>10} │ {mnrg:>10} │ {arch:>10}")

        # Statistiques globales
        print(f"\n  Dur ratio — corpus: [{min(gt_dur):.2f}, {max(gt_dur):.2f}] median={np.median(gt_dur):.2f}")
        print(f"  Dur ratio — modele: [{min(model_dur):.2f}, {max(model_dur):.2f}] median={np.median(model_dur):.2f}")
        print(f"  F0 range  — corpus: [{min(gt_f0_st):.1f}, {max(gt_f0_st):+.1f}] st")
        print(f"  F0 range  — modele: [{min(model_f0_st):.1f}, {max(model_f0_st):+.1f}] st")

    # ── Synthese ──

    # v11 = modele phase 3 actuel
    audio_v11 = engine.synthesize_groups(groups, **COMMON_PARAMS)
    sf.write(str(OUT / f"model_{name}.wav"), audio_v11, 44100)

    # Ground truth (prosodie corpus injectee)
    audio_gt = _synth_with_ground_truth(
        groups, corpus_stats["gt_prosody"], corpus_stats["base_f0"])
    sf.write(str(OUT / f"gt_{name}.wav"), audio_gt, 44100)

    d_v11 = len(audio_v11) / 44100
    d_gt = len(audio_gt) / 44100
    d_corpus = corpus_stats["total_dur_ms"] / 1000
    print(f"\n  Durees: corpus={d_corpus:.2f}s  gt_synth={d_gt:.2f}s  model={d_v11:.2f}s")
    print(f"  Ratio model/corpus: {d_v11/d_corpus:.2f}x  gt/corpus: {d_gt/d_corpus:.2f}x")

print(f"\n{'=' * 80}")
print(f"  {len(SELECTED) * 2} fichiers generes dans {OUT}/")
print(f"{'=' * 80}")
print(f"\nEcoute :")
print(f"  aplay {OUT}/gt_short_decl.wav {OUT}/model_short_decl.wav")
print(f"  aplay {OUT}/gt_medium_decl.wav {OUT}/model_medium_decl.wav")
