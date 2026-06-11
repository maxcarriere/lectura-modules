#!/usr/bin/env python3
"""Comparaison v9/v10/v11 — genere des WAV.

  v9  — regles AP (LHiLH*)
  v10 — contours corpus 20pts (phase 2)
  v11 — archetypes syllabiques + transitions (phase 3)

Sortie : /tmp/tts_demo_v11/
"""

import sys
import random as _random

import numpy as np
import numpy.core.numeric

# Compat numpy 1.x ← pickle genere sous numpy 2.x
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

OUT = Path("/tmp/tts_demo_v11")
OUT.mkdir(exist_ok=True)

# ── 20 phrases variees ──────────────────────────────────────────
PHRASES = {
    "decl_court":     "Le chat dort.",
    "decl_simple":    "Le petit chat est mort.",
    "decl_moyen":     "Le matin, je prends un café.",
    "decl_neutre":    "Il fait beau aujourd'hui.",
    "decl_long":      "Les enfants jouent dans le jardin, pendant que les parents discutent tranquillement.",
    "decl_tres_long": "La prochaine fois que vous viendrez, je vous montrerai les photos de notre voyage en Espagne.",
    "quest_court":    "Comment allez-vous?",
    "quest_moyen":    "Est-ce que vous avez vu le dernier film?",
    "quest_long":     "Savez-vous combien de temps il faut pour aller de Paris à Lyon en train?",
    "exclam_court":   "Quel bonheur!",
    "exclam_moyen":   "Il fait vraiment beau aujourd'hui!",
    "exclam_long":    "Je suis tellement content de vous revoir après toutes ces années!",
    "susp_court":     "Je ne sais pas...",
    "susp_moyen":     "Si seulement j'avais su plus tôt...",
    "dial_decl":      "Bonjour, comment allez-vous?",
    "dial_reponse":   "Je vais bien, merci beaucoup.",
    "narr_multi":     "Il ouvrit la porte, regarda dehors, et ne vit personne.",
    "narr_longue":    "Après avoir longuement hésité, Marie décida finalement de partir avec son frère.",
    "monosyl":        "Non.",
    "deux_mots":      "Bonne nuit.",
}

# ── v9 : ancien _group_f0_contour (regles AP LHiLH*) ───────────

def _v9_group_f0_contour(phones, mode, info, word_boundaries=None):
    n = len(phones)
    gi = info.get("group_idx", 0)
    n_groups = info.get("n_groups", 1)
    boundary = info.get("boundary", "none")
    base_f0 = info.get("base_f0", 175.0)
    k = info.get("macro_expressivity", 1.0)
    is_sentence_final = (gi == n_groups - 1) or boundary in ("period", "question", "exclamation", "suspensive")
    is_question = boundary == "question"
    is_exclamation = boundary == "exclamation"
    is_suspensive = boundary == "suspensive"
    wb = word_boundaries if word_boundaries else []
    aps = DiphoneEngine._segment_aps(phones, wb)
    if not aps:
        return [base_f0] * n
    n_aps = len(aps)
    f0s = [base_f0] * n
    all_ap_targets = []
    for ap_idx, (ap_start, ap_end) in enumerate(aps):
        is_last_ap = (ap_idx == n_aps - 1)
        decl_st = max(-2.0, -0.5 * ap_idx)
        ap_base = DiphoneEngine._st_to_hz(base_f0, decl_st)
        vowel_positions = [j for j in range(ap_start, ap_end)
                           if phones[j] and phones[j][0] in _VOWELS]
        n_syl = len(vowel_positions)
        if n_syl == 0:
            for j in range(ap_start, ap_end): f0s[j] = ap_base
            continue
        syl_targets_st = []
        if n_syl <= 2:
            for si in range(n_syl):
                t = si / max(1, n_syl - 1)
                syl_targets_st.append((-1.5 + 4.5 * t) * k)
        elif n_syl <= 4:
            for si in range(n_syl):
                if si < n_syl - 1:
                    t = si / max(1, n_syl - 2)
                    syl_targets_st.append((-1.5 + 0.5 * t) * k)
                else:
                    syl_targets_st.append(3.5 * k)
        else:
            for si in range(n_syl):
                if si == 0: syl_targets_st.append(-1.0 * k)
                elif si == 1: syl_targets_st.append(2.0 * k)
                elif si < n_syl - 1:
                    t = (si - 1) / max(1, n_syl - 3)
                    syl_targets_st.append((-0.5 + 0.3 * t) * k)
                else: syl_targets_st.append(3.5 * k)
        if not is_last_ap:
            cont_scale = 0.50 if is_question else 0.75
            syl_targets_st = [st * cont_scale for st in syl_targets_st]
        if is_last_ap and is_sentence_final:
            if is_question: syl_targets_st[-1] = 5.0 * k
            elif is_exclamation:
                if n_syl >= 2: syl_targets_st[0] = max(syl_targets_st[0], 2.5 * k)
                syl_targets_st[-1] = -3.0 * k
            elif is_suspensive: syl_targets_st[-1] = -0.5 * k
            else: syl_targets_st[-1] = -3.0 * k
        all_ap_targets.append({"ap_start": ap_start, "ap_end": ap_end,
            "ap_base": ap_base, "vowel_positions": vowel_positions,
            "syl_targets_st": syl_targets_st})
    rng = _random.Random(hash((gi, n_groups, n)))
    all_vp = []
    for d in all_ap_targets: all_vp.extend(d["vowel_positions"])
    spans = _phones_to_syllables(phones, all_vp)
    gsi = 0
    for d in all_ap_targets:
        for si in range(len(d["vowel_positions"])):
            hz = d["ap_base"] * (2.0 ** (d["syl_targets_st"][si] / 12.0))
            if si < len(d["vowel_positions"]) - 1:
                jitter = rng.uniform(-1.0, 1.0)
                if gsi > 0 and rng.random() < 0.3:
                    hz = f0s[spans[gsi - 1][0]]
                else:
                    hz *= 2.0 ** (jitter / 12.0)
            if gsi < len(spans):
                for j in range(spans[gsi][0], spans[gsi][1]): f0s[j] = hz
            gsi += 1
    return f0s

# ── v10 : contours corpus 20pts (phase 2) ───────────────────────

from lectura_tts_diphone._prosody_corpus import (
    map_f0_contour, select_cluster,
)

def _v10_group_f0_contour(phones, mode, info, word_boundaries=None):
    n = len(phones)
    gi = info.get("group_idx", 0)
    n_groups = info.get("n_groups", 1)
    boundary = info.get("boundary", "none")
    base_f0 = info.get("base_f0", 175.0)
    k = info.get("macro_expressivity", 1.0)
    vowel_positions = [j for j in range(n) if phones[j] and phones[j][0] in _VOWELS]
    n_syl = len(vowel_positions)
    if n_syl == 0:
        return [base_f0] * n
    mode_map = {"period": "declaratif", "question": "question",
                "exclamation": "exclamation", "suspensive": "suspensif",
                "comma": "declaratif", "none": "declaratif"}
    mode_str = mode_map.get(boundary, "declaratif")
    rng = _random.Random(hash((gi, n_groups, n)))
    registre = info.get("registre", "narratif")
    cluster = select_cluster(registre, mode_str, n_syl, rng)
    syl_f0_hz = map_f0_contour(cluster, n_syl, base_f0, k, rng)
    f0s = [base_f0] * n
    spans = _phones_to_syllables(phones, vowel_positions)
    for si, (start, end) in enumerate(spans):
        if si < len(syl_f0_hz):
            for j in range(start, end): f0s[j] = syl_f0_hz[si]
    return f0s


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

def _synth_with_contour(contour_fn):
    """Monkey-patch _group_f0_contour et synthetise."""
    original = DiphoneEngine._group_f0_contour
    DiphoneEngine._group_f0_contour = classmethod(
        lambda cls, *a, **kw: contour_fn(*a, **kw))
    try:
        return engine.synthesize_groups(groups, **COMMON_PARAMS)
    finally:
        DiphoneEngine._group_f0_contour = original

# ── Generation ───────────────────────────────────────────────────
print("=" * 70)
print("  v9 (regles AP) vs v10 (corpus 20pts) vs v11 (archetypes syl)")
print("=" * 70)

for name, text in PHRASES.items():
    groups = engine._g2p_backend.phonemize(text)
    total_vowels = sum(1 for g in groups for p in g["phones"] if p and p[0] in _VOWELS)
    boundaries = [g.get("boundary", "?") for g in groups]
    print(f"\n[{name}] \"{text}\"")
    print(f"  {len(groups)} gr, {total_vowels} syl, boundaries={boundaries}")

    # v11 = code actuel (phase 3 archetypes)
    audio_v11 = engine.synthesize_groups(groups, **COMMON_PARAMS)
    sf.write(str(OUT / f"v11_{name}.wav"), audio_v11, 44100)

    # v10 = phase 2 contours corpus
    audio_v10 = _synth_with_contour(_v10_group_f0_contour)
    sf.write(str(OUT / f"v10_{name}.wav"), audio_v10, 44100)

    # v9 = regles AP
    audio_v9 = _synth_with_contour(_v9_group_f0_contour)
    sf.write(str(OUT / f"v9_{name}.wav"), audio_v9, 44100)

    d9 = len(audio_v9)/44100
    d10 = len(audio_v10)/44100
    d11 = len(audio_v11)/44100
    print(f"  v9={d9:.2f}s  v10={d10:.2f}s  v11={d11:.2f}s")

print(f"\n{'=' * 70}")
print(f"  {len(PHRASES) * 3} fichiers generes dans {OUT}/")
print(f"{'=' * 70}")
print(f"\nEcoute :")
print(f"  aplay {OUT}/v9_decl_simple.wav {OUT}/v10_decl_simple.wav {OUT}/v11_decl_simple.wav")
print(f"  aplay {OUT}/v9_quest_moyen.wav {OUT}/v10_quest_moyen.wav {OUT}/v11_quest_moyen.wav")
