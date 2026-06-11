#!/usr/bin/env python3
"""Comparaison v9 (AP rules) vs v10 (corpus clusters) — genere des WAV.

Sortie : /tmp/tts_demo_v10/
  v9_<nom>.wav  — ancien F0 par regles AP (LHiLH*)
  v10_<nom>.wav — nouveau F0 corpus SIWIS (clusters)
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

OUT = Path("/tmp/tts_demo_v10")
OUT.mkdir(exist_ok=True)

# ── 20 phrases variees ──────────────────────────────────────────
PHRASES = {
    # Déclaratif court
    "decl_court":    "Le chat dort.",
    "decl_simple":   "Le petit chat est mort.",
    # Déclaratif moyen
    "decl_moyen":    "Le matin, je prends un café.",
    "decl_neutre":   "Il fait beau aujourd'hui.",
    # Déclaratif long
    "decl_long":     "Les enfants jouent dans le jardin, pendant que les parents discutent tranquillement.",
    "decl_tres_long": "La prochaine fois que vous viendrez, je vous montrerai les photos de notre voyage en Espagne.",
    # Question
    "quest_court":   "Comment allez-vous?",
    "quest_moyen":   "Est-ce que vous avez vu le dernier film?",
    "quest_long":    "Savez-vous combien de temps il faut pour aller de Paris à Lyon en train?",
    # Exclamation
    "exclam_court":  "Quel bonheur!",
    "exclam_moyen":  "Il fait vraiment beau aujourd'hui!",
    "exclam_long":   "Je suis tellement content de vous revoir après toutes ces années!",
    # Suspensif
    "susp_court":    "Je ne sais pas...",
    "susp_moyen":    "Si seulement j'avais su plus tôt...",
    # Dialogue
    "dial_decl":     "Bonjour, comment allez-vous?",
    "dial_reponse":  "Je vais bien, merci beaucoup.",
    # Narratif complexe
    "narr_multi":    "Il ouvrit la porte, regarda dehors, et ne vit personne.",
    "narr_longue":   "Après avoir longuement hésité, Marie décida finalement de partir avec son frère.",
    # Edge cases
    "monosyl":       "Non.",
    "deux_mots":     "Bonne nuit.",
}

# ── Ancien _group_f0_contour (v9 — regles AP) ───────────────────

def _v9_group_f0_contour(phones, mode, info, word_boundaries=None):
    """Ancien F0 par regles AP (LHiLH*) — copie du code v9."""
    n = len(phones)
    gi = info.get("group_idx", 0)
    n_groups = info.get("n_groups", 1)
    boundary = info.get("boundary", "none")
    base_f0 = info.get("base_f0", 175.0)
    k = info.get("macro_expressivity", 1.0)

    is_sentence_final = (gi == n_groups - 1) or boundary in (
        "period", "question", "exclamation", "suspensive",
    )
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

        vowel_positions = []
        for j in range(ap_start, ap_end):
            base_ch = phones[j][0] if phones[j] else ""
            if base_ch in _VOWELS:
                vowel_positions.append(j)
        n_syl = len(vowel_positions)

        if n_syl == 0:
            for j in range(ap_start, ap_end):
                f0s[j] = ap_base
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
                if si == 0:
                    syl_targets_st.append(-1.0 * k)
                elif si == 1:
                    syl_targets_st.append(2.0 * k)
                elif si < n_syl - 1:
                    t = (si - 1) / max(1, n_syl - 3)
                    syl_targets_st.append((-0.5 + 0.3 * t) * k)
                else:
                    syl_targets_st.append(3.5 * k)

        if not is_last_ap:
            if is_question:
                cont_scale = 0.50
            else:
                cont_scale = 0.75
            syl_targets_st = [st * cont_scale for st in syl_targets_st]

        if is_last_ap and is_sentence_final:
            if is_question:
                syl_targets_st[-1] = 5.0 * k
            elif is_exclamation:
                if n_syl >= 2:
                    syl_targets_st[0] = max(syl_targets_st[0], 2.5 * k)
                syl_targets_st[-1] = -3.0 * k
            elif is_suspensive:
                syl_targets_st[-1] = -0.5 * k
            else:
                syl_targets_st[-1] = -3.0 * k

        all_ap_targets.append({
            "ap_start": ap_start, "ap_end": ap_end,
            "ap_base": ap_base, "vowel_positions": vowel_positions,
            "syl_targets_st": syl_targets_st,
        })

    rng = _random.Random(hash((gi, n_groups, n)))

    all_vowel_positions = []
    for ap_data in all_ap_targets:
        all_vowel_positions.extend(ap_data["vowel_positions"])

    syllable_spans = _phones_to_syllables(phones, all_vowel_positions)

    global_syl_idx = 0
    for ap_data in all_ap_targets:
        ap_vps = ap_data["vowel_positions"]
        n_syl_ap = len(ap_vps)
        for si in range(n_syl_ap):
            target_hz = ap_data["ap_base"] * (2.0 ** (
                ap_data["syl_targets_st"][si] / 12.0))

            if si < n_syl_ap - 1:
                jitter_st = rng.uniform(-1.0, 1.0)
                if global_syl_idx > 0 and rng.random() < 0.3:
                    prev_span = syllable_spans[global_syl_idx - 1]
                    target_hz = f0s[prev_span[0]]
                else:
                    target_hz *= 2.0 ** (jitter_st / 12.0)

            if global_syl_idx < len(syllable_spans):
                syl_start, syl_end = syllable_spans[global_syl_idx]
                for j in range(syl_start, syl_end):
                    f0s[j] = target_hz
            global_syl_idx += 1

    return f0s


# ── Setup engine ─────────────────────────────────────────────────
engine = creer_engine()
print(f"Engine charge: {len(engine.diphones)} diphones\n")

# ── Synthese de reference (v10 = code actuel) ────────────────────
print("=" * 60)
print("  Generation v9 (regles AP) vs v10 (corpus SIWIS)")
print("=" * 60)

for name, text in PHRASES.items():
    groups = engine._g2p_backend.phonemize(text)

    # Afficher la structure
    total_phones = sum(len(g["phones"]) for g in groups)
    total_vowels = sum(
        1 for g in groups for p in g["phones"]
        if p and p[0] in _VOWELS
    )
    boundaries = [g.get("boundary", "?") for g in groups]
    print(f"\n[{name}] \"{text}\"")
    print(f"  {len(groups)} groupes, {total_phones} phones, {total_vowels} voyelles")
    print(f"  boundaries: {boundaries}")

    # ── v10 (corpus clusters) ──
    audio_v10 = engine.synthesize_groups(
        groups, mode=SynthMode.FLUIDE,
        duration_scale=1.0, pause_scale=1.0,
        macro_expressivity=1.0, micro_expressivity=1.0,
        seed=42,
        spectral_contrast=1.5,
        ap_cleanup=1.5,
        formant_sharpening=1.3,
    )
    path_v10 = OUT / f"v10_{name}.wav"
    sf.write(str(path_v10), audio_v10, 44100)

    # ── v9 (regles AP) — monkey-patch _group_f0_contour ──
    original_method = DiphoneEngine._group_f0_contour
    DiphoneEngine._group_f0_contour = classmethod(
        lambda cls, *a, **kw: _v9_group_f0_contour(*a, **kw)
    )
    try:
        audio_v9 = engine.synthesize_groups(
            groups, mode=SynthMode.FLUIDE,
            duration_scale=1.0, pause_scale=1.0,
            macro_expressivity=1.0, micro_expressivity=1.0,
            seed=42,
            spectral_contrast=1.5,
            ap_cleanup=1.5,
            formant_sharpening=1.3,
        )
    finally:
        DiphoneEngine._group_f0_contour = original_method

    path_v9 = OUT / f"v9_{name}.wav"
    sf.write(str(path_v9), audio_v9, 44100)

    dur_v9 = len(audio_v9) / 44100
    dur_v10 = len(audio_v10) / 44100
    print(f"  v9:  {path_v9.name} ({dur_v9:.2f}s)")
    print(f"  v10: {path_v10.name} ({dur_v10:.2f}s)")

    # Comparer les F0 moyens par groupe (pour info)
    for gi, g in enumerate(groups):
        phones = g["phones"]
        wb = g.get("word_boundaries", [])
        boundary = g.get("boundary", "none")
        info_v9 = {
            "group_idx": gi, "n_groups": len(groups),
            "boundary": boundary, "base_f0": 175.0,
            "macro_expressivity": 1.0,
        }
        info_v10 = dict(info_v9)
        f0s_v9 = _v9_group_f0_contour(phones, SynthMode.FLUIDE, info_v9, wb)
        f0s_v10 = DiphoneEngine._group_f0_contour(
            phones, SynthMode.FLUIDE, info_v10, wb)
        vowel_idx = [i for i, p in enumerate(phones) if p and p[0] in _VOWELS]
        if vowel_idx:
            f0_v9_vowels = [f0s_v9[i] for i in vowel_idx]
            f0_v10_vowels = [f0s_v10[i] for i in vowel_idx]
            print(f"    g{gi} ({boundary}): "
                  f"v9 F0=[{f0_v9_vowels[0]:.0f}..{f0_v9_vowels[-1]:.0f}] "
                  f"v10 F0=[{f0_v10_vowels[0]:.0f}..{f0_v10_vowels[-1]:.0f}]")

print(f"\n{'=' * 60}")
print(f"  {len(PHRASES) * 2} fichiers generes dans {OUT}/")
print(f"{'=' * 60}")
print(f"\nEcoute rapide :")
print(f"  aplay {OUT}/v9_decl_simple.wav {OUT}/v10_decl_simple.wav")
print(f"  aplay {OUT}/v9_quest_moyen.wav {OUT}/v10_quest_moyen.wav")
