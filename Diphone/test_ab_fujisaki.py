#!/usr/bin/env python3
"""Test A/B : interpolation lineaire (avant) vs Fujisaki (apres).

Compare les contours F0 phone par phone et genere des WAV A/B
dans /tmp/tts_ab_fujisaki/ pour ecoute.
"""

import sys
import numpy as np
import numpy.core.numeric

# Compat numpy 1.x ← pickle genere sous numpy 2.x
if not hasattr(np, '_core'):
    class _core: pass
    nc = _core()
    nc.numeric = numpy.core.numeric
    sys.modules['numpy._core'] = nc
    sys.modules['numpy._core.numeric'] = numpy.core.numeric

from pathlib import Path

from lectura_tts_diphone.engine import DiphoneEngine, SynthMode, _VOWELS

OUT = Path("/tmp/tts_ab_fujisaki")
OUT.mkdir(exist_ok=True)

# ── Ancienne logique (interpolation lineaire) ────────────────────

def _old_group_f0_contour(phones, mode, info, word_boundaries=None):
    """Copie de l'ancien _group_f0_contour AVANT Fujisaki."""
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
                syl_targets_st.append((-1.5 + 3.5 * t) * k)
        elif n_syl <= 4:
            for si in range(n_syl):
                if si < n_syl - 1:
                    t = si / max(1, n_syl - 2)
                    syl_targets_st.append((-1.5 + 0.5 * t) * k)
                else:
                    syl_targets_st.append(2.5 * k)
        else:
            for si in range(n_syl):
                if si == 0:
                    syl_targets_st.append(-1.0 * k)
                elif si == 1:
                    syl_targets_st.append(1.5 * k)
                elif si < n_syl - 1:
                    t = (si - 1) / max(1, n_syl - 3)
                    syl_targets_st.append((-0.5 + 0.3 * t) * k)
                else:
                    syl_targets_st.append(2.5 * k)

        if is_last_ap and is_sentence_final:
            if is_question:
                syl_targets_st[-1] = 4.0 * k
                if n_syl >= 2:
                    syl_targets_st[-2] = min(syl_targets_st[-2], -0.5 * k)
            elif is_exclamation:
                if n_syl >= 2:
                    syl_targets_st[0] = max(syl_targets_st[0], 2.0 * k)
                syl_targets_st[-1] = -3.0 * k
            elif is_suspensive:
                syl_targets_st[-1] = -0.5 * k
            else:
                syl_targets_st[-1] = -3.0 * k
                if n_syl >= 2:
                    syl_targets_st[-2] = min(syl_targets_st[-2], -0.5 * k)

        # ---- ANCIENNE LOGIQUE : interpolation lineaire ----
        for j in range(ap_start, ap_end):
            base_ch = phones[j][0] if phones[j] else ""
            is_vowel = base_ch in _VOWELS
            if is_vowel and j in vowel_positions:
                si = vowel_positions.index(j)
                st = syl_targets_st[si]
                f0s[j] = DiphoneEngine._st_to_hz(ap_base, st)
            else:
                prev_vi = None
                next_vi = None
                for vi_idx, vp in enumerate(vowel_positions):
                    if vp <= j:
                        prev_vi = vi_idx
                    if vp > j and next_vi is None:
                        next_vi = vi_idx
                if prev_vi is not None and next_vi is not None:
                    vp_prev = vowel_positions[prev_vi]
                    vp_next = vowel_positions[next_vi]
                    t = (j - vp_prev) / max(1, vp_next - vp_prev)
                    st = (syl_targets_st[prev_vi] * (1 - t)
                          + syl_targets_st[next_vi] * t)
                elif next_vi is not None:
                    st = syl_targets_st[next_vi]
                elif prev_vi is not None:
                    st = syl_targets_st[prev_vi]
                else:
                    st = 0.0
                f0s[j] = DiphoneEngine._st_to_hz(ap_base, st)

    return f0s


# ── Phrases de test ──────────────────────────────────────────────

TESTS = {
    "declaratif": {
        "phones": ["l", "ə", "p", "ə", "t", "i", "ʃ", "a", "ɛ", "m", "ɔ", "ʁ"],
        "wb": [3, 6, 8],
        "info": {"group_idx": 0, "n_groups": 1, "boundary": "period",
                 "base_f0": 175.0, "macro_expressivity": 1.0},
        "label": "Le petit chat est mort.",
    },
    "question": {
        "phones": ["b", "ɔ̃", "ʒ", "u", "ʁ", "k", "ɔ", "m", "ɑ̃", "a", "l", "e", "v", "u"],
        "wb": [5, 9],
        "info": {"group_idx": 0, "n_groups": 1, "boundary": "question",
                 "base_f0": 175.0, "macro_expressivity": 1.0},
        "label": "Bonjour, comment allez-vous?",
    },
    "exclamatif": {
        "phones": ["i", "l", "f", "ɛ", "b", "o", "o", "ʒ", "u", "ʁ", "d", "ɥ", "i"],
        "wb": [2, 4, 6],
        "info": {"group_idx": 0, "n_groups": 1, "boundary": "exclamation",
                 "base_f0": 175.0, "macro_expressivity": 1.0},
        "label": "Il fait beau aujourd'hui !",
    },
    "suspensif": {
        "phones": ["ʒ", "ə", "n", "ə", "s", "ɛ", "p", "a"],
        "wb": [2, 4],
        "info": {"group_idx": 0, "n_groups": 1, "boundary": "suspensive",
                 "base_f0": 175.0, "macro_expressivity": 1.0},
        "label": "Je ne sais pas...",
    },
    "long_multi_ap": {
        "phones": ["l", "e", "z", "ɑ̃", "f", "ɑ̃", "ʒ", "u", "d", "ɑ̃",
                    "l", "ə", "ʒ", "a", "ʁ", "d", "ɛ̃"],
        "wb": [4, 8, 12],
        "info": {"group_idx": 0, "n_groups": 2, "boundary": "comma",
                 "base_f0": 175.0, "macro_expressivity": 1.0},
        "label": "Les enfants jouent dans le jardin,",
    },
}


# ── Comparaison F0 ──────────────────────────────────────────────

def fmt_f0(val):
    return f"{val:6.1f}"


print("=" * 72)
print("  TEST A/B : Interpolation lineaire  vs  Fujisaki")
print("=" * 72)

for name, cfg in TESTS.items():
    phones = cfg["phones"]
    wb = cfg["wb"]
    info = cfg["info"]

    f0_old = _old_group_f0_contour(phones, SynthMode.FLUIDE, info, wb)
    f0_new = DiphoneEngine._group_f0_contour(phones, SynthMode.FLUIDE, info, wb)

    print(f"\n{'─' * 72}")
    print(f"  {name.upper()} : {cfg['label']}")
    print(f"{'─' * 72}")
    print(f"  {'phone':>6s}  {'ancien':>8s}  {'fujisaki':>8s}  {'delta':>7s}  {'smooth':>6s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}")

    max_jump_old = 0.0
    max_jump_new = 0.0
    for i, ph in enumerate(phones):
        delta = f0_new[i] - f0_old[i]
        # Smoothness: jump from previous phone
        if i > 0:
            jump_old = abs(f0_old[i] - f0_old[i-1])
            jump_new = abs(f0_new[i] - f0_new[i-1])
            max_jump_old = max(max_jump_old, jump_old)
            max_jump_new = max(max_jump_new, jump_new)
            smooth = f"{jump_new:5.1f}"
        else:
            smooth = "  ---"
        base_ch = ph[0] if ph else ""
        marker = "*" if base_ch in _VOWELS else " "
        print(f"  {marker}{ph:>5s}  {f0_old[i]:8.1f}  {f0_new[i]:8.1f}  {delta:+7.1f}  {smooth}")

    mean_old = sum(f0_old) / len(f0_old)
    mean_new = sum(f0_new) / len(f0_new)
    print(f"\n  Moyenne: ancien={mean_old:.1f} Hz, fujisaki={mean_new:.1f} Hz")
    print(f"  Max saut: ancien={max_jump_old:.1f} Hz, fujisaki={max_jump_new:.1f} Hz")


# ── Generation audio A/B ────────────────────────────────────────

try:
    from lectura_tts_diphone import creer_engine
    import soundfile as sf
    from unittest.mock import patch

    engine = creer_engine()
    print(f"\n\n{'=' * 72}")
    print("  GENERATION AUDIO A/B")
    print(f"{'=' * 72}")

    PHRASES_AUDIO = [
        ("declaratif",  "Le petit chat est mort."),
        ("question",    "Bonjour, comment allez-vous ?"),
        ("exclamatif",  "Il fait beau aujourd'hui !"),
        ("suspensif",   "Je ne sais pas..."),
        ("longue",      "Les enfants jouent dans le jardin, pendant que les parents discutent tranquillement."),
    ]

    synth_kwargs = dict(
        mode=SynthMode.FLUIDE,
        duration_scale=1.0, macro_expressivity=1.0,
        micro_expressivity=0.0,  # pas de micro pour isoler le contour
        seed=42,
    )

    for name, text in PHRASES_AUDIO:
        groups = engine._g2p_backend.phonemize(text)

        # A = ancien (interpolation lineaire) — on patche la methode
        with patch.object(DiphoneEngine, '_group_f0_contour',
                          classmethod(lambda cls, *a, **kw: _old_group_f0_contour(*a, **kw))):
            audio_a = engine.synthesize_groups(groups, **synth_kwargs)
        path_a = OUT / f"{name}_A_lineaire.wav"
        sf.write(str(path_a), audio_a, 44100)

        # B = Fujisaki (code actuel)
        audio_b = engine.synthesize_groups(groups, **synth_kwargs)
        path_b = OUT / f"{name}_B_fujisaki.wav"
        sf.write(str(path_b), audio_b, 44100)

        dur_a = len(audio_a) / 44100
        dur_b = len(audio_b) / 44100
        print(f"\n  {name}:")
        print(f"    A (lineaire) : {path_a.name} ({dur_a:.2f}s)")
        print(f"    B (fujisaki) : {path_b.name} ({dur_b:.2f}s)")

    print(f"\n  Fichiers dans {OUT}/")
    print(f"  Ecouter :")
    print(f"    aplay {OUT}/*_A_lineaire.wav   # ancien")
    print(f"    aplay {OUT}/*_B_fujisaki.wav   # nouveau")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n  (Generation audio non disponible: {e})")
    print("  La comparaison numerique ci-dessus reste valide.")
