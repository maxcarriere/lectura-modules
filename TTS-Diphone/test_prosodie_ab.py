#!/usr/bin/env python3
"""Test A/B prosodie — compare la synthese actuelle vs apply_f0_contour branche.

Genere des WAV dans /tmp/tts_diphone_prosodie/ pour ecoute.
"""

import soundfile as sf
import numpy as np
from pathlib import Path

OUT = Path("/tmp/tts_diphone_prosodie")
OUT.mkdir(exist_ok=True)

# ── Phrases de test ──────────────────────────────────────────────
PHRASES = {
    "declaratif": "Le chat dort sur le canapé.",
    "interrogatif": "Est-ce que tu viens demain?",
    "exclamatif": "C'est vraiment incroyable!",
    "virgule": "Le matin, je prends un café.",
    "longue": "Les enfants jouent dans le jardin, pendant que les parents discutent tranquillement.",
}

# ── Setup engine ──────────────────────────────────────────────────
from lectura_tts_diphone import creer_engine
from lectura_tts_diphone.engine import SynthMode, DiphoneEngine

engine = creer_engine()

# ── Version A : actuelle (scaling simple) ────────────────────────
print("=== Version A : scaling simple (actuel) ===")
for name, text in PHRASES.items():
    groups = engine._g2p_backend.phonemize(text)
    audio = engine.synthesize_groups(groups, mode=SynthMode.FLUIDE,
                                      duration_scale=1.2, pause_scale=1.2)
    path = OUT / f"A_{name}.wav"
    sf.write(str(path), audio, 44100)
    print(f"  {path.name} ({len(audio)/44100:.2f}s)")

# ── Version B : apply_f0_contour branche ─────────────────────────
# Monkey-patch synthesize_phones pour utiliser apply_f0_contour
_orig_synth = DiphoneEngine.synthesize_phones

def _patched_synth(self, phones, mode=SynthMode.SYLLABES, prosody=None,
                   use_corpus_stats=False, group_info=None,
                   duration_scale=1.0, word_boundaries=None):
    """Version avec apply_f0_contour branche a la place du scaling simple."""
    import pyworld as pw
    from lectura_tts_diphone._world import (
        FRAME_PERIOD, OVERLAP_FRAMES, SIWIS_SR,
        concat_diphones, ensure_full_spectrum, stretch_params, synthesize,
    )

    if not self.loaded:
        self.load()
    if isinstance(mode, str):
        mode = SynthMode(mode)

    chain = self.build_diphone_chain(phones)

    # Duration (identique)
    if use_corpus_stats and self.diphone_stats and mode != SynthMode.SYLLABES:
        diphone_durations = self.compute_durations_from_stats(chain, phones, mode)
    else:
        phone_durations = self.compute_phone_durations(phones, mode)
        diphone_durations = self._phone_durs_to_diphone_durs(chain, phone_durations)

    if duration_scale != 1.0:
        diphone_durations = [d * duration_scale for d in diphone_durations]

    # Pre-boundary lengthening (identique)
    if group_info is not None and len(diphone_durations) > 2:
        n_di = len(diphone_durations)
        boundary = group_info.get("boundary", "none")
        if boundary in ("period", "question", "exclamation"):
            lengthen_factor = 1.4
        elif boundary == "comma":
            lengthen_factor = 1.25
        else:
            lengthen_factor = 1.1
        start_idx = max(1, int(n_di * 0.7))
        for di in range(start_idx, n_di):
            progress = (di - start_idx) / max(1, n_di - 1 - start_idx)
            factor = 1.0 + (lengthen_factor - 1.0) * progress
            diphone_durations[di] *= factor

    f0_targets = self.compute_f0_targets(phones, mode, group_info=group_info)

    if prosody is not None:
        if "f0_hz" in prosody and prosody["f0_hz"] > 0:
            avg_f0 = np.mean(f0_targets) if f0_targets else 190.0
            if avg_f0 > 0:
                f0_scale = prosody["f0_hz"] / avg_f0
                f0_targets = [f * f0_scale for f in f0_targets]
        if "duration_scale" in prosody and prosody["duration_scale"] > 0:
            dur_s = prosody["duration_scale"]
            diphone_durations = [d * dur_s for d in diphone_durations]

    # Build segments — AVEC apply_f0_contour
    diphone_segments = []
    for di_idx, di_key in enumerate(chain):
        diphone = self.diphones.get(di_key)
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
        elif di_key.endswith("-#"):
            target_f0 = f0_targets[-1]
        else:
            a_idx = di_idx - 1
            b_idx = di_idx
            f0_a = f0_targets[a_idx] if a_idx < len(f0_targets) else 190.0
            f0_b = f0_targets[b_idx] if b_idx < len(f0_targets) else 190.0
            target_f0 = (f0_a + f0_b) / 2

        n_target = max(4, int(target_ms / FRAME_PERIOD))
        f0_s, sp_s, ap_s = stretch_params(f0, sp, ap, n_target)

        # <<< DIFFERENCE : apply_f0_contour au lieu du scaling simple >>>
        f0_s = self.apply_f0_contour(f0_s, di_key, target_f0, variability=0.6)

        diphone_segments.append({"f0": f0_s, "sp": sp_s, "ap": ap_s, "key": di_key})

    if not diphone_segments:
        return np.array([], dtype=np.float32)

    f0_cat, sp_cat, ap_cat = concat_diphones(diphone_segments)
    audio = synthesize(f0_cat, sp_cat, ap_cat, SIWIS_SR, FRAME_PERIOD)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio * 0.9 / peak).astype(np.float32)
    return audio


print("\n=== Version B : apply_f0_contour branche ===")
DiphoneEngine.synthesize_phones = _patched_synth
for name, text in PHRASES.items():
    groups = engine._g2p_backend.phonemize(text)
    audio = engine.synthesize_groups(groups, mode=SynthMode.FLUIDE,
                                      duration_scale=1.2, pause_scale=1.2)
    path = OUT / f"B_{name}.wav"
    sf.write(str(path), audio, 44100)
    print(f"  {path.name} ({len(audio)/44100:.2f}s)")

# Restaurer
DiphoneEngine.synthesize_phones = _orig_synth

print(f"\nFichiers generes dans {OUT}/")
print("Comparer A_* (actuel) vs B_* (apply_f0_contour)")
