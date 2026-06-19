#!/usr/bin/env python3
"""Test prosodie AP — genere des WAV pour ecoute.

Genere des WAV dans /tmp/tts_diphone_prosodie/ pour ecoute.
Utilise la nouvelle prosodie basee sur les Phrases Accentuelles (AP).
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

import soundfile as sf
from pathlib import Path

OUT = Path("/tmp/tts_diphone_prosodie")
OUT.mkdir(exist_ok=True)

# ── Phrases de test ──────────────────────────────────────────────
PHRASES = {
    "bonjour":     "Bonjour, comment allez-vous?",
    "declaratif":  "Le petit chat est mort.",
    "exclamatif":  "Il fait beau aujourd'hui!",
    "suspensif":   "Je ne sais pas...",
    "virgule":     "Le matin, je prends un café.",
    "longue":      "Les enfants jouent dans le jardin, pendant que les parents discutent tranquillement.",
}

# ── Setup engine ──────────────────────────────────────────────────
from lectura_tts_diphone import creer_engine
from lectura_tts_diphone.engine import SynthMode

engine = creer_engine()

# ── Generation ────────────────────────────────────────────────────
print("=== Prosodie AP (LHiLH*) ===\n")
for name, text in PHRASES.items():
    groups = engine._g2p_backend.phonemize(text)
    # Afficher les phones pour debug
    for gi, g in enumerate(groups):
        wb = g.get("word_boundaries", [])
        print(f"  [{name}] groupe {gi}: {g['phones']}  wb={wb}  boundary={g.get('boundary','?')}")

    audio = engine.synthesize_groups(
        groups, mode=SynthMode.FLUIDE,
        duration_scale=1.0, pause_scale=1.0,
        macro_expressivity=1.0, micro_expressivity=1.0,
        seed=42,
        spectral_contrast=1.5,
    )
    path = OUT / f"{name}.wav"
    sf.write(str(path), audio, 44100)
    dur_s = len(audio) / 44100
    print(f"  -> {path.name} ({dur_s:.2f}s)\n")

print(f"Fichiers generes dans {OUT}/")
print("Ouvrir avec : aplay /tmp/tts_diphone_prosodie/*.wav")
