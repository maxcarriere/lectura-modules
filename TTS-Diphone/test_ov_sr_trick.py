"""Test : trick SR avec OpenVoice — voix aigu/grave.

On trompe OpenVoice sur la frequence de la reference SIWIS pour
obtenir des timbres differents (formants decales).

Principe :
  - SIWIS ref est en 44100 Hz natif
  - On la resample vers sr_override (ex: 18000)
  - On passe le tableau a extract_se() SANS declarer le sr
  - OpenVoice traite comme 22050 Hz → formants decales
  - Factor = 22050 / sr_override

Genere dans /tmp/tts_sr_trick/ :
  f0{pitch}_*.wav          (source diphone a ce pitch)
  sr{SR}_f0{pitch}_*.wav   (conversion avec sr_override)
"""

import sys
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# ── Fix numpy compat ──
import numpy.core.numeric
if not hasattr(np, '_core'):
    class _core: pass
    nc = _core()
    nc.numeric = numpy.core.numeric
    sys.modules['numpy._core'] = nc
    sys.modules['numpy._core.numeric'] = numpy.core.numeric

from lectura_vc._openvoice import OpenVoiceConverter
from lectura_tts_diphone.engine import DiphoneEngine, SynthMode

# ── Config ──
OPENVOICE_MODELS = Path("/data/work/projets/lectura/workspace/Modules/VC/src/lectura_vc/modeles")
SIWIS_REF = Path("/home/moi/Documents/work/projets/lectura/workspace/Modules/VC/extraits/siwis.wav")
OUTPUT_DIR = Path("/tmp/tts_sr_trick")

SR_OVERRIDES = [11025, 16000, 18000, 20000, 22050, 32000, 44100]
BASE_F0S = [150, 175, 200]

# Phrases avec phonemes connus
PHRASES = {
    "bonjour": {
        "phones": ["b", "ɔ̃", "ʒ", "u", "ʁ"],
        "boundary": "period",
        "word_boundaries": [],
    },
    "le_chat_dort": {
        "phones": ["l", "ə", "ʃ", "a", "d", "ɔ", "ʁ"],
        "boundary": "period",
        "word_boundaries": [2, 4],
    },
    "comment_allez_vous": {
        "phones": ["k", "ɔ", "m", "ɑ̃", "a", "l", "e", "v", "u"],
        "boundary": "question",
        "word_boundaries": [4, 7],
    },
    "il_fait_beau": {
        "phones": ["i", "l", "f", "ɛ", "b", "o"],
        "boundary": "exclamation",
        "word_boundaries": [2, 4],
    },
}


def generate_sources(engine):
    """Genere les sources diphone a differents base_f0."""
    sources = {}  # (phrase_name, f0) → (audio, sr)
    for name, group in PHRASES.items():
        for f0 in BASE_F0S:
            audio = engine.synthesize_groups(
                [group],
                mode=SynthMode.FLUIDE,
                prosody_style="regles",
                base_f0=float(f0),
                macro_expressivity=1.0,
                micro_expressivity=1.0,
                seed=42,
            )
            sources[(name, f0)] = (audio, 44100)
            out_path = OUTPUT_DIR / f"f0{f0}_{name}.wav"
            sf.write(str(out_path), audio, 44100)
    return sources


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SIWIS_REF.exists():
        print(f"ERREUR: reference SIWIS introuvable: {SIWIS_REF}")
        return

    # Charger DiphoneEngine
    print("Chargement DiphoneEngine...")
    engine = DiphoneEngine()
    engine.load()

    # Generer sources
    print(f"\n{'='*60}")
    print(f"  Generation sources ({len(PHRASES)} phrases × {len(BASE_F0S)} pitches)")
    print(f"{'='*60}")
    sources = generate_sources(engine)
    for (name, f0), (audio, sr) in sorted(sources.items()):
        print(f"  f0={f0:3d} {name:25s} → {len(audio)/sr:.2f}s")

    # Charger OpenVoice
    print(f"\n{'='*60}")
    print(f"  Chargement OpenVoice")
    print(f"{'='*60}")
    ov = OpenVoiceConverter(OPENVOICE_MODELS)

    # Charger reference SIWIS
    ref_audio_native, ref_sr_native = sf.read(str(SIWIS_REF), dtype="float32")
    if ref_audio_native.ndim > 1:
        ref_audio_native = ref_audio_native.mean(axis=1)
    print(f"  SIWIS ref: sr_natif={ref_sr_native}, duree={len(ref_audio_native)/ref_sr_native:.1f}s")

    # Extraire embeddings a chaque sr_override
    print(f"\n{'='*60}")
    print(f"  Extraction embeddings SIWIS × {len(SR_OVERRIDES)} SR")
    print(f"{'='*60}")
    embeddings = {}
    for sr_override in SR_OVERRIDES:
        if sr_override != ref_sr_native:
            ref_resampled = librosa.resample(
                ref_audio_native, orig_sr=ref_sr_native, target_sr=sr_override
            )
        else:
            ref_resampled = ref_audio_native.copy()

        se = ov.extract_se(ref_resampled, sr=None)
        embeddings[sr_override] = se
        factor = 22050 / sr_override
        label = "aigu" if factor > 1 else "grave" if factor < 1 else "neutre"
        print(f"  sr={sr_override:5d} → factor={factor:.2f} ({label})")

    # Conversions
    print(f"\n{'='*60}")
    print(f"  Conversions")
    print(f"{'='*60}")

    for (name, f0), (src_audio, src_sr) in sorted(sources.items()):
        src_22k = librosa.resample(src_audio, orig_sr=src_sr, target_sr=22050)
        src_se = ov.extract_se(src_22k, sr=22050)

        for sr_override in SR_OVERRIDES:
            tgt_se = embeddings[sr_override]
            factor = 22050 / sr_override

            converted, conv_sr = ov.convert(
                src_22k, src_se, tgt_se, sr=22050, tau=0.3
            )

            peak = np.max(np.abs(converted))
            if peak > 0:
                converted = converted * 0.9 / peak

            out_name = f"sr{sr_override}_f0{f0}_{name}.wav"
            sf.write(str(OUTPUT_DIR / out_name), converted, conv_sr)

        print(f"  f0={f0:3d} {name:25s} → {len(SR_OVERRIDES)} conversions")

    # Resume
    print(f"\n{'='*60}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print()
    total = len(sources) * len(SR_OVERRIDES) + len(sources)
    print(f"{total} fichiers generes ({len(sources)} sources + {len(sources) * len(SR_OVERRIDES)} conversions)")
    print()
    print("Organisation :")
    print("  f0{pitch}_{phrase}.wav          — source diphone")
    print("  sr{sr}_f0{pitch}_{phrase}.wav   — conversion OpenVoice")
    print()
    print("SR overrides :")
    for sr in SR_OVERRIDES:
        f = 22050 / sr
        label = "aigu" if f > 1 else "grave" if f < 1 else "neutre"
        print(f"  sr={sr:5d}  factor={f:.2f}  ({label})")


if __name__ == "__main__":
    main()
