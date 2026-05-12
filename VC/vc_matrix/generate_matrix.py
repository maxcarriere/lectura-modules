"""Génère une matrice de conversions vocales :
   sources (diphone, monospeaker) × références (6 speakers) × sr_overrides

Chaque combinaison produit un WAV dans /tmp/vc_matrix/
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import logging

logging.basicConfig(level=logging.WARNING)

OUT_DIR = "/tmp/vc_matrix"
os.makedirs(OUT_DIR, exist_ok=True)

TEXT = "Bonjour, comment allez-vous aujourd'hui ?"

# --- Sample rates à tester ---
SR_OVERRIDES = [11025, 16000, 22050, 32000, 44100, 48000]

# --- Références (speaker → chemin vers un WAV) ---
REF_BASE = "/mnt/wd_black/Data/Voix/Corpus 1 - SIWIS-LibriVox/raw"
REFERENCES = {
    "ezwa": f"{REF_BASE}/fr_FR/female/ezwa/keraban_le_tetu/wavs/keraban_le_tetu_1_01_f000001.wav",
    "nadine": f"{REF_BASE}/fr_FR/female/nadine_eckert_boulet",  # will find first wav
    "bernard": f"{REF_BASE}/fr_FR/male/bernard",
    "gilles": f"{REF_BASE}/fr_FR/male/gilles_g_le_blanc",
    "zeckou": f"{REF_BASE}/fr_FR/male/zeckou",
    "siwis": f"{REF_BASE}/siwis_extract/SiwisFrenchSpeechSynthesisDatabase/wavs/part1/neut_parl_s01_0001.wav",
}

def find_first_wav(path):
    """Trouve le premier WAV dans un répertoire."""
    if os.path.isfile(path) and path.endswith(".wav"):
        return path
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            if f.endswith(".wav"):
                return os.path.join(root, f)
    return None

# Résoudre les chemins de référence
resolved_refs = {}
for name, path in REFERENCES.items():
    wav = find_first_wav(path)
    if wav:
        info = sf.info(wav)
        resolved_refs[name] = wav
        print(f"  ref {name}: {wav} (sr={info.samplerate})")
    else:
        print(f"  ref {name}: NOT FOUND at {path}")

# --- Générer les sources TTS ---
print("\n=== Generating TTS sources ===")
sources = {}

# Diphone
try:
    from lectura_tts_diphone import synthetiser
    audio = synthetiser(TEXT)
    sr = 44100
    sources["diphone"] = (audio, sr)
    sf.write(f"{OUT_DIR}/source_diphone.wav", audio, sr)
    print(f"  diphone: {len(audio)/sr:.1f}s @ {sr} Hz")
except Exception as e:
    print(f"  diphone: FAILED ({e})")

# Monospeaker
try:
    from lectura_tts_monospeaker import synthetiser as mono_synth
    audio = mono_synth(TEXT)
    sr = 22050  # monospeaker default
    if isinstance(audio, tuple):
        audio, sr = audio[0], audio[1]
    sources["mono"] = (audio, sr)
    sf.write(f"{OUT_DIR}/source_mono.wav", audio, sr)
    print(f"  mono: {len(audio)/sr:.1f}s @ {sr} Hz")
except Exception as e:
    print(f"  mono: FAILED ({e})")

# --- VC Engine ---
print("\n=== Loading VC engine ===")
from lectura_vc import creer_engine
vc_engine = creer_engine(mode="auto")
ov = vc_engine._get_openvoice()
print(f"  OpenVoice loaded (sr={ov.sr})")

# --- Extraire les embeddings à différents sr ---
print("\n=== Extracting embeddings ===")
embeddings = {}  # (speaker, sr_override) → se

for ref_name, ref_path in resolved_refs.items():
    # Charger la référence à son sr natif
    ref_audio_native, ref_sr_native = sf.read(ref_path, dtype="float32")
    if ref_audio_native.ndim > 1:
        ref_audio_native = ref_audio_native.mean(axis=1)

    for sr_override in SR_OVERRIDES:
        # Resampler vers sr_override
        if sr_override != ref_sr_native:
            ref_resampled = librosa.resample(
                ref_audio_native, orig_sr=ref_sr_native, target_sr=sr_override
            )
        else:
            ref_resampled = ref_audio_native.copy()

        # Passer comme ndarray SANS sr → OpenVoice traite comme 22050
        se = ov.extract_se(ref_resampled, sr=None)
        embeddings[(ref_name, sr_override)] = se
        factor = 22050 / sr_override
        print(f"  {ref_name} @ {sr_override} Hz → factor={factor:.2f}")

# --- Générer la matrice de conversions ---
print(f"\n=== Generating VC matrix ===")
total = len(sources) * len(resolved_refs) * len(SR_OVERRIDES)
count = 0

for src_name, (src_audio, src_sr) in sources.items():
    # Extraire l'embedding source (correct, à la bonne sr)
    src_22k = librosa.resample(src_audio, orig_sr=src_sr, target_sr=22050)
    src_se = ov.extract_se(src_22k, sr=22050)

    for ref_name in resolved_refs:
        for sr_override in SR_OVERRIDES:
            count += 1
            tgt_se = embeddings[(ref_name, sr_override)]
            factor = 22050 / sr_override

            # Conversion
            converted, conv_sr = ov.convert(
                src_22k, src_se, tgt_se, sr=22050, tau=0.3
            )

            # Nom du fichier
            fname = f"{src_name}_ref-{ref_name}_sr{sr_override}_f{factor:.2f}.wav"
            fpath = os.path.join(OUT_DIR, fname)
            sf.write(fpath, converted, conv_sr)

            print(f"  [{count}/{total}] {fname} ({len(converted)/conv_sr:.1f}s)")

print(f"\n=== Done: {count} files in {OUT_DIR} ===")
