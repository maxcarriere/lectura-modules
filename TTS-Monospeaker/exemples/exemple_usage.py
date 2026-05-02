#!/usr/bin/env python3
"""Exemples d'utilisation de lectura-tts-monospeaker."""

# --- Exemple 1 : Convenience function ---

from lectura_tts_monospeaker import synthetiser

# Necessite lectura-g2p pour le texte → IPA
# audio = synthetiser("Bonjour le monde")


# --- Exemple 2 : Engine ONNX local ---

from lectura_tts_monospeaker import creer_engine

engine = creer_engine(mode="local")

# Depuis des phonemes IPA
result = engine.synthesize_phonemes(
    "bɔ̃ʒuʁ ləmɔ̃d",
    phrase_type=0,        # declaratif
    pitch_range=1.3,      # compenser le lissage
    duration_scale=1.0,
)

print(f"Audio : {len(result.samples)} samples, {result.sample_rate} Hz")
print(f"Duree : {len(result.samples) / result.sample_rate:.2f}s")
print(f"Timings : {len(result.phoneme_timings)} phonemes")


# --- Exemple 3 : Controles prosodiques ---

# Question
result_q = engine.synthesize_phonemes(
    "kɔmɑ̃ taləvu",
    phrase_type=1,       # interrogatif
    pitch_shift=2.0,     # voix plus haute
)

# Lent et emphatique
result_slow = engine.synthesize_phonemes(
    "atɑ̃sjɔ̃",
    phrase_type=2,        # exclamatif
    duration_scale=1.5,   # plus lent
    energy_scale=1.2,     # plus fort
)


# --- Exemple 4 : Sauvegarder en WAV ---

import soundfile as sf
sf.write("bonjour.wav", result.samples, result.sample_rate)
