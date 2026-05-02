#!/usr/bin/env python3
"""Demo CLI — synthese vocale TTS monospeaker.

Usage :
    python3 demo_cli.py "Bonjour le monde"
    python3 demo_cli.py --ipa "bɔ̃ʒuʁ" --output bonjour.wav
    python3 demo_cli.py --mode api "Bonjour"
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Synthese vocale TTS monospeaker francais"
    )
    parser.add_argument("text", nargs="?", default=None,
                        help="Texte a synthetiser")
    parser.add_argument("--ipa", type=str, default=None,
                        help="Phonemes IPA a synthetiser (au lieu du texte)")
    parser.add_argument("--output", "-o", type=str, default="output_tts.wav",
                        help="Fichier de sortie (defaut: output_tts.wav)")
    parser.add_argument("--mode", choices=["auto", "local", "api"],
                        default="auto", help="Mode d'inference")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Repertoire des modeles ONNX")
    parser.add_argument("--phrase-type", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="0=decl, 1=inter, 2=excl, 3=susp")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--pitch-range", type=float, default=1.3)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    parser.add_argument("--pause-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.text is None and args.ipa is None:
        parser.error("Specifiez un texte ou --ipa")

    from lectura_tts_monospeaker import creer_engine

    engine = creer_engine(mode=args.mode, models_dir=args.models_dir)

    if args.ipa:
        result = engine.synthesize_phonemes(
            args.ipa,
            phrase_type=args.phrase_type,
            duration_scale=args.duration_scale,
            pitch_shift=args.pitch_shift,
            pitch_range=args.pitch_range,
            energy_scale=args.energy_scale,
            pause_scale=args.pause_scale,
        )
    else:
        result = engine.synthesize(args.text)

    # Sauvegarder
    try:
        import soundfile as sf
        sf.write(args.output, result.samples, result.sample_rate)
    except ImportError:
        # Fallback : ecrire en WAV brut
        import struct
        import wave
        import numpy as np

        samples_int16 = (result.samples * 32767).astype(np.int16)
        with wave.open(args.output, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(result.sample_rate)
            wf.writeframes(samples_int16.tobytes())

    duration_s = len(result.samples) / result.sample_rate
    print(f"Audio genere : {args.output} ({duration_s:.2f}s, {result.sample_rate} Hz)")

    if result.phoneme_timings:
        print(f"Timings : {len(result.phoneme_timings)} phonemes")
        for t in result.phoneme_timings[:5]:
            print(f"  {t.ipa:4s} {t.start_ms:6.0f}–{t.end_ms:6.0f} ms")
        if len(result.phoneme_timings) > 5:
            print(f"  ... (+{len(result.phoneme_timings)-5} phonemes)")


if __name__ == "__main__":
    main()
