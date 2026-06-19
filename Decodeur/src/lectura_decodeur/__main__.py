"""CLI pour lectura-ctc — transcription phonetique audio → IPA.

Usage :
    # Transcrire un fichier WAV
    python -m lectura_decodeur fichier.wav

    # Enregistrer depuis le micro (Entree pour arreter)
    python -m lectura_decodeur --micro

    # Enregistrer 5 secondes depuis le micro
    python -m lectura_decodeur --micro --duree 5

    # Mode continu : boucle d'enregistrements micro
    python -m lectura_decodeur --micro --continu

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import struct
import sys
import wave
from pathlib import Path

import numpy as np


def _lire_wav(chemin: str | Path) -> tuple[np.ndarray, int]:
    """Lit un fichier WAV et retourne (audio float32 mono, sample_rate)."""
    chemin = Path(chemin)
    if not chemin.exists():
        print(f"Erreur : fichier introuvable : {chemin}", file=sys.stderr)
        sys.exit(1)

    with wave.open(str(chemin), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        print(f"Erreur : format non supporte ({sampwidth * 8} bits)", file=sys.stderr)
        sys.exit(1)

    # Mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample vers 16kHz si necessaire
    if sr != 16000:
        duration = len(samples) / sr
        n_target = int(duration * 16000)
        indices = np.linspace(0, len(samples) - 1, n_target)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
        print(f"  (resample {sr} Hz -> 16000 Hz)", file=sys.stderr)
        sr = 16000

    return samples, sr


def _enregistrer_micro(duree: float | None = None) -> tuple[np.ndarray, int]:
    """Enregistre depuis le micro.

    Si duree est None, enregistre jusqu'a ce que l'utilisateur appuie sur Entree.
    """
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "Erreur : sounddevice requis pour le micro.\n"
            "  pip install lectura-ctc[micro]",
            file=sys.stderr,
        )
        sys.exit(1)

    sr = 16000

    if duree is not None:
        # Enregistrement a duree fixe
        n_samples = int(duree * sr)
        print(f"Enregistrement ({duree:.1f}s)...", file=sys.stderr)
        audio = sd.rec(n_samples, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        print("Termine.", file=sys.stderr)
        return audio.flatten(), sr

    # Enregistrement interactif (Entree pour arreter)
    print("Enregistrement... (Entree pour arreter)", file=sys.stderr)
    chunks: list[np.ndarray] = []
    stop = False

    def _callback(indata, frames, time_info, status):
        if status:
            pass  # ignorer les warnings de buffer
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sr, channels=1, dtype="float32",
        callback=_callback, blocksize=1600,  # 100ms
    )
    with stream:
        try:
            input()  # bloque jusqu'a Entree
        except (KeyboardInterrupt, EOFError):
            pass

    if not chunks:
        print("Aucun audio enregistre.", file=sys.stderr)
        sys.exit(1)

    audio = np.concatenate(chunks).flatten()
    duree_s = len(audio) / sr
    print(f"Termine ({duree_s:.1f}s).", file=sys.stderr)
    return audio, sr


def main():
    parser = argparse.ArgumentParser(
        prog="python -m lectura_decodeur",
        description="Transcription phonetique audio → IPA (CTC)",
    )
    parser.add_argument(
        "fichier", nargs="?", default=None,
        help="Fichier WAV a transcrire",
    )
    parser.add_argument(
        "--micro", "-m", action="store_true",
        help="Enregistrer depuis le micro",
    )
    parser.add_argument(
        "--duree", "-d", type=float, default=None,
        help="Duree d'enregistrement en secondes (defaut : Entree pour arreter)",
    )
    parser.add_argument(
        "--continu", "-c", action="store_true",
        help="Mode continu : boucle d'enregistrements micro",
    )
    parser.add_argument(
        "--mode", default="auto", choices=["auto", "onnx", "api"],
        help="Backend d'inference (defaut : auto)",
    )
    args = parser.parse_args()

    if args.fichier is None and not args.micro:
        parser.print_help()
        sys.exit(1)

    # Charger l'engine
    from lectura_decodeur import creer_engine
    engine = creer_engine(mode=args.mode)
    print(f"Engine : {engine}", file=sys.stderr)

    if args.fichier:
        # Mode fichier
        audio, sr = _lire_wav(args.fichier)
        duree_s = len(audio) / sr
        print(f"Audio : {args.fichier} ({duree_s:.1f}s, {sr} Hz)", file=sys.stderr)
        ipa = engine.transcrire(audio, sr=sr)
        print(ipa)

    elif args.micro and args.continu:
        # Mode continu
        print("Mode continu (Ctrl+C pour quitter)", file=sys.stderr)
        try:
            while True:
                audio, sr = _enregistrer_micro(duree=args.duree)
                ipa = engine.transcrire(audio, sr=sr)
                print(ipa)
                print("---", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nArret.", file=sys.stderr)

    elif args.micro:
        # Mode micro unique
        audio, sr = _enregistrer_micro(duree=args.duree)
        ipa = engine.transcrire(audio, sr=sr)
        print(ipa)


if __name__ == "__main__":
    main()
