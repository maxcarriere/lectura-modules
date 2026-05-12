#!/usr/bin/env python3
"""Build la version privee du package lectura-tts-multispeaker.

Chiffre les modeles ONNX et construit un wheel contenant les modeles embarques.
Les utilisateurs du wheel prive n'ont pas besoin de telecharger les modeles separement.

Usage:
    python scripts/build_private.py [--models-dir DIR] [--output-dir DIR] [--int8]

Par defaut, cherche les modeles dans ~/.lectura/models/tts_multispeaker/
Avec --int8, utilise les modeles quantifies (encoder_int8.onnx, etc.).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Repertoire racine du package
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PACKAGE_ROOT / "src" / "lectura_tts_multispeaker"
MODELES_DIR = SRC_DIR / "modeles"
DATA_DIR = SRC_DIR / "data"

# Noms des fichiers ONNX attendus (layout unifie)
ONNX_FILES = [
    "encoder.onnx",
    "decoder.onnx",
    "hifigan.onnx",
]

# config.json va dans modeles/ (utilise par l'engine a cote des ONNX)
# phoneme_vocab.json et speakers.json sont deja dans data/ (permanents)
MODEL_CONFIG = "config.json"


def find_models_dir() -> Path:
    """Localise le repertoire des modeles ONNX."""
    candidates = [
        Path.home() / ".lectura" / "models" / "tts_multispeaker",
        Path("/opt/lectura/modeles/tts_multispeaker"),
    ]
    for d in candidates:
        if d.exists() and (d / ONNX_FILES[0]).exists():
            return d
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build version privee TTS-MultiSpeaker")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Repertoire contenant les ONNX et fichiers config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "dist",
        help="Repertoire de sortie pour le wheel",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Utiliser les modeles INT8 quantifies (encoder_int8.onnx, etc.)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or find_models_dir()

    # Mapping des fichiers sources selon le mode
    if args.int8:
        source_files = {name: name.replace(".onnx", "_int8.onnx") for name in ONNX_FILES}
        print("Mode INT8 : utilisation des modeles quantifies")
    else:
        source_files = {name: name for name in ONNX_FILES}

    # Verifier que tous les fichiers sources existent
    missing = []
    for name in ONNX_FILES:
        src_name = source_files[name]
        if not (models_dir / src_name).exists():
            missing.append(src_name)
    if not (models_dir / MODEL_CONFIG).exists():
        missing.append(MODEL_CONFIG)

    if missing:
        print(f"ERREUR: fichiers manquants dans {models_dir}:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"Source des modeles: {models_dir}")

    # Importer le module crypto du package
    sys.path.insert(0, str(SRC_DIR.parent))
    from lectura_tts_multispeaker._crypto import encrypt_model

    # Creer les repertoires cibles
    MODELES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Chiffrer les 3 ONNX
    for name in ONNX_FILES:
        src_name = source_files[name]
        src = models_dir / src_name
        dst = MODELES_DIR / (name.replace(".onnx", ".enc"))
        print(f"  Chiffrement: {src_name} -> {dst.name} ({src.stat().st_size / 1024 / 1024:.1f} Mo)")
        encrypt_model(src, dst)

    # Copier config.json dans modeles/ (l'engine le charge depuis models_dir)
    config_src = models_dir / MODEL_CONFIG
    config_dst = MODELES_DIR / MODEL_CONFIG
    shutil.copy2(config_src, config_dst)
    print(f"  Copie: {MODEL_CONFIG} -> modeles/{MODEL_CONFIG}")

    # Verifier la taille totale
    total = sum(f.stat().st_size for f in MODELES_DIR.glob("*.enc"))
    total += config_dst.stat().st_size
    print(f"\n  Taille totale embarquee: {total / 1024 / 1024:.1f} Mo")

    # Build le wheel
    print("\nConstruction du wheel...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(args.output_dir)],
        cwd=str(PACKAGE_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("ERREUR lors du build:")
        print(result.stderr)
        sys.exit(1)

    # Afficher le wheel genere
    wheels = list(args.output_dir.glob("lectura_tts_multispeaker-*.whl"))
    if wheels:
        wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        print(f"\n  Wheel genere: {wheel}")
        print(f"  Taille: {wheel.stat().st_size / 1024 / 1024:.1f} Mo")

    # Nettoyer les .enc du src (ne pas les garder dans le repo)
    print("\nNettoyage des fichiers temporaires...")
    for enc in MODELES_DIR.glob("*.enc"):
        enc.unlink()
    # Supprimer config.json copie (deja present dans modeles/ du repo en permanence)
    # On ne le supprime pas s'il existait deja avant le build
    # Les .enc sont les seuls fichiers temporaires

    print("\nBuild prive termine avec succes.")


if __name__ == "__main__":
    main()
