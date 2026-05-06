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

# Fichiers de config a embarquer dans data/
CONFIG_FILES = [
    "config.json",
    "phoneme_vocab.json",
    "speakers.json",
]


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
    for name in CONFIG_FILES:
        if not (models_dir / name).exists():
            missing.append(name)

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

    # Copier les fichiers de config dans data/
    for name in CONFIG_FILES:
        config_src = models_dir / name
        config_dst = DATA_DIR / name
        shutil.copy2(config_src, config_dst)
        print(f"  Copie: {name} -> data/{name}")

    # Verifier la taille totale
    total = sum(f.stat().st_size for f in MODELES_DIR.glob("*.enc"))
    total += sum((DATA_DIR / name).stat().st_size for name in CONFIG_FILES)
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
    print("\nNettoyage des .enc du repertoire source...")
    for enc in MODELES_DIR.glob("*.enc"):
        enc.unlink()
    # Supprimer les config copies (garder seulement dans le wheel)
    for name in CONFIG_FILES:
        config_dst = DATA_DIR / name
        if config_dst.exists():
            config_dst.unlink()

    print("\nBuild prive termine avec succes.")


if __name__ == "__main__":
    main()
