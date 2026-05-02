#!/usr/bin/env python3
"""Build la version privee du package lectura-tts-monospeaker.

Chiffre les modeles ONNX et construit un wheel contenant les modeles embarques.
Les utilisateurs du wheel prive n'ont pas besoin de telecharger les modeles separement.

Usage:
    python scripts/build_private.py [--models-dir DIR] [--output-dir DIR]

Par defaut, cherche les modeles dans ~/.lectura/models/tts_mono/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Repertoire racine du package
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PACKAGE_ROOT / "src" / "lectura_tts_monospeaker"
MODELES_DIR = SRC_DIR / "modeles"
DATA_DIR = SRC_DIR / "data"

# Noms des fichiers ONNX attendus
ONNX_FILES = [
    "fastpitch_encoder.onnx",
    "fastpitch_decoder.onnx",
    "hifigan.onnx",
]

CONFIG_FILE = "config.json"


def find_models_dir() -> Path:
    """Localise le repertoire des modeles ONNX."""
    candidates = [
        Path.home() / ".lectura" / "models" / "tts_mono",
        Path("/opt/lectura/modeles/tts_mono"),
    ]
    for d in candidates:
        if d.exists() and (d / ONNX_FILES[0]).exists():
            return d
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build version privee TTS-Monospeaker")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Repertoire contenant les ONNX et config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "dist",
        help="Repertoire de sortie pour le wheel",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or find_models_dir()

    # Verifier que tous les fichiers sources existent
    missing = []
    for name in ONNX_FILES:
        if not (models_dir / name).exists():
            missing.append(name)
    if not (models_dir / CONFIG_FILE).exists():
        missing.append(CONFIG_FILE)

    if missing:
        print(f"ERREUR: fichiers manquants dans {models_dir}:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"Source des modeles: {models_dir}")

    # Importer le module crypto du package
    sys.path.insert(0, str(SRC_DIR.parent))
    from lectura_tts_monospeaker._crypto import encrypt_model

    # Creer les repertoires cibles
    MODELES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Chiffrer les 3 ONNX
    for name in ONNX_FILES:
        src = models_dir / name
        dst = MODELES_DIR / (name.replace(".onnx", ".enc"))
        print(f"  Chiffrement: {name} -> {dst.name} ({src.stat().st_size / 1024 / 1024:.1f} Mo)")
        encrypt_model(src, dst)

    # Copier config.json (contient les poids embedding pitch/energy)
    config_src = models_dir / CONFIG_FILE
    config_dst = DATA_DIR / CONFIG_FILE
    shutil.copy2(config_src, config_dst)
    print(f"  Copie: {CONFIG_FILE} -> data/{CONFIG_FILE}")

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
    wheels = list(args.output_dir.glob("lectura_tts_monospeaker-*.whl"))
    if wheels:
        wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        print(f"\n  Wheel genere: {wheel}")
        print(f"  Taille: {wheel.stat().st_size / 1024 / 1024:.1f} Mo")

    # Nettoyer les .enc du src (ne pas les garder dans le repo)
    print("\nNettoyage des .enc du repertoire source...")
    for enc in MODELES_DIR.glob("*.enc"):
        enc.unlink()
    # Supprimer config.json copie (garder seulement dans le wheel)
    if config_dst.exists():
        config_dst.unlink()

    print("\nBuild prive termine avec succes.")


if __name__ == "__main__":
    main()
