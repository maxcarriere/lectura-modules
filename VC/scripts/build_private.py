#!/usr/bin/env python3
"""Build la version privee du package lectura-vc.

Chiffre les 10 modeles ONNX et construit un wheel contenant les modeles embarques.
Les utilisateurs du wheel prive n'ont pas besoin de telecharger les modeles separement.

Usage:
    python scripts/build_private.py [--models-dir DIR] [--output-dir DIR]

Par defaut, cherche les modeles dans ~/.lectura/models/vc/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Repertoire racine du package
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PACKAGE_ROOT / "src" / "lectura_vc"
MODELES_DIR = SRC_DIR / "modeles"

# 10 fichiers ONNX attendus
ONNX_FILES = [
    # Shared backends
    "hubert.onnx",
    "rmvpe.onnx",
    # OpenVoice
    "openvoice_se.onnx",
    "openvoice_vc.onnx",
    # RVC synthesizers (6 speakers)
    "synthesizer_ezwa.onnx",
    "synthesizer_nadine.onnx",
    "synthesizer_bernard.onnx",
    "synthesizer_gilles.onnx",
    "synthesizer_zeckou.onnx",
    "synthesizer_siwis.onnx",
]


def find_models_dir() -> Path:
    """Localise le repertoire des modeles ONNX."""
    candidates = [
        Path.home() / ".lectura" / "models" / "vc",
        Path("/opt/lectura/modeles/vc"),
    ]
    for d in candidates:
        if d.exists() and (d / ONNX_FILES[0]).exists():
            return d
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build version privee lectura-vc")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Repertoire contenant les fichiers ONNX",
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

    if missing:
        print(f"ERREUR: fichiers manquants dans {models_dir}:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"Source des modeles: {models_dir}")

    # Importer le module crypto du package
    sys.path.insert(0, str(SRC_DIR.parent))
    from lectura_vc._crypto import encrypt_model

    # Creer le repertoire cible
    MODELES_DIR.mkdir(parents=True, exist_ok=True)

    # Chiffrer les 10 ONNX
    for name in ONNX_FILES:
        src = models_dir / name
        dst = MODELES_DIR / (name + ".enc")
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"  Chiffrement: {name} -> {dst.name} ({size_mb:.1f} Mo)")
        encrypt_model(src, dst)

    # Verifier la taille totale
    total = sum(f.stat().st_size for f in MODELES_DIR.glob("*.enc"))
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
    wheels = list(args.output_dir.glob("lectura_vc-*.whl"))
    if wheels:
        wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        print(f"\n  Wheel genere: {wheel}")
        print(f"  Taille: {wheel.stat().st_size / 1024 / 1024:.1f} Mo")

    # Nettoyer les .enc du src (ne pas les garder dans le repo)
    print("\nNettoyage des fichiers temporaires...")
    for enc in MODELES_DIR.glob("*.enc"):
        enc.unlink()

    print("\nBuild prive termine avec succes.")


if __name__ == "__main__":
    main()
