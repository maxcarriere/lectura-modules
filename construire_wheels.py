#!/usr/bin/env python3
"""
Construit les wheels binaires (Cython) de tous les modules Lectura.

Pour chaque module :
  1. cd Module/
  2. python -m build --wheel
  3. Verifie que le .whl ne contient pas de .py (sauf __init__)
  4. Copie le .whl dans dist/

Usage :
    python construire_wheels.py                 Construit tous les modules
    python construire_wheels.py Tokeniseur G2P  Construit uniquement ceux-ci
    python construire_wheels.py --check-only    Verifie les wheels existants
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

MODULES = ["Tokeniseur", "G2P", "P2G", "Aligneur", "Formules", "Lexique"]

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"


def verifier_wheel(whl_path: Path) -> list[str]:
    """Verifie qu'un .whl ne contient pas de .py ni de .c (sauf __init__.py).

    Renvoie la liste des fichiers problematiques trouves.
    """
    problemes = []
    with zipfile.ZipFile(whl_path) as zf:
        for name in zf.namelist():
            if name.endswith(".py") and not name.endswith("__init__.py"):
                problemes.append(name)
            elif name.endswith(".c") and "/modeles/" not in name:
                problemes.append(name)
    return problemes


def build_module(module: str) -> Path | None:
    """Construit le wheel d'un module. Renvoie le chemin du .whl ou None."""
    module_dir = ROOT / module
    if not module_dir.is_dir():
        print(f"  ERREUR : {module}/ introuvable")
        return None

    # Nettoyer les builds precedents
    for d in ["build", "dist"]:
        p = module_dir / d
        if p.exists():
            shutil.rmtree(p)

    # Nettoyer les .c generes par Cython des builds precedents
    for c_file in module_dir.rglob("*.c"):
        if c_file.parent.name != "modeles":
            c_file.unlink()

    print(f"\n{'=' * 50}")
    print(f"  Build : {module}")
    print(f"{'=' * 50}")

    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=module_dir,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  ECHEC : {module} (code {result.returncode})")
        return None

    # Trouver le .whl genere
    wheels = list((module_dir / "dist").glob("*.whl"))
    if not wheels:
        print(f"  ECHEC : aucun .whl genere pour {module}")
        return None

    whl = wheels[0]

    # Verifier l'absence de .py
    problemes = verifier_wheel(whl)
    if problemes:
        print(f"  ATTENTION : .py trouves dans {whl.name} :")
        for p in problemes:
            print(f"    - {p}")
        return None

    print(f"  OK : {whl.name}")
    return whl


def main():
    parser = argparse.ArgumentParser(description="Build des wheels binaires Lectura")
    parser.add_argument(
        "modules",
        nargs="*",
        default=MODULES,
        help=f"Modules a builder (defaut: tous). Choix : {', '.join(MODULES)}",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verifie les .whl existants dans dist/ sans rebuilder",
    )
    args = parser.parse_args()

    # Valider les noms de modules
    for m in args.modules:
        if m not in MODULES:
            print(f"Module inconnu : {m}. Choix : {', '.join(MODULES)}")
            sys.exit(1)

    if args.check_only:
        print("\nVerification des wheels dans dist/ :")
        if not DIST.exists():
            print("  dist/ introuvable")
            sys.exit(1)
        ok = True
        for whl in sorted(DIST.glob("*.whl")):
            problemes = verifier_wheel(whl)
            if problemes:
                print(f"  PROBLEME {whl.name} :")
                for p in problemes:
                    print(f"    - {p}")
                ok = False
            else:
                print(f"  OK {whl.name}")
        sys.exit(0 if ok else 1)

    # Build
    DIST.mkdir(exist_ok=True)

    resultats: dict[str, str] = {}

    for module in args.modules:
        whl = build_module(module)
        if whl:
            dest = DIST / whl.name
            shutil.copy2(whl, dest)
            resultats[module] = f"OK → {whl.name}"
        else:
            resultats[module] = "ECHEC"

    # Resume
    print(f"\n{'=' * 50}")
    print("  Resume")
    print(f"{'=' * 50}")
    for module, status in resultats.items():
        print(f"  {module:<15} {status}")

    echecs = sum(1 for s in resultats.values() if s == "ECHEC")
    if echecs:
        print(f"\n  {echecs} echec(s) sur {len(resultats)} module(s)")
        sys.exit(1)
    else:
        print(f"\n  {len(resultats)} wheel(s) dans {DIST}/")
        print("  Prochaine etape : twine upload dist/*")


if __name__ == "__main__":
    main()
