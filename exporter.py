#!/usr/bin/env python3
"""
Export les fichiers du workspace vers output/Modules/ pour publication.

Copie tous les fichiers trackes par git (= version propre apres nettoyage)
puis ajoute les modeles numpy (trop lourds pour le git local mais necessaires
pour GitHub/PyPI).

Usage :
    python exporter.py              Exporte vers ../../output/Modules/
    python exporter.py --dry-run    Montre ce qui serait copie sans rien faire
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Dossiers de modeles numpy a copier en plus des fichiers git
EXTRAS = [
    "G2P/modeles_numpy",
    "P2G/modeles_numpy",
]

# Fichiers a exclure de l'export (outils internes du workspace)
EXCLUDE = [
    "exporter.py",
    "construire_wheels.py",
]

# Dossiers a exclure (modules pas encore prets)
EXCLUDE_DIRS = [
    "Correcteur",
]


def _est_source_protege(filepath: str) -> bool:
    """True si le fichier est un .py de src/ qui doit etre protege.

    On garde les __init__.py (surface API) mais on exclut le reste
    pour ne pas exposer le code source sur GitHub.
    """
    return ("/src/" in filepath
            and filepath.endswith(".py")
            and not filepath.endswith("__init__.py"))


def get_git_files(repo_root: Path) -> list[str]:
    """Renvoie la liste des fichiers trackes par git."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.splitlines() if f]


def collect_extra_files(repo_root: Path) -> list[str]:
    """Collecte les fichiers des dossiers extras (modeles_numpy, etc.)."""
    files = []
    for extra_dir in EXTRAS:
        extra_path = repo_root / extra_dir
        if extra_path.is_dir():
            for root, _, filenames in os.walk(extra_path):
                for filename in filenames:
                    full = Path(root) / filename
                    files.append(str(full.relative_to(repo_root)))
    return files


def format_size(size_bytes: int) -> str:
    """Formate une taille en octets en unite lisible."""
    if size_bytes < 1024:
        return f"{size_bytes} o"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} Ko"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} Mo"


def main():
    parser = argparse.ArgumentParser(description="Export workspace vers output/Modules/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche ce qui serait copie sans rien faire",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root.parent.parent / "output" / "Modules"

    # Collecter les fichiers
    git_files = get_git_files(repo_root)
    extra_files = collect_extra_files(repo_root)

    # Fusionner sans doublons, en gardant l'ordre
    all_files = list(dict.fromkeys(git_files + extra_files))

    # Exclure les fichiers internes, les dossiers pas prets, et les .py proteges
    all_files = [
        f for f in all_files
        if f not in EXCLUDE
        and not any(f.startswith(d + "/") for d in EXCLUDE_DIRS)
        and not _est_source_protege(f)
    ]

    # Verifier que tous les fichiers existent
    missing = [f for f in all_files if not (repo_root / f).exists()]
    if missing:
        print(f"ATTENTION : {len(missing)} fichier(s) manquant(s) :")
        for f in missing[:10]:
            print(f"  - {f}")
        if len(missing) > 10:
            print(f"  ... et {len(missing) - 10} autres")

    existing_files = [f for f in all_files if (repo_root / f).exists()]

    # Stats par module
    modules: dict[str, dict] = {}
    for f in existing_files:
        parts = f.split("/")
        module = parts[0] if len(parts) > 1 else "(racine)"
        if module not in modules:
            modules[module] = {"count": 0, "size": 0}
        modules[module]["count"] += 1
        modules[module]["size"] += (repo_root / f).stat().st_size

    total_size = sum(m["size"] for m in modules.values())
    total_count = len(existing_files)

    # Afficher le resume
    print(f"\n{'=' * 55}")
    if args.dry_run:
        print("  MODE DRY-RUN — rien ne sera copie")
    print(f"  Export : {repo_root}")
    print(f"     ->   {output_dir}")
    print(f"{'=' * 55}")
    print(f"\n  {'Module':<20} {'Fichiers':>10} {'Taille':>10}")
    print(f"  {'-' * 42}")
    for module in sorted(modules.keys()):
        m = modules[module]
        print(f"  {module:<20} {m['count']:>10} {format_size(m['size']):>10}")
    print(f"  {'-' * 42}")
    print(f"  {'TOTAL':<20} {total_count:>10} {format_size(total_size):>10}")

    # Fichiers extras
    if extra_files:
        print(f"\n  Extras (non-git) : {len(extra_files)} fichier(s)")
        for f in extra_files:
            size = (repo_root / f).stat().st_size if (repo_root / f).exists() else 0
            print(f"    + {f} ({format_size(size)})")

    if args.dry_run:
        print(f"\n  Dry-run termine. Relancer sans --dry-run pour copier.\n")
        return

    # Nettoyer l'ancien output (en preservant .git/)
    if output_dir.exists():
        print(f"\n  Nettoyage de {output_dir} ...")
        for item in output_dir.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    # Copier les fichiers
    print(f"  Copie de {total_count} fichiers ...")
    copied = 0
    for f in existing_files:
        src = repo_root / f
        dst = output_dir / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"  {copied} fichiers copies.")

    # Verification
    actual_size = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    )
    actual_count = sum(1 for f in output_dir.rglob("*") if f.is_file())
    print(f"\n  Verification : {actual_count} fichiers, {format_size(actual_size)}")
    print(f"  Export termine avec succes !\n")


if __name__ == "__main__":
    main()
