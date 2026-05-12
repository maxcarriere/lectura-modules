#!/usr/bin/env python3
"""
Export les fichiers du workspace vers output/ pour publication.

Deux modes d'export :
- PUBLIC  → output/Modules/         Code source (AGPL), sans modeles ni serveur
                                     → GitHub + PyPI
- PRIVE   → output/Modules-private/  Tout : code + modeles + serveur
                                     → VPS / livraison client

Usage :
    python exporter.py                        Export public + prive
    python exporter.py --mode public          Export public uniquement
    python exporter.py --mode private         Export prive uniquement
    python exporter.py --dry-run              Apercu sans copie
    python exporter.py --mode public --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Fichiers toujours exclus (outils internes du workspace) ──────────────

ALWAYS_EXCLUDE_FILES = [
    "exporter.py",
    "construire_wheels.py",
]

# Extensions toujours exclues (artefacts de compilation)
ALWAYS_EXCLUDE_EXTENSIONS = {".c", ".so"}

# ── Exclusions PUBLIC (→ GitHub + PyPI) ──────────────────────────────────

PUBLIC_EXCLUDE_DIRS = [
    "Exporter",
    "TTS",
    "TTS-Concat",
    "Pseudo-ortho",
    "dist_manylinux",
    "serveur",
    "_Anciens Modules",
]

PUBLIC_EXCLUDE_PATTERNS = [
    "modeles/",
    "modeles_numpy/",
    "LICENCE-COMMERCIALE.md",
    "editeur_weights.json.gz",
    "ngram.db",
    "lexique_lectura.db",
    "lexique_lectura_v4.db",
    "Correcteur/src/lectura_correcteur/data/g2p_v2/",
    "lexique_correcteur.csv.gz",
    "lexique_correcteur.db",
    "Correcteur/scripts/",
    "Correcteur/benchmark/",
    "Correcteur/checkpoints/",
    "Correcteur/data/",
]

# ── Exclusions PRIVE (→ VPS / client) ───────────────────────────────────

PRIVATE_EXCLUDE_DIRS = [
    "Exporter",
    "TTS",
    "TTS-Concat",
    "Pseudo-ortho",
    "dist_manylinux",
    "_Anciens Modules",
]

PRIVATE_EXCLUDE_PATTERNS = [
    "LICENCE.txt",
    "lexique_lectura.db",
    "lexique_lectura_v4.db",
]


# ── Fonctions utilitaires ───────────────────────────────────────────────

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


def get_untracked_files(repo_root: Path) -> list[str]:
    """Renvoie les fichiers non trackes (modeles, serveur, etc.)."""
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.splitlines() if f]


def _should_exclude(
    filepath: str,
    exclude_dirs: list[str],
    exclude_patterns: list[str],
) -> bool:
    """Determine si un fichier doit etre exclu de l'export."""
    if filepath in ALWAYS_EXCLUDE_FILES:
        return True

    if any(filepath.startswith(d + "/") for d in exclude_dirs):
        return True

    if any(p in filepath for p in exclude_patterns):
        return True

    for ext in ALWAYS_EXCLUDE_EXTENSIONS:
        if filepath.endswith(ext):
            return True

    return False


def format_size(size_bytes: int) -> str:
    """Formate une taille en octets en unite lisible."""
    if size_bytes < 1024:
        return f"{size_bytes} o"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} Ko"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} Mo"


# ── Export principal ────────────────────────────────────────────────────

def _exporter(
    repo_root: Path,
    output_dir: Path,
    exclude_dirs: list[str],
    exclude_patterns: list[str],
    label: str,
    dry_run: bool,
    include_untracked: bool = False,
) -> None:
    """Exporte les fichiers vers output_dir avec les exclusions donnees."""

    # Collecter les fichiers
    all_files = get_git_files(repo_root)

    if include_untracked:
        untracked = get_untracked_files(repo_root)
        # Fusionner sans doublons
        seen = set(all_files)
        for f in untracked:
            if f not in seen:
                all_files.append(f)
                seen.add(f)

    # Appliquer les exclusions
    all_files = [
        f for f in all_files
        if not _should_exclude(f, exclude_dirs, exclude_patterns)
    ]

    # Verifier que tous les fichiers existent
    missing = [f for f in all_files if not (repo_root / f).exists()]
    if missing:
        print(f"  ATTENTION : {len(missing)} fichier(s) manquant(s) :")
        for f in missing[:10]:
            print(f"    - {f}")
        if len(missing) > 10:
            print(f"    ... et {len(missing) - 10} autres")

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
    if dry_run:
        print("  MODE DRY-RUN — rien ne sera copie")
    print(f"  {label}")
    print(f"  Source : {repo_root}")
    print(f"     ->   {output_dir}")
    print(f"{'=' * 55}")
    print(f"\n  {'Module':<20} {'Fichiers':>10} {'Taille':>10}")
    print(f"  {'-' * 42}")
    for module in sorted(modules.keys()):
        m = modules[module]
        print(f"  {module:<20} {m['count']:>10} {format_size(m['size']):>10}")
    print(f"  {'-' * 42}")
    print(f"  {'TOTAL':<20} {total_count:>10} {format_size(total_size):>10}")

    # Fichiers exclus notables
    all_candidates = get_git_files(repo_root)
    if include_untracked:
        all_candidates = list(dict.fromkeys(all_candidates + get_untracked_files(repo_root)))
    excluded_files = [
        f for f in all_candidates
        if _should_exclude(f, exclude_dirs, exclude_patterns)
    ]
    if excluded_files:
        print(f"\n  Exclus : {len(excluded_files)} fichier(s)")
        for f in excluded_files[:15]:
            print(f"    - {f}")
        if len(excluded_files) > 15:
            print(f"    ... et {len(excluded_files) - 15} autres")

    if dry_run:
        print(f"\n  Dry-run termine.\n")
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


def main():
    parser = argparse.ArgumentParser(
        description="Export workspace vers output/ (public et/ou prive)"
    )
    parser.add_argument(
        "--mode",
        choices=["public", "private", "both"],
        default="both",
        help="Mode d'export : public (GitHub/PyPI), private (VPS/client), both (defaut)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche ce qui serait copie sans rien faire",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    output_base = repo_root.parent.parent / "output"

    if args.mode in ("public", "both"):
        _exporter(
            repo_root=repo_root,
            output_dir=output_base / "Modules",
            exclude_dirs=PUBLIC_EXCLUDE_DIRS,
            exclude_patterns=PUBLIC_EXCLUDE_PATTERNS,
            label="Export PUBLIC (modules ouverts, sans modeles)",
            dry_run=args.dry_run,
        )

    if args.mode in ("private", "both"):
        _exporter(
            repo_root=repo_root,
            output_dir=output_base / "Modules-private",
            exclude_dirs=PRIVATE_EXCLUDE_DIRS,
            exclude_patterns=PRIVATE_EXCLUDE_PATTERNS,
            label="Export PRIVE (code + modeles + serveur)",
            dry_run=args.dry_run,
            include_untracked=True,
        )


if __name__ == "__main__":
    main()
