#!/usr/bin/env python3
"""
Export les fichiers du workspace vers output/ pour publication.

Trois modes d'export :
- PUBLIC  → output/Modules/         Code source (AGPL), sans modeles ni serveur
                                     → GitHub + PyPI
- PRIVE   → output/Modules-private/  Tout : code + modeles + serveur
                                     → VPS / livraison client
- SERVEUR → output/Serveur/          Dossier autonome pour deploiement VPS
                                     Assemble a partir de l'export prive + lexique

Usage :
    python exporter.py                        Export public + prive
    python exporter.py --mode public          Export public uniquement
    python exporter.py --mode private         Export prive uniquement
    python exporter.py --mode serveur         Assembler output/Serveur/ (necessite export prive)
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
    "_transitions",
    # Anciens noms (remplacés par Decodeur, Monospeaker, MultiSpeaker, Diphone)
    "CTC",
    "TTS-Monospeaker",
    "TTS-MultiSpeaker",
    "TTS-Diphone",
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
    "benchmark_posttraitement.py",
]

# ── Exclusions PRIVE (→ VPS / client) ───────────────────────────────────

PRIVATE_EXCLUDE_DIRS = [
    "Exporter",
    "TTS",
    "TTS-Concat",
    "Pseudo-ortho",
    "dist_manylinux",
    "_Anciens Modules",
    "_transitions",
    # Anciens noms (remplacés par Decodeur, Monospeaker, MultiSpeaker, Diphone)
    "CTC",
    "TTS-Monospeaker",
    "TTS-MultiSpeaker",
    "TTS-Diphone",
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
    """Renvoie les fichiers non trackes, y compris ceux dans .gitignore.

    Pour l'export prive, on a besoin des modeles ONNX qui sont dans
    le .gitignore (trop gros pour git) mais doivent etre exportes.
    On filtre les artefacts de build/cache (pycache, dist, egg-info, etc.).
    """
    result = subprocess.run(
        ["git", "ls-files", "--others"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    junk_patterns = (
        "__pycache__/", ".pyc", ".egg-info/", "/dist/",
        "/build/", ".pytest_cache/", ".mypy_cache/",
    )
    return [
        f for f in result.stdout.splitlines()
        if f and not any(p in f for p in junk_patterns)
    ]


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


# ── Export SERVEUR (assemblage output/Serveur/) ────────────────────────

# Mapping : categorie → (module_name, sous-chemin vers les modeles)
SERVEUR_MODELS_MAP = {
    "g2p":              ("Phonemiseur",       "src/lectura_phonemiseur/modeles"),
    "p2g":              ("Graphemiseur",      "src/lectura_graphemiseur/modeles"),
    "ctc":              ("Decodeur",          "src/lectura_decodeur/modeles"),
    "stt_formules":     ("STT-Formules",      "src/lectura_stt_formules/modeles"),
    "tts_mono":         ("Monospeaker",       "src/lectura_monospeaker/modeles"),
    "tts_multispeaker": ("MultiSpeaker",      "src/lectura_multispeaker/modeles"),
    "tts_diphone":      ("Diphone",           "src/lectura_diphone/modeles"),
}

# VC : deux sources fusionnees dans la meme categorie
SERVEUR_VC_SOURCES = [
    ("VC-ZeroShot",   "src/lectura_vc_zeroshot/modeles"),
    ("VC-Locuteurs",  "src/lectura_vc_locuteurs/modeles"),
]

# Correcteur : data/ entiere (g2p_v2/modeles + *.db)
SERVEUR_CORRECTEUR = ("Correcteur", "src/lectura_correcteur/data")

# Fichiers deploy a copier a la racine de output/Serveur/
SERVEUR_DEPLOY_FILES = [
    "deploy.sh",
    "reinstall.sh",
    "lectura-api.service",
    "nginx.conf",
    "nginx-lexique.conf",
    "nginx-redirect.conf",
]

# Fichiers lexique a copier
SERVEUR_LEXIQUE_FILES = [
    "lexique_lectura_v6.db",
    "lexique_correcteur.db",
]


def _exporter_serveur(output_base: Path, dry_run: bool) -> None:
    """Assemble output/Serveur/ a partir de l'export prive et du lexique."""

    private_dir = output_base / "Modules-private"
    lexique_dir = output_base / "Lexique"
    serveur_dir = output_base / "Serveur"

    # Verifier les prerequis
    if not private_dir.exists():
        print("\n  ERREUR : output/Modules-private/ n'existe pas.")
        print("  Lancez d'abord : python exporter.py --mode private")
        sys.exit(1)

    if not lexique_dir.exists():
        print("\n  ERREUR : output/Lexique/ n'existe pas.")
        sys.exit(1)

    # Collecter tous les fichiers a copier : (source, destination_relative)
    copies: list[tuple[Path, str]] = []

    # 1. Code serveur → serveur/
    serveur_src = private_dir / "serveur"
    if serveur_src.exists():
        for f in serveur_src.rglob("*"):
            if f.is_file() and "__pycache__" not in str(f):
                rel = f.relative_to(serveur_src)
                # Exclure le dossier deploy/ (copie separement a la racine)
                if str(rel).startswith("deploy/") or str(rel) == "MEMO-Serveur.md":
                    continue
                copies.append((f, f"serveur/{rel}"))

    # 2. requirements.txt → racine (depuis serveur/)
    req_file = serveur_src / "requirements.txt"
    if req_file.exists():
        copies.append((req_file, "requirements.txt"))

    # 3. Fichiers deploy → racine
    deploy_src = serveur_src / "deploy"
    for filename in SERVEUR_DEPLOY_FILES:
        src = deploy_src / filename
        if src.exists():
            copies.append((src, filename))

    # 4. Modeles par categorie → models/<categorie>/
    for categorie, (module_name, subpath) in SERVEUR_MODELS_MAP.items():
        models_src = private_dir / module_name / subpath
        if models_src.exists():
            for f in models_src.rglob("*"):
                if f.is_file() and f.name != "__init__.py":
                    rel = f.relative_to(models_src)
                    copies.append((f, f"models/{categorie}/{rel}"))

    # 5. VC (deux sources fusionnees) → models/vc/
    for module_name, subpath in SERVEUR_VC_SOURCES:
        vc_src = private_dir / module_name / subpath
        if vc_src.exists():
            for f in vc_src.rglob("*"):
                if f.is_file() and f.name != "__init__.py":
                    rel = f.relative_to(vc_src)
                    copies.append((f, f"models/vc/{rel}"))

    # 6. Correcteur data → models/correcteur/
    corr_module, corr_subpath = SERVEUR_CORRECTEUR
    corr_src = private_dir / corr_module / corr_subpath
    if corr_src.exists():
        for f in corr_src.rglob("*"):
            if f.is_file() and f.name != "__init__.py":
                rel = f.relative_to(corr_src)
                copies.append((f, f"models/correcteur/{rel}"))

    # 7. Lexique → lexique/
    for filename in SERVEUR_LEXIQUE_FILES:
        src = lexique_dir / filename
        if src.exists():
            copies.append((src, f"lexique/{filename}"))

    # Calculer les stats
    categories: dict[str, dict] = {}
    total_size = 0
    for src, dst_rel in copies:
        cat = dst_rel.split("/")[0]
        if cat not in categories:
            categories[cat] = {"count": 0, "size": 0}
        fsize = src.stat().st_size
        categories[cat]["count"] += 1
        categories[cat]["size"] += fsize
        total_size += fsize

    # Afficher le resume
    print(f"\n{'=' * 55}")
    if dry_run:
        print("  MODE DRY-RUN — rien ne sera copie")
    print(f"  Export SERVEUR (assemblage deploiement VPS)")
    print(f"  Sources : {private_dir}")
    print(f"            {lexique_dir}")
    print(f"     ->    {serveur_dir}")
    print(f"{'=' * 55}")
    print(f"\n  {'Categorie':<20} {'Fichiers':>10} {'Taille':>10}")
    print(f"  {'-' * 42}")
    for cat in sorted(categories.keys()):
        c = categories[cat]
        print(f"  {cat:<20} {c['count']:>10} {format_size(c['size']):>10}")
    print(f"  {'-' * 42}")
    print(f"  {'TOTAL':<20} {len(copies):>10} {format_size(total_size):>10}")

    if dry_run:
        print(f"\n  Dry-run termine.\n")
        return

    # Nettoyer l'ancien output/Serveur/
    if serveur_dir.exists():
        print(f"\n  Nettoyage de {serveur_dir} ...")
        shutil.rmtree(serveur_dir)

    # Copier les fichiers
    print(f"  Copie de {len(copies)} fichiers ...")
    copied = 0
    for src, dst_rel in copies:
        dst = serveur_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    # Rendre les scripts executables
    for script in ("deploy.sh", "reinstall.sh"):
        script_path = serveur_dir / script
        if script_path.exists():
            script_path.chmod(script_path.stat().st_mode | 0o755)

    print(f"  {copied} fichiers copies.")

    # Verification
    actual_size = sum(
        f.stat().st_size for f in serveur_dir.rglob("*") if f.is_file()
    )
    actual_count = sum(1 for f in serveur_dir.rglob("*") if f.is_file())
    print(f"\n  Verification : {actual_count} fichiers, {format_size(actual_size)}")
    print(f"  Export serveur termine avec succes !\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export workspace vers output/ (public, prive et/ou serveur)"
    )
    parser.add_argument(
        "--mode",
        choices=["public", "private", "both", "serveur"],
        default="both",
        help="Mode d'export : public, private, both (defaut), serveur (assemblage VPS)",
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

    if args.mode == "serveur":
        _exporter_serveur(output_base, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
