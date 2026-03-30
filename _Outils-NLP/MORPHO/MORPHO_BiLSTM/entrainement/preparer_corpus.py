#!/usr/bin/env python3
"""Télécharge et fusionne les treebanks Universal Dependencies français.

Treebanks (licences libres) :
  - UD_French-GSD      (~16K phrases, CC BY-SA 4.0)
  - UD_French-Sequoia  (~3K phrases, LGPL-LR)
  - UD_French-Rhapsodie (~1.3K phrases, CC BY-SA 4.0)

Usage :
    python preparer_corpus.py
    python preparer_corpus.py --output donnees --ud-dir ud_repos

Note : les données CoNLL-U sont partagées avec POS_CRF. Si elles sont
déjà présentes dans POS/POS_CRF/entrainement/donnees/, vous pouvez
y faire référence directement sans les re-télécharger.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

UD_REPOS = [
    ("UD_French-GSD", "https://github.com/UniversalDependencies/UD_French-GSD.git"),
    ("UD_French-Sequoia", "https://github.com/UniversalDependencies/UD_French-Sequoia.git"),
    ("UD_French-Rhapsodie", "https://github.com/UniversalDependencies/UD_French-Rhapsodie.git"),
]

_CONLLU_PREFIXES = {
    "UD_French-GSD": "fr_gsd-ud",
    "UD_French-Sequoia": "fr_sequoia-ud",
    "UD_French-Rhapsodie": "fr_rhapsodie-ud",
}


def clone_or_update(repo_name: str, repo_url: str, base_dir: Path) -> Path:
    """Clone un dépôt UD ou le met à jour."""
    target = base_dir / repo_name
    if target.exists():
        print(f"  {repo_name} existe, mise à jour...")
        try:
            subprocess.run(
                ["git", "-C", str(target), "pull", "--ff-only"],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            print(f"    Pull échoué, version actuelle conservée")
    else:
        print(f"  Clonage de {repo_name}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target)],
            check=True,
        )
    return target


def find_conllu_files(repo_dir: Path, prefix: str, split: str) -> list[Path]:
    pattern = f"{prefix}-{split}.conllu"
    files = list(repo_dir.glob(pattern))
    if not files:
        files = list(repo_dir.rglob(f"*-{split}.conllu"))
    return files


def merge_conllu(files: list[Path], output: Path) -> int:
    sentence_count = 0
    with open(output, "w", encoding="utf-8") as out:
        for filepath in files:
            with open(filepath, encoding="utf-8") as f:
                in_sentence = False
                for line in f:
                    out.write(line)
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        in_sentence = True
                    elif not stripped and in_sentence:
                        sentence_count += 1
                        in_sentence = False
                if in_sentence:
                    out.write("\n")
                    sentence_count += 1
    return sentence_count


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Télécharge et fusionne les corpus UD français"
    )
    parser.add_argument(
        "--output", type=Path, default=here / "donnees",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--ud-dir", type=Path, default=here / "ud_repos",
        help="Répertoire des dépôts UD clonés",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.ud_dir.mkdir(parents=True, exist_ok=True)

    print("Téléchargement des treebanks UD français...")
    repo_dirs: dict[str, Path] = {}
    for name, url in UD_REPOS:
        try:
            repo_dirs[name] = clone_or_update(name, url, args.ud_dir)
        except Exception as e:
            print(f"  ERREUR pour {name}: {e}")

    if not repo_dirs:
        print("ERREUR : aucun corpus téléchargé !")
        sys.exit(1)

    for split in ("train", "dev", "test"):
        print(f"\nFusion du split '{split}'...")
        all_files: list[Path] = []

        for name, repo_dir in repo_dirs.items():
            prefix = _CONLLU_PREFIXES.get(name, "")
            if not prefix:
                continue
            files = find_conllu_files(repo_dir, prefix, split)
            if files:
                all_files.extend(files)
                print(f"  {name}: {len(files)} fichier(s)")
            else:
                print(f"  {name}: pas de fichier {split}")

        if not all_files:
            print(f"  Aucun fichier pour le split '{split}'")
            continue

        output_path = args.output / f"morpho_{split}_merged.conllu"
        n_sentences = merge_conllu(all_files, output_path)
        print(f"  → {output_path} ({n_sentences} phrases)")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
