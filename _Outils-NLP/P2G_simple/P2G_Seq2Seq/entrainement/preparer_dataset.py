#!/usr/bin/env python3
"""Prepare le dataset P2G Seq2Seq depuis un dictionnaire de prononciation.

Lit un dictionnaire (ortho, phone) et produit un CSV avec les paires
(phone, ortho) pour l'entrainement Seq2Seq P2G.

En Seq2Seq, l'entree est la chaine IPA complete et la sortie est
l'orthographe caractere par caractere.

Pre-requis :
    Le fichier dico.csv doit avoir les colonnes : ortho, phone

Usage :
    python preparer_dataset.py --dico dico.csv --output train_p2g.csv
    python preparer_dataset.py --dico dico.csv --split 0.9 --output-dir donnees/
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparation dataset P2G (Seq2Seq)")
    parser.add_argument("--dico", required=True, help="Fichier dico.csv")
    parser.add_argument("--output", default=None, help="Fichier CSV de sortie")
    parser.add_argument("--output-dir", default=None,
                        help="Dossier pour train/eval split")
    parser.add_argument("--split", type=float, default=0.9,
                        help="Ratio train/eval (defaut: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dico_path = Path(args.dico)
    if not dico_path.exists():
        print(f"ERREUR : {dico_path} non trouve", file=sys.stderr)
        sys.exit(1)

    print("Chargement du dictionnaire...")
    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    with open(dico_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()
            if not ortho or not phone or " " in ortho:
                continue
            key = (phone, ortho)
            if key not in seen:
                seen.add(key)
                entries.append(key)

    print(f"  {len(entries)} paires chargees")

    # Melanger et splitter
    random.seed(args.seed)
    random.shuffle(entries)

    def write_csv(path: Path, data: list[tuple[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phone", "ortho"])
            for phone, ortho in data:
                writer.writerow([phone, ortho])
        print(f"  {path} : {len(data)} entrees")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_idx = int(len(entries) * args.split)
        write_csv(out_dir / "train_p2g.csv", entries[:split_idx])
        write_csv(out_dir / "eval_p2g.csv", entries[split_idx:])
    elif args.output:
        write_csv(Path(args.output), entries)
    else:
        print("ERREUR : specifier --output ou --output-dir", file=sys.stderr)
        sys.exit(1)

    print("Termine.")


if __name__ == "__main__":
    main()
