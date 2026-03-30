#!/usr/bin/env python3
"""Prépare le dataset de paires G2P depuis dico.csv pour le Seq2Seq.

Lit un dictionnaire de prononciation (ortho, phone) et produit un CSV
avec paires (ortho, phone) pour l'entraînement encoder-decoder.

Contrairement au CRF/BiLSTM qui nécessitent un alignement caractère par
caractère, le Seq2Seq apprend directement la correspondance séquence
d'entrée → séquence de sortie.

Pré-requis :
    Le fichier dico.csv doit avoir les colonnes : ortho, phone
    (optionnel : POS, freq)

Usage :
    python preparer_dataset.py --dico dico.csv --output train_g2p.csv
    python preparer_dataset.py --dico dico.csv --split 0.9 --output-dir donnees/
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import unicodedata
from pathlib import Path


def iter_phonemes(ipa: str) -> list[str]:
    """Regroupe chaque caractère de base avec ses combining marks."""
    if not ipa:
        return []
    phonemes: list[str] = []
    current = ""
    for ch in ipa:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            current += ch
        else:
            if current:
                phonemes.append(current)
            current = ch
    if current:
        phonemes.append(current)
    return phonemes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Préparation dataset G2P (Seq2Seq)"
    )
    parser.add_argument("--dico", required=True, help="Fichier dico.csv")
    parser.add_argument("--output", default=None, help="Fichier CSV de sortie")
    parser.add_argument("--output-dir", default=None,
                        help="Dossier pour train/eval split")
    parser.add_argument("--split", type=float, default=0.9,
                        help="Ratio train/eval (défaut: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dico_path = Path(args.dico)
    if not dico_path.exists():
        print(f"ERREUR : {dico_path} non trouvé", file=sys.stderr)
        sys.exit(1)

    print("Chargement du dictionnaire...")
    entries: list[tuple[str, str, list[str]]] = []
    skipped = 0

    with open(dico_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()
            if not ortho or not phone or " " in ortho:
                skipped += 1
                continue
            # Tokenisation des phonèmes pour le Seq2Seq
            phonemes = iter_phonemes(phone)
            if not phonemes:
                skipped += 1
                continue
            entries.append((ortho, phone, phonemes))

    print(f"  {len(entries)} paires valides, {skipped} ignorées")

    # Mélanger et splitter
    random.seed(args.seed)
    random.shuffle(entries)

    def write_csv(path: Path, data: list[tuple[str, str, list[str]]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ortho", "phone", "phone_tokens"])
            for ortho, phone, phonemes in data:
                writer.writerow([ortho, phone, " ".join(phonemes)])
        print(f"  {path} : {len(data)} entrées")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_idx = int(len(entries) * args.split)
        train_entries = entries[:split_idx]
        eval_entries = entries[split_idx:]

        write_csv(out_dir / "train_g2p.csv", train_entries)
        write_csv(out_dir / "eval_g2p.csv", eval_entries)
    elif args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(out_path, entries)
    else:
        print("ERREUR : spécifier --output ou --output-dir", file=sys.stderr)
        sys.exit(1)

    print("Terminé.")


if __name__ == "__main__":
    main()
