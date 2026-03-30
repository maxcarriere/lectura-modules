#!/usr/bin/env python3
"""Prepare le dataset d'alignement P2G depuis un dictionnaire de prononciation.

Lit un dictionnaire (ortho, phone) et produit un CSV avec alignements
phoneme-par-phoneme pour l'entrainement CRF P2G.

En P2G, les features sont extraites depuis les caracteres IPA et les labels
sont des graphemes (inverse du G2P).

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
import unicodedata
from pathlib import Path

_CONT = "_CONT"


def iter_phonemes(ipa: str) -> list[str]:
    """Segmente une chaine IPA en phonemes."""
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


def align_phone_ortho(phone_str: str, ortho: str) -> list[str] | None:
    """Aligne une transcription phonemique avec un mot (P2G).

    Retourne une liste de labels (un par phoneme), ou None si l'alignement echoue.
    Les labels sont des graphemes, avec _CONT pour les phonemes sans grapheme propre.
    """
    phonemes = iter_phonemes(phone_str)
    n_phones = len(phonemes)
    n_chars = len(ortho)

    if n_phones == 0 or n_chars == 0:
        return None

    # Cas simple : meme nombre → 1:1
    if n_phones == n_chars:
        return list(ortho)

    # Plus de phonemes que de caracteres → distribuer avec _CONT
    if n_phones > n_chars:
        labels = [_CONT] * n_phones
        step = n_phones / n_chars
        for i, ch in enumerate(ortho):
            pos = min(int(i * step), n_phones - 1)
            while pos < n_phones and labels[pos] != _CONT:
                pos += 1
            if pos < n_phones:
                labels[pos] = ch
        # Verifier que le premier phoneme a un label
        if labels[0] == _CONT and ortho:
            labels[0] = ortho[0]
        return labels

    # Plus de caracteres que de phonemes → concatener des graphemes
    if n_chars > n_phones:
        labels = [""] * n_phones
        step = n_chars / n_phones
        for i in range(n_phones):
            start = int(i * step)
            end = int((i + 1) * step)
            end = min(end, n_chars)
            if start >= n_chars:
                labels[i] = _CONT
            else:
                labels[i] = ortho[start:end]
        return labels

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparation dataset P2G")
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
    entries: list[tuple[str, str, list[str]]] = []
    skipped = 0

    with open(dico_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()
            if not ortho or not phone or " " in ortho:
                continue
            labels = align_phone_ortho(phone, ortho)
            if labels is None:
                skipped += 1
                continue
            entries.append((phone, ortho, labels))

    print(f"  {len(entries)} alignements reussis, {skipped} ignores")

    # Melanger et splitter
    random.seed(args.seed)
    random.shuffle(entries)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_idx = int(len(entries) * args.split)
        train_entries = entries[:split_idx]
        eval_entries = entries[split_idx:]

        for name, data in [("train_p2g.csv", train_entries),
                           ("eval_p2g.csv", eval_entries)]:
            path = out_dir / name
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ipa", "ortho", "aligned_labels"])
                for phone, ortho, labels in data:
                    writer.writerow([phone, ortho, ",".join(labels)])
            print(f"  {path} : {len(data)} entrees")
    elif args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ipa", "ortho", "aligned_labels"])
            for phone, ortho, labels in entries:
                writer.writerow([phone, ortho, ",".join(labels)])
        print(f"  {out_path} : {len(entries)} entrees")
    else:
        print("ERREUR : specifier --output ou --output-dir", file=sys.stderr)
        sys.exit(1)

    print("Termine.")


if __name__ == "__main__":
    main()
