#!/usr/bin/env python3
"""Prepare le dataset d'alignement G2P depuis dico.csv.

Lit un dictionnaire de prononciation (ortho, phone) et produit un CSV
avec alignements caractere-par-caractere pour l'entrainement CRF/BiLSTM.

Pre-requis :
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
from pathlib import Path

_CONT = "_CONT"


def align_word_phone(word: str, phone_str: str) -> list[str] | None:
    """Aligne un mot avec sa transcription phonemique (heuristique simple).

    Retourne une liste de labels (un par caractere du mot), ou None
    si l'alignement echoue.

    L'heuristique distribue les phonemes sur les caracteres du mot
    de gauche a droite, en placant _CONT pour les graphemes multi-lettres.
    """
    # Import depuis le module parent
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lectura_g2p import iter_phonemes

    phonemes = iter_phonemes(phone_str)
    n_chars = len(word)
    n_phones = len(phonemes)

    if n_chars == 0 or n_phones == 0:
        return None

    # Cas simple : meme nombre → 1:1
    if n_chars == n_phones:
        return phonemes

    # Plus de caracteres que de phonemes → distribuer avec _CONT
    if n_chars > n_phones:
        labels = [_CONT] * n_chars
        # Placer les phonemes aux positions estimees
        step = n_chars / n_phones
        for i, ph in enumerate(phonemes):
            pos = min(int(i * step), n_chars - 1)
            # Trouver la prochaine position libre (non encore attribuee)
            while pos < n_chars and labels[pos] != _CONT:
                pos += 1
            if pos < n_chars:
                labels[pos] = ph
        # Verifier que le premier caractere a un label
        if labels[0] == _CONT and phonemes:
            labels[0] = phonemes[0]
        return labels

    # Plus de phonemes que de caracteres → ne peut pas aligner simplement
    # (cas rare, on skip)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparation dataset G2P")
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
            labels = align_word_phone(ortho, phone)
            if labels is None:
                skipped += 1
                continue
            entries.append((ortho, phone, labels))

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

        for name, data in [("train_g2p.csv", train_entries),
                           ("eval_g2p.csv", eval_entries)]:
            path = out_dir / name
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ortho", "phone", "aligned_labels"])
                for ortho, phone, labels in data:
                    writer.writerow([ortho, phone, ",".join(labels)])
            print(f"  {path} : {len(data)} entrees")
    elif args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ortho", "phone", "aligned_labels"])
            for ortho, phone, labels in entries:
                writer.writerow([ortho, phone, ",".join(labels)])
        print(f"  {out_path} : {len(entries)} entrees")
    else:
        print("ERREUR : specifier --output ou --output-dir", file=sys.stderr)
        sys.exit(1)

    print("Termine.")


if __name__ == "__main__":
    main()
