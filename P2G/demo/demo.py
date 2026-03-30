#!/usr/bin/env python3
"""Démo interactive du modèle P2G unifié.

Usage :
    python demo/demo.py --modele modeles/unifie_p2g.pt --donnees entrainement/donnees/

Tapez une phrase IPA (mots séparés par des espaces) :
    Input : le ʃa ɛ bɔ̃
    Output : le chat est bon (NOM VER ART:def ADJ)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
from lectura_p2g.modele import UnifiedP2G
from lectura_p2g.utils.p2g_labels import reconstruct_ortho, _CONT
from lectura_p2g.tokeniseur import tokeniser_ipa, ipa_phrase_vers_chars


def main() -> None:
    parser = argparse.ArgumentParser(description="Démo P2G unifié")
    parser.add_argument(
        "--modele", type=Path,
        default=_ROOT / "modeles" / "unifie_p2g.pt",
    )
    parser.add_argument(
        "--donnees", type=Path,
        default=_ROOT / "entrainement" / "donnees",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Chargement modèle : {args.modele}")
    checkpoint = torch.load(args.modele, map_location=device, weights_only=False)
    vocabs = checkpoint["vocabs"]
    config = checkpoint["config"]

    model = UnifiedP2G.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} paramètres")

    char2idx = vocabs["char2idx"]
    idx2p2g = {v: k for k, v in vocabs["p2g_label2idx"].items()}
    idx2pos = {v: k for k, v in vocabs["pos2idx"].items()}
    idx2morpho = {}
    for feat, vocab in vocabs["morpho_vocabs"].items():
        idx2morpho[feat] = {v: k for k, v in vocab.items()}

    print("\nTapez une phrase IPA (mots séparés par des espaces).")
    print("Exemples : le ʃa ɛ bɔ̃")
    print("Tapez 'q' pour quitter.\n")

    while True:
        try:
            line = input("IPA > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line or line.lower() == "q":
            break

        ipa_words = tokeniser_ipa(line)
        if not ipa_words:
            continue

        chars, word_ids, word_starts, word_ends = ipa_phrase_vers_chars(ipa_words)

        char_ids = torch.tensor(
            [[char2idx.get(ch, 1) for ch in chars]], dtype=torch.long
        ).to(device)
        ws = torch.tensor([word_starts], dtype=torch.long).to(device)
        we = torch.tensor([word_ends], dtype=torch.long).to(device)
        wl = torch.tensor([len(ipa_words)], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(char_ids, None, ws, we, wl)

        # P2G
        p2g_preds = outputs["p2g_logits"][0].argmax(dim=-1).cpu().tolist()
        ortho_words = []
        for w in range(len(ipa_words)):
            s, e = word_starts[w], word_ends[w]
            labels = [idx2p2g.get(p2g_preds[i], _CONT) for i in range(s, e + 1)]
            ortho_words.append(reconstruct_ortho(labels))

        # POS
        pos_words = []
        if "pos_logits" in outputs:
            pos_preds = outputs["pos_logits"][0].argmax(dim=-1).cpu().tolist()
            for w in range(len(ipa_words)):
                pos_words.append(idx2pos.get(pos_preds[w], "?"))

        # Morpho
        morpho_info = {}
        for feat_name, idx2label in idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key in outputs:
                feat_preds = outputs[key][0].argmax(dim=-1).cpu().tolist()
                morpho_info[feat_name] = [
                    idx2label.get(feat_preds[w], "_") for w in range(len(ipa_words))
                ]

        # Display
        print(f"  Ortho : {' '.join(ortho_words)}")
        if pos_words:
            print(f"  POS   : {' '.join(pos_words)}")
        for feat_name, values in morpho_info.items():
            non_blank = [(ipa_words[i], v) for i, v in enumerate(values) if v != "_"]
            if non_blank:
                print(f"  {feat_name:8s}: {', '.join(f'{w}={v}' for w, v in non_blank)}")
        print()


if __name__ == "__main__":
    main()
