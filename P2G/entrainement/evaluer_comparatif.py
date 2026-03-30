#!/usr/bin/env python3
"""Évaluation comparative : v1 / v1+PP / v2 / v2+PP.

Compare les 4 configurations et affiche un tableau récapitulatif.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from lectura_p2g.modele import UnifiedP2G, UnifiedP2GDataset, collate_unified
from lectura_p2g.posttraitement import corriger_p2g
from lectura_p2g.utils.p2g_labels import _CONT, reconstruct_ortho


def _levenshtein(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def evaluate_with_pp(
    model: UnifiedP2G,
    dataset: UnifiedP2GDataset,
    vocabs: dict,
    device: torch.device,
    use_postprocessing: bool = False,
    batch_size: int = 32,
) -> dict:
    """Évalue le modèle avec ou sans post-traitement."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_unified,
    )

    idx2p2g = {v: k for k, v in vocabs["p2g_label2idx"].items()}
    idx2pos = {v: k for k, v in vocabs["pos2idx"].items()}
    idx2morpho = {}
    for feat, vocab in vocabs["morpho_vocabs"].items():
        idx2morpho[feat] = {v: k for k, v in vocab.items()}

    p2g_word_total = 0
    p2g_word_correct = 0
    p2g_char_total = 0
    p2g_edit_sum = 0

    with torch.no_grad():
        for batch_data in loader:
            char_ids = batch_data["char_ids"].to(device)
            char_lengths = batch_data["char_lengths"].to(device)
            word_starts = batch_data["word_starts"].to(device)
            word_ends = batch_data["word_ends"].to(device)
            word_lengths = batch_data["word_lengths"].to(device)

            outputs = model(
                char_ids, char_lengths,
                word_starts, word_ends, word_lengths,
            )

            p2g_preds = outputs["p2g_logits"].argmax(dim=-1).cpu()
            p2g_targets = batch_data["p2g_ids"]

            # POS + Morpho for post-processing
            pos_preds = None
            morpho_preds = {}
            if use_postprocessing:
                if "pos_logits" in outputs:
                    pos_preds = outputs["pos_logits"].argmax(dim=-1).cpu()
                for feat_name in idx2morpho:
                    key = f"morpho_{feat_name}_logits"
                    if key in outputs:
                        morpho_preds[feat_name] = outputs[key].argmax(dim=-1).cpu()

            for b in range(char_ids.size(0)):
                cl = char_lengths[b].item()
                pred_labels_all = p2g_preds[b, :cl].tolist()
                gold_labels_all = p2g_targets[b, :cl].tolist()

                wl = word_lengths[b].item()
                for w in range(wl):
                    ws = word_starts[b, w].item()
                    we = word_ends[b, w].item()
                    if ws >= cl or we >= cl:
                        continue

                    pred_word = [idx2p2g.get(pred_labels_all[i], _CONT) for i in range(ws, we + 1)]
                    gold_word = [idx2p2g.get(gold_labels_all[i], "<PAD>") for i in range(ws, we + 1)]

                    if any(g == "<PAD>" for g in gold_word):
                        continue

                    pred_ortho = reconstruct_ortho(pred_word)
                    gold_ortho = reconstruct_ortho(gold_word)

                    # Apply post-processing if enabled
                    if use_postprocessing and pos_preds is not None:
                        pos_tag = idx2pos.get(pos_preds[b, w].item(), "")
                        word_morpho = {}
                        for feat_name, feat_pred in morpho_preds.items():
                            val = idx2morpho[feat_name].get(feat_pred[b, w].item(), "_")
                            word_morpho[feat_name] = val
                        pred_ortho = corriger_p2g(pred_ortho, pos=pos_tag, morpho=word_morpho)

                    p2g_word_total += 1
                    if pred_ortho == gold_ortho:
                        p2g_word_correct += 1

                    pred_chars = list(pred_ortho)
                    gold_chars = list(gold_ortho)
                    p2g_char_total += len(gold_chars)
                    p2g_edit_sum += _levenshtein(pred_chars, gold_chars)

    word_acc = p2g_word_correct / p2g_word_total if p2g_word_total else 0
    cer = p2g_edit_sum / p2g_char_total if p2g_char_total else 0
    return {
        "word_acc": word_acc,
        "cer": cer,
        "n_words": p2g_word_total,
        "n_correct": p2g_word_correct,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    donnees = Path(__file__).parent / "donnees"
    with open(donnees / "phone_to_graphs.json", encoding="utf-8") as f:
        phone_to_graphs = json.load(f)
    with open(donnees / "sentences_test.json", encoding="utf-8") as f:
        sentences = json.load(f)

    results = {}

    for model_name, model_path in [
        ("v1", _ROOT / "modeles" / "unifie_p2g.pt"),
        ("v2", _ROOT / "modeles" / "unifie_p2g_v2.pt"),
    ]:
        if not model_path.exists():
            print(f"  {model_name} : fichier manquant ({model_path}), skip")
            continue

        print(f"\nChargement {model_name} : {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        vocabs = checkpoint["vocabs"]
        config = checkpoint["config"]

        model = UnifiedP2G.from_config(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {n_params:,} paramètres")

        dataset = UnifiedP2GDataset(sentences, vocabs, phone_to_graphs)

        # Sans post-traitement
        print(f"  Évaluation {model_name} (sans PP)...")
        r = evaluate_with_pp(model, dataset, vocabs, device, use_postprocessing=False)
        results[model_name] = r

        # Avec post-traitement
        print(f"  Évaluation {model_name}+PP...")
        r_pp = evaluate_with_pp(model, dataset, vocabs, device, use_postprocessing=True)
        results[f"{model_name}+PP"] = r_pp

    # Tableau récapitulatif
    print("\n" + "=" * 60)
    print("COMPARAISON")
    print("=" * 60)
    print(f"{'Config':<12} {'Word Acc':>10} {'CER':>8} {'Correct':>10} {'Total':>8}")
    print("-" * 60)
    for name in ["v1", "v1+PP", "v2", "v2+PP"]:
        if name in results:
            r = results[name]
            print(f"{name:<12} {r['word_acc']:>9.1%} {r['cer']:>7.2%} {r['n_correct']:>10,} {r['n_words']:>8,}")
    print("-" * 60)

    # Gains
    if "v1" in results and "v2+PP" in results:
        gain = results["v2+PP"]["word_acc"] - results["v1"]["word_acc"]
        print(f"\nGain total (v1 → v2+PP) : +{gain:.1%}")


if __name__ == "__main__":
    main()
