#!/usr/bin/env python3
"""Évalue le modèle unifié P2G sur toutes les tâches.

Métriques :
- P2G : Word Accuracy, CER (Character Error Rate)
- POS : Accuracy
- Morpho : Accuracy par feature (Number, Gender, VerbForm, Mood, Tense, Person)

Usage :
    python entrainement/evaluer.py --modele modeles/unifie_p2g.pt --donnees entrainement/donnees/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from lectura_p2g.modele import UnifiedP2G, UnifiedP2GDataset, collate_unified
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


def evaluate(
    model: UnifiedP2G,
    dataset: UnifiedP2GDataset,
    vocabs: dict,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Évalue toutes les tâches et retourne les métriques."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_unified,
    )

    idx2p2g = {v: k for k, v in vocabs["p2g_label2idx"].items()}
    idx2pos = {v: k for k, v in vocabs["pos2idx"].items()}
    idx2morpho = {}
    for feat, vocab in vocabs["morpho_vocabs"].items():
        idx2morpho[feat] = {v: k for k, v in vocab.items()}

    # Accumulateurs
    p2g_word_total = 0
    p2g_word_correct = 0
    p2g_char_total = 0
    p2g_edit_sum = 0

    pos_total = 0
    pos_correct = 0

    morpho_total: dict[str, int] = defaultdict(int)
    morpho_correct: dict[str, int] = defaultdict(int)

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

            # P2G evaluation (per sentence/word)
            p2g_logits = outputs["p2g_logits"]
            p2g_preds = p2g_logits.argmax(dim=-1).cpu()
            p2g_targets = batch_data["p2g_ids"]

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

                    # Skip words with PAD gold labels (unaligned)
                    if any(g == "<PAD>" for g in gold_word):
                        continue

                    p2g_word_total += 1
                    if pred_word == gold_word:
                        p2g_word_correct += 1

                    pred_ortho = reconstruct_ortho(pred_word)
                    gold_ortho = reconstruct_ortho(gold_word)
                    pred_chars = list(pred_ortho)
                    gold_chars = list(gold_ortho)
                    p2g_char_total += len(gold_chars)
                    p2g_edit_sum += _levenshtein(pred_chars, gold_chars)

            # POS evaluation
            if "pos_logits" in outputs:
                pos_preds = outputs["pos_logits"].argmax(dim=-1).cpu()
                pos_targets = batch_data["pos_ids"]

                for b in range(char_ids.size(0)):
                    wl = word_lengths[b].item()
                    for w in range(wl):
                        pred = idx2pos.get(pos_preds[b, w].item(), "<PAD>")
                        gold = idx2pos.get(pos_targets[b, w].item(), "<PAD>")
                        if gold == "<PAD>":
                            continue
                        pos_total += 1
                        if pred == gold:
                            pos_correct += 1

            # Morpho evaluation
            for feat_name in idx2morpho:
                key = f"morpho_{feat_name}_logits"
                if key not in outputs:
                    continue

                feat_preds = outputs[key].argmax(dim=-1).cpu()
                feat_targets = batch_data["morpho_ids"][feat_name]

                for b in range(char_ids.size(0)):
                    wl = word_lengths[b].item()
                    for w in range(wl):
                        pred = idx2morpho[feat_name].get(feat_preds[b, w].item(), "<PAD>")
                        gold = idx2morpho[feat_name].get(feat_targets[b, w].item(), "<PAD>")
                        if gold == "<PAD>" or gold == "_":
                            continue
                        morpho_total[feat_name] += 1
                        if pred == gold:
                            morpho_correct[feat_name] += 1

    # ── Compile metrics ──
    metrics = {}

    # P2G
    metrics["p2g_word_acc"] = p2g_word_correct / p2g_word_total if p2g_word_total else 0
    metrics["p2g_cer"] = p2g_edit_sum / p2g_char_total if p2g_char_total else 0
    metrics["p2g_n_words"] = p2g_word_total

    # POS
    metrics["pos_acc"] = pos_correct / pos_total if pos_total else 0
    metrics["pos_n_tokens"] = pos_total

    # Morpho
    morpho_acc = {}
    for feat in sorted(morpho_total):
        acc = morpho_correct[feat] / morpho_total[feat] if morpho_total[feat] else 0
        morpho_acc[feat] = {"acc": acc, "n": morpho_total[feat]}
    metrics["morpho"] = morpho_acc

    return metrics


def print_metrics(metrics: dict) -> None:
    """Affiche les métriques de façon formatée."""
    print("\n" + "=" * 60)
    print("RÉSULTATS D'ÉVALUATION")
    print("=" * 60)

    print(f"\n── P2G ({metrics['p2g_n_words']} mots) ──")
    print(f"  Word Accuracy : {metrics['p2g_word_acc']:.1%}")
    print(f"  CER           : {metrics['p2g_cer']:.1%}")

    print(f"\n── POS ({metrics['pos_n_tokens']} tokens) ──")
    print(f"  Accuracy : {metrics['pos_acc']:.1%}")

    print(f"\n── Morphologie ──")
    for feat, d in metrics.get("morpho", {}).items():
        print(f"  {feat:12s} : {d['acc']:.1%} ({d['n']} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue le modèle P2G unifié")
    parser.add_argument(
        "--modele", type=Path, default=_ROOT / "modeles" / "unifie_p2g.pt",
    )
    parser.add_argument(
        "--donnees", type=Path,
        default=Path(__file__).parent / "donnees",
    )
    parser.add_argument(
        "--split", default="test",
        choices=["dev", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print(f"Chargement modèle : {args.modele}")
    checkpoint = torch.load(args.modele, map_location=device, weights_only=False)
    vocabs = checkpoint["vocabs"]
    config = checkpoint["config"]

    model = UnifiedP2G.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  {sum(p.numel() for p in model.parameters()):,} paramètres")

    with open(args.donnees / "phone_to_graphs.json", encoding="utf-8") as f:
        phone_to_graphs = json.load(f)

    with open(args.donnees / f"sentences_{args.split}.json", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"  {args.split}: {len(sentences)} phrases")

    print("Préparation dataset...")
    dataset = UnifiedP2GDataset(sentences, vocabs, phone_to_graphs)

    print("Évaluation...")
    metrics = evaluate(model, dataset, vocabs, device, args.batch_size)
    print_metrics(metrics)

    metrics_path = args.modele.parent / f"metrics_p2g_{args.split}.json"
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            serializable[k] = {
                sk: sv if not isinstance(sv, dict)
                else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in sv.items()}
                for sk, sv in v.items()
            }
        elif isinstance(v, float):
            serializable[k] = round(v, 4)
        else:
            serializable[k] = v

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\nMétriques sauvegardées : {metrics_path}")


if __name__ == "__main__":
    main()
