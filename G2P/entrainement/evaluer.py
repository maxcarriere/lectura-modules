#!/usr/bin/env python3
"""Évalue le modèle unifié sur toutes les tâches.

Métriques :
- G2P : Word Accuracy, Phoneme Error Rate (PER)
- POS : Accuracy
- Morpho : Accuracy par feature (Number, Gender, VerbForm, Mood, Tense, Person)
- Liaison : Precision / Recall / F1 par classe et macro

Usage :
    python entrainement/evaluer.py --modele modeles/unifie.pt --donnees entrainement/donnees/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from lectura_nlp.modele import UnifiedFrenchNLP, UnifiedDataset, collate_unified
from lectura_nlp.utils.g2p_labels import _CONT, reconstruct_ipa
from lectura_nlp.utils.ipa import iter_phonemes


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
    model: UnifiedFrenchNLP,
    dataset: UnifiedDataset,
    vocabs: dict,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Évalue toutes les tâches et retourne les métriques."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_unified,
    )

    idx2g2p = {v: k for k, v in vocabs["g2p_label2idx"].items()}
    idx2pos = {v: k for k, v in vocabs["pos2idx"].items()}
    idx2liaison = {v: k for k, v in vocabs["liaison2idx"].items()}
    idx2morpho = {}
    for feat, vocab in vocabs["morpho_vocabs"].items():
        idx2morpho[feat] = {v: k for k, v in vocab.items()}

    # Accumulateurs
    g2p_word_total = 0
    g2p_word_correct = 0
    g2p_phoneme_total = 0
    g2p_edit_sum = 0

    pos_total = 0
    pos_correct = 0
    pos_confusion: dict[str, Counter] = defaultdict(Counter)

    liaison_tp: dict[str, int] = defaultdict(int)
    liaison_fp: dict[str, int] = defaultdict(int)
    liaison_fn: dict[str, int] = defaultdict(int)

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

            # G2P evaluation (per sentence/word)
            g2p_logits = outputs["g2p_logits"]
            g2p_preds = g2p_logits.argmax(dim=-1).cpu()
            g2p_targets = batch_data["g2p_ids"]

            for b in range(char_ids.size(0)):
                cl = char_lengths[b].item()
                pred_labels_all = g2p_preds[b, :cl].tolist()
                gold_labels_all = g2p_targets[b, :cl].tolist()

                # Extract per-word G2P labels using word boundaries
                wl = word_lengths[b].item()
                for w in range(wl):
                    ws = word_starts[b, w].item()
                    we = word_ends[b, w].item()
                    if ws >= cl or we >= cl:
                        continue

                    pred_word = [idx2g2p.get(pred_labels_all[i], _CONT) for i in range(ws, we + 1)]
                    gold_word = [idx2g2p.get(gold_labels_all[i], "<PAD>") for i in range(ws, we + 1)]

                    # Skip words with PAD gold labels (unaligned)
                    if any(g == "<PAD>" for g in gold_word):
                        continue

                    g2p_word_total += 1
                    if pred_word == gold_word:
                        g2p_word_correct += 1

                    pred_ipa = reconstruct_ipa(pred_word)
                    gold_ipa = reconstruct_ipa(gold_word)
                    pred_ph = iter_phonemes(pred_ipa)
                    gold_ph = iter_phonemes(gold_ipa)
                    g2p_phoneme_total += len(gold_ph)
                    g2p_edit_sum += _levenshtein(pred_ph, gold_ph)

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
                        pos_confusion[gold][pred] += 1

            # Liaison evaluation
            if "liaison_logits" in outputs:
                liaison_preds = outputs["liaison_logits"].argmax(dim=-1).cpu()
                liaison_targets = batch_data["liaison_ids"]

                for b in range(char_ids.size(0)):
                    wl = word_lengths[b].item()
                    for w in range(wl):
                        pred = idx2liaison.get(liaison_preds[b, w].item(), "<PAD>")
                        gold = idx2liaison.get(liaison_targets[b, w].item(), "<PAD>")
                        if gold == "<PAD>":
                            continue

                        if pred == gold:
                            if pred != "none":
                                liaison_tp[pred] += 1
                        else:
                            if pred != "none":
                                liaison_fp[pred] += 1
                            if gold != "none":
                                liaison_fn[gold] += 1

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

    # G2P
    metrics["g2p_word_acc"] = g2p_word_correct / g2p_word_total if g2p_word_total else 0
    metrics["g2p_per"] = g2p_edit_sum / g2p_phoneme_total if g2p_phoneme_total else 0
    metrics["g2p_n_words"] = g2p_word_total

    # POS
    metrics["pos_acc"] = pos_correct / pos_total if pos_total else 0
    metrics["pos_n_tokens"] = pos_total

    # Liaison
    liaison_classes = set(liaison_tp) | set(liaison_fp) | set(liaison_fn)
    liaison_details = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    n_classes = 0

    for cls in sorted(liaison_classes):
        tp = liaison_tp.get(cls, 0)
        fp = liaison_fp.get(cls, 0)
        fn = liaison_fn.get(cls, 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        liaison_details[cls] = {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        macro_p += p
        macro_r += r
        macro_f1 += f1
        n_classes += 1

    if n_classes > 0:
        metrics["liaison_macro_p"] = macro_p / n_classes
        metrics["liaison_macro_r"] = macro_r / n_classes
        metrics["liaison_macro_f1"] = macro_f1 / n_classes
    else:
        metrics["liaison_macro_p"] = 0
        metrics["liaison_macro_r"] = 0
        metrics["liaison_macro_f1"] = 0
    metrics["liaison_details"] = liaison_details

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

    print(f"\n── G2P ({metrics['g2p_n_words']} mots) ──")
    print(f"  Word Accuracy : {metrics['g2p_word_acc']:.1%}")
    print(f"  PER           : {metrics['g2p_per']:.1%}")

    print(f"\n── POS ({metrics['pos_n_tokens']} tokens) ──")
    print(f"  Accuracy : {metrics['pos_acc']:.1%}")

    print(f"\n── Liaison ──")
    print(f"  Macro P  : {metrics['liaison_macro_p']:.1%}")
    print(f"  Macro R  : {metrics['liaison_macro_r']:.1%}")
    print(f"  Macro F1 : {metrics['liaison_macro_f1']:.1%}")
    for cls, d in metrics.get("liaison_details", {}).items():
        print(f"    {cls:5s} : P={d['p']:.1%} R={d['r']:.1%} F1={d['f1']:.1%} "
              f"(tp={d['tp']} fp={d['fp']} fn={d['fn']})")

    print(f"\n── Morphologie ──")
    for feat, d in metrics.get("morpho", {}).items():
        print(f"  {feat:12s} : {d['acc']:.1%} ({d['n']} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue le modèle unifié")
    parser.add_argument(
        "--modele", type=Path, default=_ROOT / "modeles" / "unifie.pt",
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

    # Charger le modèle
    print(f"Chargement modèle : {args.modele}")
    checkpoint = torch.load(args.modele, map_location=device, weights_only=False)
    vocabs = checkpoint["vocabs"]
    config = checkpoint["config"]

    model = UnifiedFrenchNLP.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  {sum(p.numel() for p in model.parameters()):,} paramètres")

    # Charger les données
    with open(args.donnees / "phone_to_graphs.json", encoding="utf-8") as f:
        phone_to_graphs = json.load(f)

    with open(args.donnees / f"sentences_{args.split}.json", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"  {args.split}: {len(sentences)} phrases")

    # Dataset
    print("Préparation dataset...")
    dataset = UnifiedDataset(sentences, vocabs, phone_to_graphs)

    # Évaluer
    print("Évaluation...")
    metrics = evaluate(model, dataset, vocabs, device, args.batch_size)
    print_metrics(metrics)

    # Sauvegarder les métriques
    metrics_path = args.modele.parent / f"metrics_{args.split}.json"
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
