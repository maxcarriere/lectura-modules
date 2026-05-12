#!/usr/bin/env python3
"""Compare multiple pipeline configurations on the same corpus slice.

Tests all feature flags systematically and prints a comparison table.

Usage:
    python scripts/comparer_configs.py [--n 2000] [--offset 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(_PROJECT_ROOT), "Lexique", "src"))

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
CORPUS_PATH = os.path.join(_PROJECT_ROOT, "data", "corpus", "corpus_10000.jsonl")


def _norm(text: str) -> str:
    return " ".join(text.strip().split())


def _norm_mot(mot: str) -> str:
    return mot.lower().rstrip(".,;:!?")


def _build_word_map(src: list[str], dst: list[str]) -> dict[int, str | None]:
    sm = SequenceMatcher(None, [w.lower() for w in src], [w.lower() for w in dst])
    word_map: dict[int, str | None] = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                word_map[i1 + k] = src[i1 + k]
        elif op == "replace":
            n_src, n_dst = i2 - i1, j2 - j1
            for k in range(n_src):
                word_map[i1 + k] = dst[j1 + k] if k < n_dst else None
        elif op == "delete":
            for k in range(i2 - i1):
                word_map[i1 + k] = None
    return word_map


def _compter_insertions(src: list[str], dst: list[str]) -> list[str]:
    sm = SequenceMatcher(None, [w.lower() for w in src], [w.lower() for w in dst])
    insertions: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            insertions.extend(dst[j1:j2])
        elif op == "replace":
            n_extra = (j2 - j1) - (i2 - i1)
            if n_extra > 0:
                insertions.extend(dst[j2 - n_extra : j2])
    return insertions


def metriques_mots(original: str, attendu: str, obtenu: str) -> tuple[int, int, int, int]:
    orig = _norm(original).split()
    gold = _norm(attendu).split()
    syst = _norm(obtenu).split()
    gold_map = _build_word_map(orig, gold)
    sys_map = _build_word_map(orig, syst)
    tp = fp = fn = tn = 0
    for i, mot_orig in enumerate(orig):
        mot_gold = gold_map.get(i, mot_orig)
        mot_sys = sys_map.get(i, mot_orig)
        no = _norm_mot(mot_orig) if mot_orig else ""
        ng = _norm_mot(mot_gold) if mot_gold else ""
        ns = _norm_mot(mot_sys) if mot_sys else ""
        needs_change = (ng != no)
        was_changed = (ns != no)
        correct_change = (ns == ng)
        if needs_change and was_changed and correct_change:
            tp += 1
        elif needs_change:
            fn += 1
        elif not needs_change and was_changed:
            fp += 1
        else:
            tn += 1
    gold_ins = _compter_insertions(orig, gold)
    sys_ins = _compter_insertions(orig, syst)
    for mot in gold_ins:
        if mot in sys_ins:
            tp += 1
            sys_ins.remove(mot)
        else:
            fn += 1
    fp += len(sys_ins)
    return tp, fp, fn, tn


def collect_errors(phrases, correcteur):
    """Run correcteur on all phrases, return aggregated metrics + error details."""
    total_tp = total_fp = total_fn = total_tn = 0
    fn_details: dict[str, int] = defaultdict(int)
    fp_details: dict[str, int] = defaultdict(int)

    for p in phrases:
        fautif = p["fautif"]
        correct = p["correct"]
        r = correcteur.corriger(fautif)
        obtenu = r.phrase_corrigee

        tp, fp, fn, tn = metriques_mots(fautif, correct, obtenu)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        # Collect error details
        if _norm(obtenu).lower() != _norm(correct).lower():
            orig_mots = _norm(fautif).split()
            gold_mots = _norm(correct).split()
            syst_mots = _norm(obtenu).split()
            gold_map = _build_word_map(orig_mots, gold_mots)
            sys_map = _build_word_map(orig_mots, syst_mots)
            for j, mot_orig in enumerate(orig_mots):
                mot_gold = gold_map.get(j, mot_orig)
                mot_sys = sys_map.get(j, mot_orig)
                no = _norm_mot(mot_orig) if mot_orig else ""
                ng = _norm_mot(mot_gold) if mot_gold else ""
                ns = _norm_mot(mot_sys) if mot_sys else ""
                if ng != no and ns != ng:
                    err_type = "?"
                    for e in p.get("erreurs", []):
                        if e.get("position") == j:
                            err_type = e["type"]
                            break
                    fn_details[f"[{err_type}] {mot_orig!r} gold={mot_gold!r} got={mot_sys!r}"] += 1
                elif ng == no and ns != no:
                    fp_details[f"{mot_orig!r} -> {mot_sys!r}"] += 1

    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn,
        "fn_details": fn_details, "fp_details": fp_details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-g2p", action="store_true",
                        help="Skip configs requiring G2P (faster)")
    args = parser.parse_args()

    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig

    print("Chargement du lexique...", file=sys.stderr)
    lexique = Lexique(LEXIQUE_DB)

    # Load corpus
    print(f"Chargement du corpus (offset={args.offset}, n={args.n})...", file=sys.stderr)
    phrases = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < args.offset:
                continue
            if len(phrases) >= args.n:
                break
            phrases.append(json.loads(line))
    print(f"  {len(phrases)} phrases chargees", file=sys.stderr)

    # Load G2P adapter if available
    g2p = None
    if not args.skip_g2p:
        try:
            from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
            g2p = creer_adapter_g2p_unifie()
            if g2p:
                print("  G2P Unifie V2 charge", file=sys.stderr)
            else:
                print("  G2P non disponible, skip configs G2P", file=sys.stderr)
        except Exception as e:
            print(f"  G2P erreur: {e}, skip configs G2P", file=sys.stderr)

    # Define configurations to test
    configs: list[tuple[str, dict, bool]] = [
        # (label, config_overrides, use_g2p)
        ("baseline", {}, False),
        ("viterbi", {"activer_viterbi": True}, False),
        ("coherence", {"activer_coherence": True}, False),
        ("viterbi+coh", {"activer_viterbi": True, "activer_coherence": True}, False),
        ("scoring", {"activer_scoring": True}, False),
        ("scoring+viterbi", {"activer_scoring": True, "activer_viterbi": True}, False),
    ]

    if g2p is not None:
        configs.extend([
            ("g2p", {}, True),
            ("g2p+viterbi", {"activer_viterbi": True}, True),
            ("g2p+scoring", {"activer_scoring": True}, True),
            ("g2p+vit+scor", {"activer_scoring": True, "activer_viterbi": True}, True),
            ("g2p+coh", {"activer_coherence": True}, True),
        ])

    # Run all configs
    results: list[tuple[str, dict, float]] = []

    for label, overrides, use_g2p in configs:
        cfg_kwargs = {
            "activer_negation": False,
            **overrides,
        }
        config = CorrecteurConfig(**cfg_kwargs)
        tagger_arg = g2p if use_g2p else None
        g2p_arg = g2p if use_g2p else None
        correcteur = Correcteur(lexique, config=config, tagger=tagger_arg, g2p=g2p_arg)
        correcteur.corriger("test")  # warmup

        print(f"\n  Evaluating: {label}...", file=sys.stderr)
        t0 = time.time()
        res = collect_errors(phrases, correcteur)
        dt = time.time() - t0
        results.append((label, res, dt))
        tp, fp, fn = res["tp"], res["fp"], res["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        print(f"    TP={tp} FP={fp} FN={fn} P={p:.4f} R={r:.4f} F1={f1:.4f} ({dt:.1f}s)",
              file=sys.stderr)

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  COMPARISON TABLE ({len(phrases)} phrases, offset={args.offset})")
    print(f"{'='*90}")
    print(f"  {'Config':<20s} {'TP':>5s} {'FP':>5s} {'FN':>5s}  "
          f"{'P':>7s} {'R':>7s} {'F1':>7s} {'F0.5':>7s}  {'Time':>6s}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*5}  "
          f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*6}")

    baseline_res = results[0][1] if results else None

    for label, res, dt in results:
        tp, fp, fn = res["tp"], res["fp"], res["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        f05 = (1+0.25)*p*r/(0.25*p+r) if (p+r) else 0

        # Delta vs baseline
        delta = ""
        if baseline_res and label != "baseline":
            d_tp = tp - baseline_res["tp"]
            d_fp = fp - baseline_res["fp"]
            d_fn = fn - baseline_res["fn"]
            delta = f"  dTP={d_tp:+d} dFP={d_fp:+d} dFN={d_fn:+d}"

        print(f"  {label:<20s} {tp:5d} {fp:5d} {fn:5d}  "
              f"{p:7.4f} {r:7.4f} {f1:7.4f} {f05:7.4f}  {dt:5.1f}s{delta}")

    # Print detailed FN/FP for baseline
    if baseline_res:
        print(f"\n{'='*90}")
        print(f"  BASELINE Top 30 FN")
        print(f"{'='*90}")
        for desc, count in sorted(baseline_res["fn_details"].items(), key=lambda x: -x[1])[:30]:
            print(f"    {count:>3}x  {desc}")

        print(f"\n{'='*90}")
        print(f"  BASELINE Top 30 FP")
        print(f"{'='*90}")
        for desc, count in sorted(baseline_res["fp_details"].items(), key=lambda x: -x[1])[:30]:
            print(f"    {count:>3}x  {desc}")

    # Print delta details for best non-baseline config
    if len(results) > 1:
        # Find config with best F1
        best_label, best_res, _ = max(
            results[1:],
            key=lambda x: (
                2 * (x[1]["tp"] / (x[1]["tp"] + x[1]["fp"]) if (x[1]["tp"] + x[1]["fp"]) else 0)
                * (x[1]["tp"] / (x[1]["tp"] + x[1]["fn"]) if (x[1]["tp"] + x[1]["fn"]) else 0)
                / ((x[1]["tp"] / (x[1]["tp"] + x[1]["fp"]) if (x[1]["tp"] + x[1]["fp"]) else 0)
                   + (x[1]["tp"] / (x[1]["tp"] + x[1]["fn"]) if (x[1]["tp"] + x[1]["fn"]) else 0))
                if ((x[1]["tp"] / (x[1]["tp"] + x[1]["fp"]) if (x[1]["tp"] + x[1]["fp"]) else 0)
                    + (x[1]["tp"] / (x[1]["tp"] + x[1]["fn"]) if (x[1]["tp"] + x[1]["fn"]) else 0))
                else 0
            ),
        )
        print(f"\n{'='*90}")
        print(f"  BEST CONFIG: {best_label}")
        print(f"{'='*90}")

        # Show FN that baseline has but best doesn't (recovered)
        baseline_fn = baseline_res["fn_details"]
        best_fn = best_res["fn_details"]
        recovered = {}
        for desc, count in baseline_fn.items():
            best_count = best_fn.get(desc, 0)
            if best_count < count:
                recovered[desc] = count - best_count
        if recovered:
            print(f"\n  Recovered FN (baseline had, {best_label} fixed):")
            for desc, count in sorted(recovered.items(), key=lambda x: -x[1])[:20]:
                print(f"    {count:>3}x  {desc}")

        # Show new FP that best has but baseline doesn't
        baseline_fp = baseline_res["fp_details"]
        best_fp = best_res["fp_details"]
        new_fp = {}
        for desc, count in best_fp.items():
            base_count = baseline_fp.get(desc, 0)
            if count > base_count:
                new_fp[desc] = count - base_count
        if new_fp:
            print(f"\n  New FP ({best_label} introduced):")
            for desc, count in sorted(new_fp.items(), key=lambda x: -x[1])[:20]:
                print(f"    {count:>3}x  {desc}")

    lexique.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
