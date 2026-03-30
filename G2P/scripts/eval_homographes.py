#!/usr/bin/env python3
"""Évalue l'impact des corrections G2P POS-aware (homographes) sur le test set.

Compare 3 modes :
  1. Modèle brut (sans corrections)
  2. Corrections plates (table seule, sans homographes)
  3. Corrections POS-aware (homographes + table)

Usage :
    python scripts/eval_homographes.py
    python scripts/eval_homographes.py --backend onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from lectura_nlp.tokeniseur import tokeniser
from lectura_nlp.posttraitement import (
    charger_corrections,
    charger_homographes,
    corriger_g2p,
    appliquer_regles_g2p,
)
from lectura_nlp.posttraitement import (
    _CORRECTIONS_TABLE,
    _HOMOGRAPHES_TABLE,
)


def create_engine(backend: str, models_dir: Path):
    vocab_path = models_dir / "unifie_vocab.json"
    weights_path = models_dir / "unifie_weights.json"
    onnx_path = models_dir / "unifie_int8.onnx"
    if backend == "onnx":
        from lectura_nlp.inference_onnx import OnnxInferenceEngine
        return OnnxInferenceEngine(onnx_path, vocab_path)
    elif backend == "numpy":
        from lectura_nlp.inference_numpy import NumpyInferenceEngine
        return NumpyInferenceEngine(weights_path, vocab_path)
    elif backend == "pure":
        from lectura_nlp.inference_pure import PureInferenceEngine
        return PureInferenceEngine(weights_path, vocab_path)
    else:
        raise ValueError(f"Backend inconnu : {backend}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="numpy", choices=["onnx", "numpy", "pure"])
    parser.add_argument("--modeles", type=Path, default=_ROOT / "modeles")
    parser.add_argument(
        "--test-data", type=Path,
        default=Path("/data/work/projets/lectura/workspace/Outils_NLP/Modeles_complets/P2G/entrainement/donnees/sentences_test.json"),
    )
    args = parser.parse_args()

    # Charger le test set
    print(f"Chargement des données : {args.test_data}")
    with open(args.test_data, encoding="utf-8") as f:
        sentences = json.load(f)
    print(f"  {len(sentences)} phrases")

    # Charger le modèle
    print(f"Chargement du modèle ({args.backend})...")
    engine = create_engine(args.backend, args.modeles)

    # Charger les corrections
    corrections_path = args.modeles / "g2p_corrections_unifie.json"
    homographes_path = args.modeles / "homographes.json"

    # ── Évaluation : 3 modes ────────────────────────────────────────

    modes = [
        ("Brut (modèle seul)", False, False),
        ("+ corrections plates", True, False),
        ("+ corrections + homographes", True, True),
    ]

    for mode_name, use_corrections, use_homographes in modes:
        # Reset les tables
        import lectura_nlp.posttraitement as pt
        pt._CORRECTIONS_TABLE = None
        pt._HOMOGRAPHES_TABLE = None

        if use_corrections:
            charger_corrections(corrections_path)
        if use_homographes:
            charger_homographes(homographes_path)

        total_words = 0
        correct_words = 0
        # Compteurs pour les homographes spécifiquement
        homo_total = 0
        homo_correct = 0
        # Exemples d'améliorations
        improvements = []
        regressions = []

        for sent in sentences:
            tokens_gold = sent["tokens"]
            forms = [t["form"] for t in tokens_gold]

            # Inférence
            tokens = tokeniser(sent["text"])
            if not tokens:
                continue
            result = engine.analyser(tokens)

            pred_g2p = result.get("g2p", [])
            pred_pos = result.get("pos", [])

            # Aligner tokens modèle ↔ tokens gold (par forme)
            # Simple: itérer sur les gold tokens et chercher dans les preds
            pred_idx = 0
            for tg in tokens_gold:
                form = tg["form"]
                gold_phone = tg["phone"]
                gold_pos = tg["pos_tag"]

                if not gold_phone or gold_phone == "_":
                    continue

                # Trouver le token prédit correspondant
                while pred_idx < len(tokens) and tokens[pred_idx].lower() != form.lower():
                    pred_idx += 1

                if pred_idx >= len(tokens) or pred_idx >= len(pred_g2p):
                    continue

                raw_pred = pred_g2p[pred_idx]

                if use_homographes:
                    pos_tag = pred_pos[pred_idx] if pred_idx < len(pred_pos) else None
                    corrected = corriger_g2p(form, raw_pred, pos_tag)
                elif use_corrections:
                    corrected = corriger_g2p(form, raw_pred)
                else:
                    corrected = raw_pred

                total_words += 1
                if corrected == gold_phone:
                    correct_words += 1

                # Tracker les homographes
                if pt._HOMOGRAPHES_TABLE and form.lower() in pt._HOMOGRAPHES_TABLE:
                    homo_total += 1
                    if corrected == gold_phone:
                        homo_correct += 1

                pred_idx += 1

        acc = correct_words / total_words if total_words else 0
        homo_acc = homo_correct / homo_total if homo_total else 0

        print(f"\n{'─' * 50}")
        print(f"Mode : {mode_name}")
        print(f"  Word Accuracy : {acc:.2%}  ({correct_words}/{total_words})")
        if homo_total > 0:
            print(f"  Homographes   : {homo_acc:.2%}  ({homo_correct}/{homo_total})")

    # ── Analyse détaillée : impact par mot ──────────────────────────

    print(f"\n{'═' * 50}")
    print("ANALYSE DÉTAILLÉE : mots impactés par les homographes")
    print(f"{'═' * 50}")

    # Reset et charger tout
    import lectura_nlp.posttraitement as pt

    # Mode "corrections seules"
    pt._CORRECTIONS_TABLE = None
    pt._HOMOGRAPHES_TABLE = None
    charger_corrections(corrections_path)

    results_flat: dict[str, list] = {}  # (form, gold_pos) → [gold_phone, pred_flat]

    for sent in sentences:
        tokens_gold = sent["tokens"]
        tokens = tokeniser(sent["text"])
        if not tokens:
            continue
        result = engine.analyser(tokens)
        pred_g2p = result.get("g2p", [])
        pred_pos = result.get("pos", [])

        pred_idx = 0
        for tg in tokens_gold:
            form = tg["form"]
            gold_phone = tg["phone"]
            gold_pos = tg["pos_tag"]
            if not gold_phone or gold_phone == "_":
                continue
            while pred_idx < len(tokens) and tokens[pred_idx].lower() != form.lower():
                pred_idx += 1
            if pred_idx >= len(tokens) or pred_idx >= len(pred_g2p):
                continue

            raw_pred = pred_g2p[pred_idx]
            corrected_flat = corriger_g2p(form, raw_pred)
            pred_pos_tag = pred_pos[pred_idx] if pred_idx < len(pred_pos) else None

            key = (form.lower(), gold_pos, gold_phone, pred_pos_tag)
            results_flat[id(tg)] = {
                "form": form, "gold_pos": gold_pos, "gold_phone": gold_phone,
                "pred_pos": pred_pos_tag, "raw": raw_pred, "flat": corrected_flat,
            }
            pred_idx += 1

    # Mode "corrections + homographes"
    pt._CORRECTIONS_TABLE = None
    pt._HOMOGRAPHES_TABLE = None
    charger_corrections(corrections_path)
    charger_homographes(homographes_path)

    improved = []
    regressed = []
    changed = []

    for sent in sentences:
        tokens_gold = sent["tokens"]
        tokens = tokeniser(sent["text"])
        if not tokens:
            continue
        result = engine.analyser(tokens)
        pred_g2p = result.get("g2p", [])
        pred_pos = result.get("pos", [])

        pred_idx = 0
        for tg in tokens_gold:
            form = tg["form"]
            gold_phone = tg["phone"]
            if not gold_phone or gold_phone == "_":
                continue
            while pred_idx < len(tokens) and tokens[pred_idx].lower() != form.lower():
                pred_idx += 1
            if pred_idx >= len(tokens) or pred_idx >= len(pred_g2p):
                continue

            raw_pred = pred_g2p[pred_idx]
            pred_pos_tag = pred_pos[pred_idx] if pred_idx < len(pred_pos) else None
            corrected_homo = corriger_g2p(form, raw_pred, pred_pos_tag)

            info = results_flat.get(id(tg))
            if info:
                corrected_flat = info["flat"]
                if corrected_flat != corrected_homo:
                    entry = {
                        "form": form, "gold_pos": info["gold_pos"],
                        "gold_phone": gold_phone, "pred_pos": pred_pos_tag,
                        "raw": raw_pred, "flat": corrected_flat,
                        "homo": corrected_homo,
                    }
                    changed.append(entry)
                    if corrected_homo == gold_phone and corrected_flat != gold_phone:
                        improved.append(entry)
                    elif corrected_homo != gold_phone and corrected_flat == gold_phone:
                        regressed.append(entry)

            pred_idx += 1

    print(f"\nMots dont la correction change : {len(changed)}")
    print(f"  Améliorés  : {len(improved)}")
    print(f"  Régressés  : {len(regressed)}")
    print(f"  Neutres    : {len(changed) - len(improved) - len(regressed)}")

    if improved:
        print(f"\n  Améliorations (max 20) :")
        for e in improved[:20]:
            print(f"    {e['form']:15s} gold_pos={e['gold_pos']:10s} pred_pos={e['pred_pos'] or '?':10s} "
                  f"gold={e['gold_phone']:12s} flat={e['flat']:12s} → homo={e['homo']:12s}")

    if regressed:
        print(f"\n  Régressions (max 20) :")
        for e in regressed[:20]:
            print(f"    {e['form']:15s} gold_pos={e['gold_pos']:10s} pred_pos={e['pred_pos'] or '?':10s} "
                  f"gold={e['gold_phone']:12s} flat={e['flat']:12s} → homo={e['homo']:12s}")


if __name__ == "__main__":
    main()
