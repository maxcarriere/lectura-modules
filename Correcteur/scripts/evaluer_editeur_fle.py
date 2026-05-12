#!/usr/bin/env python3
"""Evaluation du BiLSTM edit tagger sur le corpus FLE (corpus_10000.jsonl).

Compare 4 configurations :
  1. BiLSTM seul (tous les tags)
  2. BiLSTM homophones seulement (HOMO_* tags)
  3. Regles seules (pipeline existant)
  4. Hybride : regles + BiLSTM pour les homophones

Metriques : Precision, Recall, F1 par categorie (HOMO, ACC, CONJ, PP).

Usage :
    python scripts/evaluer_editeur_fle.py [--n 2000] [--offset 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(_PROJECT_ROOT), "Lexique", "src"))

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
CORPUS_PATH = os.path.join(_PROJECT_ROOT, "data", "corpus", "corpus_10000.jsonl")
WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "src", "lectura_correcteur", "data", "editeur_weights.json.gz")
VOCAB_PATH = os.path.join(_PROJECT_ROOT, "src", "lectura_correcteur", "data", "editeur_vocab.json")


def _tokenize_simple(phrase: str) -> list[str]:
    """Tokenise par espaces (simplifie pour evaluation)."""
    return phrase.strip().split()


def _align_corrections(
    fautif_tokens: list[str],
    correct_tokens: list[str],
    erreurs: list[dict],
) -> dict[int, dict]:
    """Aligne les erreurs du corpus avec les positions dans les tokens fautifs.

    Returns:
        Dict position_fautif -> {original, perturbe, type}
    """
    corrections = {}
    for err in erreurs:
        pos = err["position"]
        if 0 <= pos < len(fautif_tokens):
            corrections[pos] = err
    return corrections


def evaluer_bilstm(
    editeur,
    tagger,
    entries: list[dict],
    homo_only: bool = False,
) -> dict:
    """Evalue le BiLSTM sur les phrases du corpus.

    Args:
        editeur: EditeurNumpy instance
        tagger: LexiqueTagger instance
        entries: list of corpus entries
        homo_only: si True, ne compter que les tags HOMO_*

    Returns:
        dict avec TP, FP, FN par categorie
    """
    from lectura_correcteur._tags import KEEP, TAG2IDX, _TAG_TO_HOMO

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_tp = 0
    total_fp = 0
    total_fn = 0
    details_fp = []
    details_fn = []

    for entry in entries:
        fautif_tokens = _tokenize_simple(entry["fautif"])
        correct_tokens = _tokenize_simple(entry["correct"])
        gold = _align_corrections(fautif_tokens, correct_tokens, entry["erreurs"])

        # Tagger + predire
        morpho = tagger.tag_words(fautif_tokens)
        tags_scores = editeur.predire_tags_avec_scores(fautif_tokens, morpho)

        for i, (model_tag, score) in enumerate(tags_scores):
            canon_tag = model_tag if model_tag in TAG2IDX else KEEP
            is_homo = canon_tag.startswith("HOMO_")

            if homo_only and not is_homo and canon_tag != KEEP:
                # Ignorer les tags non-homo en mode homo_only
                canon_tag = KEEP

            has_gold = i in gold
            predicted_change = canon_tag != KEEP

            if has_gold:
                err = gold[i]
                cat = err["type"]

                if predicted_change:
                    # Verifier si la correction est correcte
                    if is_homo and canon_tag in _TAG_TO_HOMO:
                        predicted_form = _TAG_TO_HOMO[canon_tag]
                        if predicted_form.lower() == err["original"].lower():
                            total_tp += 1
                            stats[cat]["tp"] += 1
                        else:
                            total_fp += 1
                            stats[cat]["fp"] += 1
                            total_fn += 1
                            stats[cat]["fn"] += 1
                            details_fp.append(
                                f"  FP+FN {cat}: '{fautif_tokens[i]}' -> pred='{predicted_form}' "
                                f"(gold='{err['original']}') [{model_tag} {score:.2f}]"
                            )
                    else:
                        # Non-homo tag: compter comme TP si le type correspond
                        total_tp += 1
                        stats[cat]["tp"] += 1
                else:
                    # Pas de prediction mais erreur presente -> FN
                    if not homo_only or cat == "HOMO":
                        total_fn += 1
                        stats[cat]["fn"] += 1
                        if cat == "HOMO":
                            details_fn.append(
                                f"  FN {cat}: '{fautif_tokens[i]}' -> gold='{err['original']}'"
                            )
            elif predicted_change:
                # Prediction mais pas d'erreur -> FP
                total_fp += 1
                cat = "HOMO" if is_homo else "AUTRE"
                stats[cat]["fp"] += 1
                if len(details_fp) < 50:
                    context = " ".join(fautif_tokens[max(0, i-2):i+3])
                    details_fp.append(
                        f"  FP {cat}: '{fautif_tokens[i]}' [{model_tag} {score:.2f}] ctx='{context}'"
                    )

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": p, "recall": r, "f1": f1,
        "par_cat": dict(stats),
        "details_fp": details_fp[:30],
        "details_fn": details_fn[:30],
    }


def evaluer_regles(correcteur, entries: list[dict]) -> dict:
    """Evalue le pipeline de regles existant."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for entry in entries:
        fautif = entry["fautif"]
        correct_tokens = _tokenize_simple(entry["correct"])
        fautif_tokens = _tokenize_simple(fautif)
        gold = _align_corrections(fautif_tokens, correct_tokens, entry["erreurs"])

        result = correcteur.corriger(fautif)
        pred_tokens = _tokenize_simple(result.phrase_corrigee)

        # Aligner les predictions avec les gold
        for i in range(min(len(fautif_tokens), len(pred_tokens))):
            has_gold = i in gold
            predicted_change = (
                i < len(pred_tokens)
                and pred_tokens[i].lower() != fautif_tokens[i].lower()
            )

            if has_gold:
                err = gold[i]
                cat = err["type"]
                if predicted_change:
                    if pred_tokens[i].lower() == err["original"].lower():
                        total_tp += 1
                        stats[cat]["tp"] += 1
                    else:
                        total_fp += 1
                        stats[cat]["fp"] += 1
                        total_fn += 1
                        stats[cat]["fn"] += 1
                else:
                    total_fn += 1
                    stats[cat]["fn"] += 1
            elif predicted_change:
                total_fp += 1

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": p, "recall": r, "f1": f1,
        "par_cat": dict(stats),
    }


def _print_results(label: str, res: dict) -> None:
    """Affiche les resultats d'evaluation."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  TP={res['tp']}  FP={res['fp']}  FN={res['fn']}")
    print(f"  P={res['precision']:.3f}  R={res['recall']:.3f}  F1={res['f1']:.3f}")

    if res.get("par_cat"):
        print(f"\n  Par categorie :")
        for cat in sorted(res["par_cat"]):
            s = res["par_cat"][cat]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            print(f"    {cat:8s}: TP={tp:3d}  FP={fp:3d}  FN={fn:3d}  "
                  f"P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

    if res.get("details_fp"):
        print(f"\n  Exemples FP (max 20) :")
        for d in res["details_fp"][:20]:
            print(d)

    if res.get("details_fn"):
        print(f"\n  Exemples FN HOMO (max 20) :")
        for d in res["details_fn"][:20]:
            print(d)


def main():
    parser = argparse.ArgumentParser(description="Evaluation BiLSTM vs regles sur corpus FLE")
    parser.add_argument("--n", type=int, default=2000, help="Nombre de phrases")
    parser.add_argument("--offset", type=int, default=0, help="Offset dans le corpus")
    parser.add_argument("--skip-rules", action="store_true", help="Ne pas evaluer les regles")
    args = parser.parse_args()

    from lectura_lexique import Lexique
    from lectura_correcteur._editeur_numpy import EditeurNumpy
    from lectura_correcteur._tagger_lexique import LexiqueTagger
    from lectura_correcteur._utils import LexiqueNormalise
    from lectura_correcteur._config import CorrecteurConfig
    from lectura_correcteur.correcteur import Correcteur

    print("Chargement du lexique...")
    lex = Lexique(LEXIQUE_DB)
    lex_norm = LexiqueNormalise(lex)

    print("Chargement du tagger...")
    tagger = LexiqueTagger(lex_norm)

    print("Chargement du BiLSTM editeur...")
    editeur = EditeurNumpy(WEIGHTS_PATH, VOCAB_PATH)

    # Charger le corpus
    print(f"Chargement du corpus ({args.n} phrases depuis offset {args.offset})...")
    entries = []
    with open(CORPUS_PATH) as f:
        for i, line in enumerate(f):
            if i < args.offset:
                continue
            if len(entries) >= args.n:
                break
            entry = json.loads(line)
            if entry.get("erreurs"):
                entries.append(entry)

    print(f"  {len(entries)} phrases avec erreurs chargees")
    total_erreurs = sum(len(e["erreurs"]) for e in entries)
    print(f"  {total_erreurs} erreurs totales")

    # Distribution par type
    type_counts = defaultdict(int)
    for entry in entries:
        for err in entry["erreurs"]:
            type_counts[err["type"]] += 1
    print("  Distribution :", dict(type_counts))

    # ---- 1. BiLSTM tous tags ----
    print("\n>>> Evaluation BiLSTM (tous tags)...")
    t0 = time.time()
    res_bilstm = evaluer_bilstm(editeur, tagger, entries, homo_only=False)
    t_bilstm = time.time() - t0
    _print_results(f"BiLSTM tous tags ({t_bilstm:.1f}s)", res_bilstm)

    # ---- 2. BiLSTM homophones seulement ----
    print("\n>>> Evaluation BiLSTM (homophones seulement)...")
    t0 = time.time()
    res_homo = evaluer_bilstm(editeur, tagger, entries, homo_only=True)
    t_homo = time.time() - t0
    _print_results(f"BiLSTM homophones seulement ({t_homo:.1f}s)", res_homo)

    # ---- 3. Regles seules ----
    if not args.skip_rules:
        print("\n>>> Evaluation regles seules...")
        config = CorrecteurConfig()
        correcteur = Correcteur(lex, config=config)
        t0 = time.time()
        res_regles = evaluer_regles(correcteur, entries)
        t_regles = time.time() - t0
        _print_results(f"Regles seules ({t_regles:.1f}s)", res_regles)

    # ---- 4. Hybride : regles + BiLSTM homophones ----
    if not args.skip_rules:
        for seuil in (0.90, 0.95, 0.99):
            print(f"\n>>> Evaluation hybride (regles + BiLSTM, seuil={seuil})...")
            config_hyb = CorrecteurConfig(
                activer_editeur_homophones=True,
                seuil_editeur=seuil,
            )
            correcteur_hyb = Correcteur(lex, config=config_hyb)
            t0 = time.time()
            res_hyb = evaluer_regles(correcteur_hyb, entries)
            t_hyb = time.time() - t0
            _print_results(f"Hybride seuil={seuil} ({t_hyb:.1f}s)", res_hyb)
            locals()[f"res_hyb_{int(seuil*100)}"] = res_hyb

    # ---- Resume ----
    print(f"\n{'='*60}")
    print(f"  RESUME ({len(entries)} phrases, {total_erreurs} erreurs)")
    print(f"{'='*60}")
    print(f"  {'Config':30s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")

    for label, res in [
        ("BiLSTM tous tags", res_bilstm),
        ("BiLSTM HOMO seulement", res_homo),
    ]:
        print(f"  {label:30s} {res['precision']:6.3f} {res['recall']:6.3f} "
              f"{res['f1']:6.3f} {res['tp']:5d} {res['fp']:5d} {res['fn']:5d}")

    if not args.skip_rules:
        print(f"  {'Regles seules':30s} {res_regles['precision']:6.3f} {res_regles['recall']:6.3f} "
              f"{res_regles['f1']:6.3f} {res_regles['tp']:5d} {res_regles['fp']:5d} {res_regles['fn']:5d}")
        for seuil in (0.90, 0.95, 0.99):
            key = f"res_hyb_{int(seuil*100)}"
            if key in locals():
                res_h = locals()[key]
                label = f"Hybride seuil={seuil}"
                print(f"  {label:30s} {res_h['precision']:6.3f} {res_h['recall']:6.3f} "
                      f"{res_h['f1']:6.3f} {res_h['tp']:5d} {res_h['fp']:5d} {res_h['fn']:5d}")

    lex.close()
    print("\nTermine.")


if __name__ == "__main__":
    main()
