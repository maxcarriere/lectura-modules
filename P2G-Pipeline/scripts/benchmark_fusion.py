#!/usr/bin/env python3
"""Benchmark de la fusion de mots composes sur le dev set P2G.

Mesure l'impact de corriger_phrase_pipeline (avec fusion) vs corriger_phrase_v3
(sans fusion) sur le dev set.

Usage:
    python scripts/benchmark_fusion.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Graphemiseur" / "src"))


def main():
    # ── Charger le dev set ──
    dev_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "Corpus" / "Kit-G2P-P2G" / "corpus" / "phrases" / "sentences_dev.json"
    )
    print(f"Chargement dev set : {dev_path}")
    with open(dev_path, encoding="utf-8") as f:
        sentences = json.load(f)
    n_phrases = len(sentences)
    n_mots = sum(len(s["tokens"]) for s in sentences)
    print(f"  {n_phrases} phrases, {n_mots} mots")

    # ── Charger le moteur ──
    print("Chargement moteur P2G V6...")
    from lectura_graphemiseur import creer_engine
    engine = creer_engine(mode="onnx")
    print(f"  Modele V6 charge")

    # ── Construire le lexique ──
    lexique_set = None
    freq_map = None
    lexique_index = None
    if engine.phone_lexicon is not None:
        lexique_set = frozenset(
            engine.phone_lexicon.phone_to_best[p][0].lower()
            for p in engine.phone_lexicon.phone_to_best
        )
        from lectura_graphemiseur.posttraitement import _build_prefix_index
        freq_map = {
            engine.phone_lexicon.phone_to_best[p][0].lower():
            engine.phone_lexicon.phone_to_best[p][1]
            for p in engine.phone_lexicon.phone_to_best
        }
        lexique_index = _build_prefix_index(lexique_set, freq_map, min_freq=0.1)

    from lectura_graphemiseur.posttraitement import corriger_phrase_v3
    from lectura_graphemiseur.utils.p2g_labels import reconstruct_ortho
    from lectura_p2g import corriger_phrase_pipeline

    # ── Evaluation ──
    print(f"\nEvaluation en cours...")
    t0 = time.time()

    total_words = 0
    v3_correct = 0
    pipeline_correct = 0

    # Tracking fusions
    fusions_applied = 0
    fusion_fixes = 0      # v3 faux → pipeline correct (grace a la fusion)
    fusion_casses = 0     # v3 correct → pipeline faux (regression)
    fusion_neutral = 0    # les deux faux ou les deux corrects

    fusion_details: list[dict] = []

    for s_idx, sent in enumerate(sentences):
        tokens = sent["tokens"]
        if not tokens:
            continue
        tokens = [t for t in tokens if t["phone"].strip()]
        if not tokens:
            continue

        ipa_words = [t["phone"] for t in tokens]
        gold_words = []
        for t in tokens:
            if t.get("formule_type"):
                gold_words.append(t.get("display_fr", t["form"]))
            else:
                gold_words.append(t["form"])
        n_words = len(ipa_words)

        # Inference
        char_ids, word_starts, word_ends, lex_features, chars = \
            engine._encode_sentence(ipa_words, use_lex=True)

        lex_cand_features = lex_cand_mask = None
        candidates_list = None
        if engine.has_lex_select and engine.phone_lexicon is not None:
            lex_cand_features, lex_cand_mask, candidates_list, _ = \
                engine._build_lex_candidates(ipa_words)

        output_dict = engine._run_session(
            char_ids, word_starts, word_ends, lex_features,
            lex_cand_features, lex_cand_mask,
        )

        # P2G brut
        p2g_logits = output_dict["p2g_logits"]
        p2g_preds = p2g_logits[0].argmax(axis=-1)
        ortho_raw = []
        for w in range(n_words):
            ws = word_starts[0, w]
            we = word_ends[0, w]
            word_labels = [engine.idx2p2g.get(int(p2g_preds[i]), "_CONT")
                           for i in range(ws, we + 1)]
            ortho_raw.append(reconstruct_ortho(word_labels))

        # Lex_select
        ortho_lex = list(ortho_raw)
        if candidates_list is not None:
            ortho_lex = engine._apply_lex_select(
                output_dict, word_starts, n_words, candidates_list, ortho_lex,
                ipa_words=ipa_words,
            )

        # POS
        pos_pred = []
        if "pos_logits" in output_dict:
            pos_preds_arr = output_dict["pos_logits"][0].argmax(axis=-1)
            for w in range(n_words):
                pos_pred.append(engine.idx2pos.get(int(pos_preds_arr[w]), "NOM"))

        # Morpho
        morpho_pred: dict[str, list[str]] = {}
        for feat_name, idx2label in engine.idx2morpho.items():
            key = f"morpho_{feat_name}_logits"
            if key in output_dict:
                feat_preds = output_dict[key][0].argmax(axis=-1)
                morpho_pred[feat_name] = [
                    idx2label.get(int(feat_preds[w]), "_") for w in range(n_words)
                ]

        morpho_features: dict[str, list[str]] = {}
        for feat_name in ("Number", "Gender", "VerbForm", "Mood", "Tense", "Person"):
            if feat_name in morpho_pred:
                morpho_features[feat_name] = morpho_pred[feat_name]
            else:
                morpho_features[feat_name] = ["_"] * n_words

        # Lex candidates avec confiance
        lex_candidates = None
        if candidates_list is not None and "lex_select_logits" in output_dict:
            lex_logits = output_dict["lex_select_logits"][0]
            lex_candidates = []
            for w in range(n_words):
                cands = candidates_list[w]
                if not cands:
                    lex_candidates.append([])
                    continue
                logits_w = lex_logits[w, :len(cands)]
                exp_l = np.exp(logits_w - logits_w.max())
                probs = exp_l / exp_l.sum()
                word_cands = [
                    (c.get("ortho", ""), float(probs[k]))
                    for k, c in enumerate(cands)
                    if c.get("ortho", "") and not c["ortho"].startswith("-")
                ]
                word_cands.sort(key=lambda x: -x[1])
                lex_candidates.append(word_cands)

        # ── corriger_phrase_v3 (baseline) ──
        ortho_v3 = corriger_phrase_v3(
            list(ortho_lex), pos_pred, morpho_features,
            lexique=lexique_set,
            lexique_index=lexique_index,
            freq_map=freq_map,
            lex_candidates=lex_candidates,
        )

        # ── corriger_phrase_pipeline (avec fusion) ──
        ortho_pipeline = corriger_phrase_pipeline(
            list(ortho_lex), pos_pred, morpho_features,
            ipa_words=ipa_words,
            phone_lexicon=engine.phone_lexicon,
            lexique=lexique_set,
            lexique_index=lexique_index,
            freq_map=freq_map,
            lex_candidates=lex_candidates,
        )

        # ── Comparer ──
        for w_idx in range(n_words):
            gold = gold_words[w_idx].lower()
            v3 = ortho_v3[w_idx].lower()
            pip = ortho_pipeline[w_idx].lower()

            total_words += 1
            if v3 == gold:
                v3_correct += 1
            if pip == gold:
                pipeline_correct += 1

            # Detecter les changements dus a la fusion/entites
            if v3 != pip:
                fusions_applied += 1
                v3_ok = (v3 == gold)
                pip_ok = (pip == gold)
                if pip_ok and not v3_ok:
                    fusion_fixes += 1
                    fusion_details.append({
                        "type": "fix",
                        "gold": gold_words[w_idx],
                        "v3": ortho_v3[w_idx],
                        "pipeline": ortho_pipeline[w_idx],
                        "ipa": ipa_words[w_idx] if w_idx < len(ipa_words) else "",
                    })
                elif v3_ok and not pip_ok:
                    fusion_casses += 1
                    fusion_details.append({
                        "type": "regression",
                        "gold": gold_words[w_idx],
                        "v3": ortho_v3[w_idx],
                        "pipeline": ortho_pipeline[w_idx],
                        "ipa": ipa_words[w_idx] if w_idx < len(ipa_words) else "",
                    })
                else:
                    fusion_neutral += 1

    dt = time.time() - t0

    # ── Resultats ──
    print(f"\n{'='*80}")
    print(f"Resultats benchmark fusion")
    print(f"{'='*80}")
    print(f"  Phrases : {n_phrases}")
    print(f"  Mots    : {total_words}")
    print(f"  Duree   : {dt:.1f}s ({total_words/dt:.0f} mots/s)")

    v3_acc = v3_correct / total_words * 100
    pip_acc = pipeline_correct / total_words * 100
    print(f"\n  Accuracy v3 (baseline) : {v3_correct}/{total_words} = {v3_acc:.3f}%")
    print(f"  Accuracy pipeline      : {pipeline_correct}/{total_words} = {pip_acc:.3f}%")
    print(f"  Delta                  : {pipeline_correct - v3_correct:+d} mots ({pip_acc - v3_acc:+.3f}%)")

    print(f"\n  Mots changes par pipeline : {fusions_applied}")
    print(f"    Corrections (fix)      : {fusion_fixes}")
    print(f"    Regressions (casse)    : {fusion_casses}")
    print(f"    Neutres (les deux faux): {fusion_neutral}")

    if fusion_fixes + fusion_casses > 0:
        prec = fusion_fixes / (fusion_fixes + fusion_casses) * 100
        print(f"    Precision corrections  : {prec:.1f}%")

    # Details
    if fusion_details:
        print(f"\n  Corrections ({fusion_fixes}):")
        for d in fusion_details:
            if d["type"] == "fix":
                print(f"    gold={d['gold']:20s}  v3={d['v3']:20s}  pipeline={d['pipeline']:20s}  ipa={d['ipa']}")

        print(f"\n  Regressions ({fusion_casses}):")
        for d in fusion_details:
            if d["type"] == "regression":
                print(f"    gold={d['gold']:20s}  v3={d['v3']:20s}  pipeline={d['pipeline']:20s}  ipa={d['ipa']}")


if __name__ == "__main__":
    main()
