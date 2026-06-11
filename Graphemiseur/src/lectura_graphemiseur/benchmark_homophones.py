"""Benchmark de la desambiguation des homophones par predictions du modele.

Mesure l'impact de :
- Step 1c etendu : homophones POS-level (ces/ses, ca/sa, son/sont, on/ont)
- Step 1d : homophones morpho-level (il/ils, au/aux, leur/leurs, etc.)

Compare la baseline (sans 1c etendu / sans 1d) au pipeline avec.

Usage :
    python -m lectura_graphemiseur.benchmark_homophones
    python -m lectura_graphemiseur.benchmark_homophones --details
    python -m lectura_graphemiseur.benchmark_homophones --export resultats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_DEV_SET_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "Corpus" / "Kit-G2P-P2G" / "corpus" / "phrases" / "sentences_dev.json"
)


def benchmark_homophones(
    dev_set_path: str | Path | None = None,
    show_details: bool = False,
    export_path: str | Path | None = None,
) -> dict:
    """Benchmark de la desambiguation des homophones."""

    # ── Charger le dev set ──
    if dev_set_path is None:
        dev_set_path = _DEV_SET_DEFAULT
    dev_set_path = Path(dev_set_path)
    if not dev_set_path.exists():
        print(f"ERREUR: dev set introuvable : {dev_set_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Chargement dev set : {dev_set_path}")
    with open(dev_set_path, encoding="utf-8") as f:
        sentences = json.load(f)
    n_phrases = len(sentences)
    n_mots_total = sum(len(s["tokens"]) for s in sentences)
    print(f"  {n_phrases} phrases, {n_mots_total} mots")

    # ── Charger le moteur P2G ──
    print("Chargement moteur P2G...")
    from lectura_graphemiseur import creer_engine
    engine = creer_engine(mode="onnx")
    print(f"  Modele charge (lex_select={engine.has_lex_select})")

    # ── Construire le lexique set ──
    lexique_set: frozenset[str] | None = None
    if engine.phone_lexicon is not None:
        lexique_set = frozenset(
            engine.phone_lexicon.phone_to_best[p][0].lower()
            for p in engine.phone_lexicon.phone_to_best
        )
        print(f"  Lexique set : {len(lexique_set)} formes")

    from lectura_graphemiseur.posttraitement import _build_prefix_index
    freq_map: dict[str, float] | None = None
    lexique_index = None
    if lexique_set is not None and engine.phone_lexicon is not None:
        freq_map = {
            engine.phone_lexicon.phone_to_best[p][0].lower():
            engine.phone_lexicon.phone_to_best[p][1]
            for p in engine.phone_lexicon.phone_to_best
        }
        lexique_index = _build_prefix_index(lexique_set, freq_map, min_freq=0.1)

    from lectura_graphemiseur.posttraitement import corriger_phrase_v3
    from lectura_graphemiseur._homophones import _HOMOPHONES_MORPHO

    # Paires POS-level ajoutees dans step 1c (en plus de a/a, ou/ou existants)
    _POS_PAIRS = {
        "ces": ("ADJ:pos", "ses"),
        "ses": ("ADJ:dem", "ces"),
        "ça":  ("ADJ:pos", "sa"),
        "sa":  ("PRO:dem", "ça"),
        "son": ("AUX", "sont"),
        "sont": ("ADJ:pos", "son"),
        "on":  ("AUX", "ont"),
        "ont": ("PRO:ind", "on"),
    }

    import numpy as np
    from lectura_graphemiseur.utils.p2g_labels import reconstruct_ortho

    # ── Evaluation ──
    print(f"\nEvaluation en cours...")
    t0 = time.time()

    total_words = 0
    baseline_correct = 0
    new_correct = 0

    # Par type
    fixes_1c: list[dict] = []   # fixes par step 1c etendu
    casses_1c: list[dict] = []  # casses par step 1c etendu
    fixes_1d: list[dict] = []   # fixes par step 1d morpho
    casses_1d: list[dict] = []  # casses par step 1d morpho

    fixes_par_paire: dict[str, int] = defaultdict(int)
    casses_par_paire: dict[str, int] = defaultdict(int)

    for s_idx, sent in enumerate(sentences):
        tokens = sent["tokens"]
        if not tokens:
            continue
        tokens = [t for t in tokens if t["phone"].strip()]
        if not tokens:
            continue

        ipa_words = [t["phone"] for t in tokens]
        gold_words = []
        is_formule = []
        for t in tokens:
            if t.get("formule_type"):
                gold_words.append(t.get("display_fr", t["form"]))
                is_formule.append(True)
            else:
                gold_words.append(t["form"])
                is_formule.append(False)
        n_words = len(ipa_words)

        # ── Inference ──
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

        # Apres lex_select
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

        # ── Baseline : step 1c original (a/a, ou/ou seulement) ──
        # On simule la baseline en appliquant corriger_phrase_v3 sans les
        # nouvelles regles. Pour ca on utilise l'ancienne version : steps 1-1b-1c(original).
        # Au lieu de dupliquer du code, on utilise le nouveau corriger_phrase_v3
        # (qui inclut 1c etendu + 1d) et on compare avec le gold.
        # Mais il nous faut aussi la baseline sans les nouvelles regles.
        # => On reconstruit manuellement la baseline.

        # Baseline = steps 1 + 1b + 1c original (a/a, ou/ou)
        # Nouveau  = steps 1 + 1b + 1c etendu + 1d morpho
        ortho_new = corriger_phrase_v3(
            list(ortho_lex), pos_pred, morpho_features,
            lexique=lexique_set,
            lexique_index=lexique_index,
            freq_map=freq_map,
        )

        # Pour la baseline, on reconstruit sans les nouvelles regles :
        # On part du meme ortho_lex, applique steps 1+1b+1c(original) manuellement.
        # C'est plus simple de comparer mot a mot en identifiant les changements.

        # Approche pragmatique : pour chaque mot, detecter si step 1c etendu
        # ou step 1d l'a modifie en comparant avec ce que le pipeline
        # aurait produit sans ces etapes.

        # On re-run le pipeline sans les nouvelles paires en utilisant une
        # version "baseline" minimale.
        from lectura_graphemiseur.posttraitement import forcer_coherence_ortho_morpho
        from lectura_graphemiseur._chargeur import (
            determinants_pluriel as _load_plur_det,
            determinants_singulier as _load_sing_det,
            invariables_pluriel as _load_no_plural_s,
        )
        _PLUR_DET = _load_plur_det()
        _SING_DET = _load_sing_det()
        _NO_PLURAL_S = _load_no_plural_s()
        _NO_STRIP_S_ENDINGS = ("ss", "is", "us", "as", "os", "ès", "ais", "ois", "urs")

        _in_lex = (lambda w: w.lower() in lexique_set) if lexique_set is not None else (lambda w: True)

        baseline = list(ortho_lex)
        # Step 1
        for i in range(n_words):
            if _in_lex(baseline[i]):
                continue
            pos_i = pos_pred[i] if i < len(pos_pred) else ""
            word_morpho = {
                feat: vals[i] if i < len(vals) else "_"
                for feat, vals in morpho_features.items()
            }
            baseline[i] = forcer_coherence_ortho_morpho(
                baseline[i], pos_i, word_morpho, lexique=lexique_set,
            )
        # Step 1b
        for i in range(n_words):
            pos_i = pos_pred[i] if i < len(pos_pred) else ""
            curr = baseline[i]
            morpho_number = "_"
            if "Number" in morpho_features and i < len(morpho_features["Number"]):
                morpho_number = morpho_features["Number"][i]
            if (i > 0 and pos_i in ("NOM", "ADJ") and morpho_number != "Plur"
                    and baseline[i - 1].lower() in _PLUR_DET
                    and not curr.endswith(("s", "x", "z"))
                    and len(curr) > 1 and curr.lower() not in _NO_PLURAL_S):
                candidate = curr + "s"
                if _in_lex(candidate):
                    baseline[i] = candidate
            curr = baseline[i]
            if (i > 0 and pos_i in ("NOM", "ADJ") and morpho_number != "Sing"
                    and baseline[i - 1].lower() in _SING_DET
                    and curr.endswith("s") and len(curr) > 2
                    and not curr.lower().endswith(_NO_STRIP_S_ENDINGS)):
                candidate = curr[:-1]
                if candidate and _in_lex(candidate):
                    baseline[i] = candidate
            curr = baseline[i]
            if (i > 0 and pos_i in ("VER", "AUX") and morpho_number != "Plur"
                    and baseline[i - 1].lower() in ("ils", "elles")
                    and curr.endswith("e") and not curr.endswith(("ent", "nt"))):
                candidate = curr + "nt"
                if _in_lex(candidate):
                    baseline[i] = candidate
        # Step 1c original (a/a, ou/ou seulement)
        for i in range(n_words):
            pos_i = pos_pred[i] if i < len(pos_pred) else ""
            lower = baseline[i].lower()
            if lower == "a" and pos_i == "PRE":
                baseline[i] = "à"
            elif lower == "à" and pos_i in ("AUX", "VER"):
                baseline[i] = "a"
            if lower == "ou" and pos_i in ("PRO:rel", "ADV"):
                baseline[i] = "où"
            elif lower == "où" and pos_i == "CON":
                baseline[i] = "ou"

        # ── Compter ──
        for w_idx in range(n_words):
            total_words += 1
            gold = gold_words[w_idx]
            g = gold.lower()
            b = baseline[w_idx].lower()
            nw = ortho_new[w_idx].lower()

            b_ok = (b == g)
            n_ok = (nw == g)

            if b_ok:
                baseline_correct += 1
            if n_ok:
                new_correct += 1

            if is_formule[w_idx]:
                continue

            # Detecter les changements
            if b != nw:
                lower_b = b
                # Identifier la paire
                pair_label = None

                # Step 1c etendu ?
                if lower_b in _POS_PAIRS or nw in _POS_PAIRS:
                    pair_label = "/".join(sorted({lower_b, nw}))
                    if not b_ok and n_ok:
                        fixes_1c.append({
                            "sent_idx": s_idx, "word_idx": w_idx,
                            "gold": gold, "baseline": baseline[w_idx],
                            "new": ortho_new[w_idx],
                            "pos": pos_pred[w_idx] if w_idx < len(pos_pred) else "",
                            "pair": pair_label,
                        })
                        fixes_par_paire[pair_label] += 1
                    elif b_ok and not n_ok:
                        casses_1c.append({
                            "sent_idx": s_idx, "word_idx": w_idx,
                            "gold": gold, "baseline": baseline[w_idx],
                            "new": ortho_new[w_idx],
                            "pos": pos_pred[w_idx] if w_idx < len(pos_pred) else "",
                            "pair": pair_label,
                        })
                        casses_par_paire[pair_label] += 1

                # Step 1d morpho ?
                elif lower_b in _HOMOPHONES_MORPHO or nw in _HOMOPHONES_MORPHO:
                    pair_label = "/".join(sorted({lower_b, nw}))
                    morpho_num = "_"
                    if "Number" in morpho_features and w_idx < len(morpho_features["Number"]):
                        morpho_num = morpho_features["Number"][w_idx]
                    if not b_ok and n_ok:
                        fixes_1d.append({
                            "sent_idx": s_idx, "word_idx": w_idx,
                            "gold": gold, "baseline": baseline[w_idx],
                            "new": ortho_new[w_idx],
                            "morpho_num": morpho_num,
                            "pair": pair_label,
                        })
                        fixes_par_paire[pair_label] += 1
                    elif b_ok and not n_ok:
                        casses_1d.append({
                            "sent_idx": s_idx, "word_idx": w_idx,
                            "gold": gold, "baseline": baseline[w_idx],
                            "new": ortho_new[w_idx],
                            "morpho_num": morpho_num,
                            "pair": pair_label,
                        })
                        casses_par_paire[pair_label] += 1

    duree = time.time() - t0

    # ── Affichage ──
    print()
    print("=" * 80)
    print("BENCHMARK HOMOPHONES — PREDICTIONS DIRECTES DU MODELE")
    print("=" * 80)
    print()
    print(f"  Dataset : {n_phrases} phrases, {total_words} mots")
    print(f"  Duree   : {duree:.1f}s")
    print()
    print(f"  Baseline (steps 1-1b-1c original) : {baseline_correct:>6} correct "
          f"({baseline_correct/total_words*100:.3f}%)")
    print(f"  Nouveau  (+ 1c etendu + 1d morpho): {new_correct:>6} correct "
          f"({new_correct/total_words*100:.3f}%)")
    delta_total = new_correct - baseline_correct
    print(f"  Delta total : {delta_total:+d}")

    # ── Step 1c etendu ──
    print()
    print(f"  Step 1c etendu (POS-level) :")
    print(f"    Fixes : {len(fixes_1c):>4}  |  Casses : {len(casses_1c):>4}  "
          f"|  Net : {len(fixes_1c) - len(casses_1c):+d}")

    # ── Step 1d morpho ──
    print(f"  Step 1d morpho (Number) :")
    print(f"    Fixes : {len(fixes_1d):>4}  |  Casses : {len(casses_1d):>4}  "
          f"|  Net : {len(fixes_1d) - len(casses_1d):+d}")

    # ── Detail par paire ──
    all_pairs = sorted(set(list(fixes_par_paire.keys()) + list(casses_par_paire.keys())),
                       key=lambda p: -(fixes_par_paire.get(p, 0) - casses_par_paire.get(p, 0)))
    if all_pairs:
        print()
        print(f"  Detail par paire :")
        print(f"    {'Paire':25s} | {'Fixes':>5s} | {'Casses':>6s} | {'Net':>5s}")
        print(f"    {'-'*25}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}")
        for pair in all_pairs:
            f = fixes_par_paire.get(pair, 0)
            c = casses_par_paire.get(pair, 0)
            n = f - c
            print(f"    {pair:25s} | {f:>5d} | {c:>6d} | {n:>+5d}")

    if show_details:
        if fixes_1c:
            print(f"\n  Fixes step 1c (POS-level, max 20) :")
            for case in fixes_1c[:20]:
                print(f"    {case['baseline']:>10s} -> {case['new']:>10s}  "
                      f"gold={case['gold']:>10s}  pos={case['pos']:>8s}  "
                      f"pair={case['pair']}")
        if casses_1c:
            print(f"\n  Casses step 1c (POS-level, max 20) :")
            for case in casses_1c[:20]:
                print(f"    {case['baseline']:>10s} -> {case['new']:>10s}  "
                      f"gold={case['gold']:>10s}  pos={case['pos']:>8s}  "
                      f"pair={case['pair']}")
        if fixes_1d:
            print(f"\n  Fixes step 1d (morpho, max 20) :")
            for case in fixes_1d[:20]:
                print(f"    {case['baseline']:>10s} -> {case['new']:>10s}  "
                      f"gold={case['gold']:>10s}  morpho_num={case['morpho_num']:>5s}  "
                      f"pair={case['pair']}")
        if casses_1d:
            print(f"\n  Casses step 1d (morpho, max 20) :")
            for case in casses_1d[:20]:
                print(f"    {case['baseline']:>10s} -> {case['new']:>10s}  "
                      f"gold={case['gold']:>10s}  morpho_num={case['morpho_num']:>5s}  "
                      f"pair={case['pair']}")

    # ── Export JSON ──
    if export_path:
        export = {
            "n_phrases": n_phrases,
            "n_mots": total_words,
            "baseline_correct": baseline_correct,
            "new_correct": new_correct,
            "delta_total": delta_total,
            "fixes_1c": len(fixes_1c),
            "casses_1c": len(casses_1c),
            "fixes_1d": len(fixes_1d),
            "casses_1d": len(casses_1d),
            "par_paire": {
                p: {"fixes": fixes_par_paire.get(p, 0), "casses": casses_par_paire.get(p, 0)}
                for p in all_pairs
            },
            "details_fixes_1c": fixes_1c,
            "details_casses_1c": casses_1c,
            "details_fixes_1d": fixes_1d,
            "details_casses_1d": casses_1d,
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        print(f"\n  Export : {export_path}")

    return {
        "baseline_correct": baseline_correct,
        "new_correct": new_correct,
        "delta_total": delta_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark homophones par predictions directes du modele",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Chemin vers sentences_dev.json",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Afficher le detail des fixes et casses",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Exporter les resultats en JSON",
    )
    args = parser.parse_args()

    benchmark_homophones(
        dev_set_path=args.data,
        show_details=args.details,
        export_path=args.export,
    )


if __name__ == "__main__":
    main()
