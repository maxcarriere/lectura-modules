#!/usr/bin/env python3
"""Benchmark Gender NOM : mesure la precision du genre predit pour les noms.

Evalue specifiquement la capacite du G2P a predire le genre (Masc/Fem)
des noms communs, en comparant au gold standard UD French-GSD (dev set)
enrichi avec Gender[lex].

Usage :
    python3 _bench_gender_nom.py                  # G2P seul, modele courant
    python3 _bench_gender_nom.py --label avant     # tag pour sauvegarde
    python3 _bench_gender_nom.py --strategies A,B   # G2P + Lexique
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

KIT_DIR = "/data/work/projets/lectura/workspace/Corpus/Kit-G2P-P2G/corpus/phrases"
LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"

# POS tags qui correspondent a des noms dans notre schema
NOM_POS_TAGS = {"NOM"}


def charger_dev_set() -> list[dict]:
    """Charge les phrases du dev set avec sent_id."""
    path = os.path.join(KIT_DIR, "sentences_dev.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pos_base(pos: str) -> str:
    return pos.split(":")[0]


# ── Strategies ────────────────────────────────────────────────────────────

def predire_g2p(phrases, tagger):
    """Strategie A : G2P seul."""
    nombre_map = {"s": "Sing", "p": "Plur"}
    genre_map = {"m": "Masc", "f": "Fem"}

    results = []
    for sent in phrases:
        mots = [t["form"] for t in sent["tokens"]]
        try:
            tags = tagger.tag_words_rich(mots)
        except Exception:
            tags = [{}] * len(mots)

        preds = []
        for i, t in enumerate(sent["tokens"]):
            g = tags[i] if i < len(tags) else {}
            preds.append({
                "pos": g.get("pos", "NOM"),
                "number": nombre_map.get(g.get("nombre", ""), "_"),
                "gender": genre_map.get(g.get("genre", ""), "_"),
                "person": g.get("personne", "_") or "_",
            })
        results.append(preds)
    return results


def predire_lexique(phrases, lexique):
    """Strategie B : Lexique normalise (entree la plus frequente)."""
    from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS

    # LexiqueNormalise renvoie m/f/s/p ; Lexique brut renvoie masculin/feminin/singulier/pluriel
    genre_map = {"m": "Masc", "f": "Fem", "masculin": "Masc", "feminin": "Fem", "": "_"}
    nombre_map = {"s": "Sing", "p": "Plur", "singulier": "Sing", "pluriel": "Plur", "": "_"}

    results = []
    for sent in phrases:
        preds = []
        for t in sent["tokens"]:
            form = t["form"].lower()
            override = _FUNCTION_WORD_POS.get(form)
            infos = lexique.info(form) if hasattr(lexique, "info") else []

            if not infos:
                preds.append({"pos": override or "NOM", "number": "_", "gender": "_", "person": "_"})
                continue

            if override:
                base = override.split(":")[0]
                filtered = [e for e in infos if e.get("cgram", "").split(":")[0] == base]
                if filtered:
                    infos = filtered

            best = max(infos, key=lambda e: float(e.get("freq") or 0))
            cgram = best.get("cgram", "NOM")
            if override and ":" not in cgram and cgram == override.split(":")[0]:
                cgram = override

            preds.append({
                "pos": cgram,
                "number": nombre_map.get(best.get("nombre", ""), "_"),
                "gender": genre_map.get(best.get("genre", ""), "_"),
                "person": best.get("personne", "") or "_",
            })
        results.append(preds)
    return results


# ── Evaluation detaillee Gender NOM ───────────────────────────────────────

def evaluer_gender_nom(phrases, predictions, label: str) -> dict:
    """Evaluation detaillee du genre pour les NOM."""

    # Compteurs globaux (tous POS)
    total_all = 0
    pos_ok_all = 0
    gender_ok_all = 0
    number_ok_all = 0

    # Compteurs NOM specifiques
    nom_total = 0
    nom_gender_annotated = 0  # gold Gender != "_"
    nom_gender_correct = 0    # prediction correcte parmi les annotes
    nom_gender_predicted = 0  # pred Gender != "_" parmi les annotes
    nom_gender_correct_when_predicted = 0

    # Confusion matrix NOM Gender (gold → pred)
    gender_confusion = Counter()  # (gold, pred) → count

    # Par genre gold
    by_gold_gender = defaultdict(lambda: {"total": 0, "correct": 0})

    # Erreurs NOM Gender detaillees
    nom_gender_errors = []  # (form, gold_gender, pred_gender, sent_id)

    # Statistiques POS pour NOM
    nom_pos_correct = 0

    for sent, preds in zip(phrases, predictions):
        tokens = sent["tokens"]
        sid = sent.get("sent_id", "")

        for i, (tok, pred) in enumerate(zip(tokens, preds)):
            total_all += 1
            gold_pos = tok["pos_tag"]
            gold_morpho = tok.get("morpho", {})
            gold_gender = gold_morpho.get("Gender", "_")
            gold_number = gold_morpho.get("Number", "_")

            pred_pos = pred["pos"]
            pred_gender = pred["gender"]
            pred_number = pred["number"]

            # Global
            if _pos_base(pred_pos) == _pos_base(gold_pos):
                pos_ok_all += 1
            if gold_gender == "_" or pred_gender == gold_gender:
                gender_ok_all += 1
            if gold_number == "_" or pred_number == gold_number:
                number_ok_all += 1

            # NOM specifique
            if _pos_base(gold_pos) in NOM_POS_TAGS:
                nom_total += 1
                if _pos_base(pred_pos) in NOM_POS_TAGS:
                    nom_pos_correct += 1

                if gold_gender != "_":
                    nom_gender_annotated += 1
                    gender_confusion[(gold_gender, pred_gender)] += 1
                    by_gold_gender[gold_gender]["total"] += 1

                    if pred_gender == gold_gender:
                        nom_gender_correct += 1
                        by_gold_gender[gold_gender]["correct"] += 1
                    else:
                        nom_gender_errors.append((
                            tok["form"], gold_gender, pred_gender, sid,
                        ))

                    if pred_gender != "_":
                        nom_gender_predicted += 1
                        if pred_gender == gold_gender:
                            nom_gender_correct_when_predicted += 1

    # ── Rapport ──
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n  --- Metriques globales (tous POS) ---")
    print(f"  Tokens            : {total_all}")
    print(f"  POS base          : {pos_ok_all}/{total_all} ({100*pos_ok_all/total_all:.2f}%)")
    print(f"  Gender (skip _)   : {gender_ok_all}/{total_all} ({100*gender_ok_all/total_all:.2f}%)")
    print(f"  Number (skip _)   : {number_ok_all}/{total_all} ({100*number_ok_all/total_all:.2f}%)")

    print(f"\n  --- NOM specifique ---")
    print(f"  NOM total         : {nom_total}")
    print(f"  NOM POS correct   : {nom_pos_correct}/{nom_total} ({100*nom_pos_correct/nom_total:.2f}%)" if nom_total else "")
    print(f"  NOM avec Gender   : {nom_gender_annotated}/{nom_total} ({100*nom_gender_annotated/nom_total:.1f}%)")

    if nom_gender_annotated:
        acc = 100 * nom_gender_correct / nom_gender_annotated
        print(f"\n  NOM Gender correct: {nom_gender_correct}/{nom_gender_annotated} ({acc:.2f}%)")
        print(f"  NOM Gender predit : {nom_gender_predicted}/{nom_gender_annotated} "
              f"({100*nom_gender_predicted/nom_gender_annotated:.1f}%)")
        if nom_gender_predicted:
            prec = 100 * nom_gender_correct_when_predicted / nom_gender_predicted
            print(f"  Precision (quand predit): {nom_gender_correct_when_predicted}/{nom_gender_predicted} ({prec:.2f}%)")

        print(f"\n  Par genre gold :")
        for g in ("Masc", "Fem"):
            d = by_gold_gender[g]
            if d["total"]:
                print(f"    {g:6s}: {d['correct']}/{d['total']} ({100*d['correct']/d['total']:.2f}%)")

        print(f"\n  Matrice de confusion Gender NOM (gold → pred) :")
        print(f"    {'':8s} {'pred _':>8s} {'pred Masc':>10s} {'pred Fem':>10s}")
        for g_gold in ("Masc", "Fem"):
            row = []
            for g_pred in ("_", "Masc", "Fem"):
                row.append(gender_confusion.get((g_gold, g_pred), 0))
            total_row = sum(row)
            print(f"    {g_gold:8s} {row[0]:8d} {row[1]:10d} {row[2]:10d}  (total={total_row})")

    # Top erreurs NOM Gender
    error_counter = Counter()
    for form, g_gold, g_pred, sid in nom_gender_errors:
        error_counter[(form.lower(), g_gold, g_pred)] += 1

    if error_counter:
        print(f"\n  Top 30 erreurs NOM Gender :")
        for (form, g_gold, g_pred), n in error_counter.most_common(30):
            print(f"    {form:20s} gold={g_gold:5s} pred={g_pred:5s} x{n}")

    # Resultat structure
    result = {
        "label": label,
        "global": {
            "total": total_all,
            "pos_base": 100 * pos_ok_all / total_all if total_all else 0,
            "gender": 100 * gender_ok_all / total_all if total_all else 0,
            "number": 100 * number_ok_all / total_all if total_all else 0,
        },
        "nom": {
            "total": nom_total,
            "gender_annotated": nom_gender_annotated,
            "gender_correct": nom_gender_correct,
            "gender_accuracy": 100 * nom_gender_correct / nom_gender_annotated if nom_gender_annotated else 0,
            "gender_predicted": nom_gender_predicted,
            "by_gold": {g: dict(d) for g, d in by_gold_gender.items()},
            "confusion": {f"{g}→{p}": n for (g, p), n in gender_confusion.items()},
        },
    }
    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Gender NOM")
    parser.add_argument("--label", type=str, default="",
                        help="Label pour identifier le run (ex: avant, apres)")
    parser.add_argument("--strategies", type=str, default="A",
                        help="Strategies : A=G2P, B=Lexique, A,B=les deux")
    parser.add_argument("--save", type=str, default="",
                        help="Chemin JSON pour sauvegarder les resultats")
    args = parser.parse_args()

    strategies = [s.strip().upper() for s in args.strategies.split(",")]

    print("Chargement dev set UD French-GSD (avec Gender[lex])...")
    phrases = charger_dev_set()
    total_tokens = sum(len(s["tokens"]) for s in phrases)

    # Stats gold Gender NOM
    nom_total = 0
    nom_with_gender = 0
    for s in phrases:
        for t in s["tokens"]:
            if _pos_base(t["pos_tag"]) in NOM_POS_TAGS:
                nom_total += 1
                if t.get("morpho", {}).get("Gender", "_") != "_":
                    nom_with_gender += 1

    print(f"  {len(phrases)} phrases, {total_tokens} tokens")
    print(f"  NOM: {nom_total} total, {nom_with_gender} avec Gender ({100*nom_with_gender/nom_total:.1f}%)")

    all_results = {}

    # A. G2P
    if "A" in strategies:
        print("\nChargement G2P Unifie V2...")
        try:
            from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
            tagger = creer_adapter_g2p_unifie()
            if tagger:
                print("  G2P charge")
                t0 = time.time()
                preds = predire_g2p(phrases, tagger)
                elapsed = time.time() - t0
                lbl = f"A. G2P seul ({elapsed:.1f}s)"
                if args.label:
                    lbl = f"A. G2P seul [{args.label}] ({elapsed:.1f}s)"
                r = evaluer_gender_nom(phrases, preds, lbl)
                all_results["A"] = r
            else:
                print("  G2P indisponible")
        except Exception as e:
            print(f"  ERREUR G2P: {e}")

    # B. Lexique
    if "B" in strategies:
        print("\nChargement Lexique...")
        from lectura_lexique import Lexique
        from lectura_correcteur._utils import LexiqueNormalise
        lexique = LexiqueNormalise(Lexique(LEXIQUE_DB))
        t0 = time.time()
        preds = predire_lexique(phrases, lexique)
        elapsed = time.time() - t0
        lbl = f"B. Lexique seul ({elapsed:.1f}s)"
        if args.label:
            lbl = f"B. Lexique seul [{args.label}] ({elapsed:.1f}s)"
        r = evaluer_gender_nom(phrases, preds, lbl)
        all_results["B"] = r

    # Comparaison rapide
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARATIF NOM Gender")
        print(f"{'='*70}")
        print(f"  {'Strategie':<30s} {'NOM Gender%':>12s} {'Masc%':>8s} {'Fem%':>8s} {'Predit%':>8s}")
        print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
        for key in ("A", "B"):
            if key not in all_results:
                continue
            r = all_results[key]
            nom = r["nom"]
            masc = nom["by_gold"].get("Masc", {"total": 0, "correct": 0})
            fem = nom["by_gold"].get("Fem", {"total": 0, "correct": 0})
            masc_acc = 100 * masc["correct"] / masc["total"] if masc["total"] else 0
            fem_acc = 100 * fem["correct"] / fem["total"] if fem["total"] else 0
            pred_pct = 100 * nom["gender_predicted"] / nom["gender_annotated"] if nom["gender_annotated"] else 0
            labels = {"A": "G2P seul", "B": "Lexique seul"}
            print(f"  {labels[key]:<30s} {nom['gender_accuracy']:12.2f} {masc_acc:8.2f} {fem_acc:8.2f} {pred_pct:8.1f}")

    # Sauvegarder
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResultats sauvegardes: {args.save}")

    print("\nTermine.")


if __name__ == "__main__":
    main()
