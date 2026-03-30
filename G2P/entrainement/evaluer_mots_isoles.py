#!/usr/bin/env python3
"""Évaluation G2P du modèle unifié sur mots isolés par tranche de fréquence.

Croise Lexique383.tsv (fréquences) avec dico.csv (gold), évalue le modèle
unifié en mode mot isolé (sans contexte phrastique).

Produit un rapport COMPLET avec TOUTES les erreurs, exploitable pour
construire une table de corrections.

Inclut une section dédiée aux élisions (j', l', d', etc.).

Usage :
    python entrainement/evaluer_mots_isoles.py [--backend onnx|numpy|pure]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LECTURA_MAIN = PROJECT_ROOT.parent / "lectura-main"

LEXIQUE_TSV = Path(
    "/home/moi/Documents/work/projets/lectura/workspace/"
    "ressources-éducatives/lexique383/data/raw/Lexique383.tsv"
)
DICO_CSV = LECTURA_MAIN / "src" / "lecteur_syllabique" / "data" / "dico.csv"
HOMOGRAPHES_JSON = LECTURA_MAIN / "src" / "lecteur_syllabique" / "data" / "homographes.json"

MODEL_DIR = PROJECT_ROOT / "modeles"
ONNX_MODEL = MODEL_DIR / "unifie_int8.onnx"
VOCAB_JSON = MODEL_DIR / "unifie_vocab.json"
NUMPY_WEIGHTS = MODEL_DIR / "unifie_weights.json"

OUTPUT_DIR = PROJECT_ROOT / "evaluation"

COVERAGE_THRESHOLDS = [0.50, 0.80, 0.90, 0.95, 0.99]

# Élisions standard du français
ELISIONS = {
    "j'": "ʒ", "t'": "t", "m'": "m", "l'": "l",
    "d'": "d", "n'": "n", "s'": "s", "c'": "s",
    "qu'": "k",
    "jusqu'": "ʒysk", "lorsqu'": "lɔʁsk", "puisqu'": "pɥisk",
    "quelqu'": "kɛlk", "quoiqu'": "kwak", "presqu'": "pʁɛsk",
}

# ── IPA utils ────────────────────────────────────────────────────────────


def iter_phonemes(ipa: str) -> list[str]:
    if not ipa:
        return []
    phonemes: list[str] = []
    current = ""
    for ch in ipa:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            current += ch
        else:
            if current:
                phonemes.append(current)
            current = ch
    if current:
        phonemes.append(current)
    return phonemes


def _normalize_o(ipa: str) -> str:
    phonemes = iter_phonemes(ipa)
    result: list[str] = []
    for ph in phonemes:
        if len(ph) == 1 and ph == "ɔ":
            result.append("o")
        else:
            result.append(ph)
    return "".join(result)


def _normalize_mid_vowels(ipa: str) -> str:
    phonemes = iter_phonemes(ipa)
    result: list[str] = []
    for ph in phonemes:
        if len(ph) == 1 and ph == "ɔ":
            result.append("o")
        elif len(ph) == 1 and ph == "ɛ":
            result.append("e")
        else:
            result.append(ph)
    return "".join(result)


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


# ── Catégorisation automatique des erreurs ───────────────────────────────


def categorize_error(word: str, pred: str, gold: str) -> str:
    """Catégorise une erreur G2P pour identifier les patterns."""
    w = word.lower()

    # 1) Cluster ex-
    if w.startswith("ex"):
        if "ɡz" in gold and "ɡz" not in pred:
            return "ex→ɛɡz (gz manquant)"
        if "ks" in gold and "ks" not in pred:
            return "ex→ɛks (ks manquant)"

    # 2) Cluster -xt- / -xp- / -xc-
    if "x" in w and not w.startswith("ex"):
        if "ks" in gold and "ks" not in pred:
            return "-x-→ks (ks manquant)"

    # 3) Yod manquant (ij → j)
    if "ij" in gold and "ij" not in pred and "j" in pred:
        return "yod manquant (ij→j)"

    # 4) Digraphe oe/oeu → œ
    if ("oe" in w or "oeu" in w or "ueil" in w or "oeil" in w):
        if "œ" in gold and "œ" not in pred:
            return "oe/oeu→œ (digraphe)"

    # 5) Consonne finale muette ajoutée
    if gold and pred:
        pred_ph = iter_phonemes(pred)
        gold_ph = iter_phonemes(gold)
        if len(pred_ph) > len(gold_ph):
            extra = pred_ph[len(gold_ph):]
            if all(c in "lstkpfʁ" for c in extra):
                return "consonne finale muette"

    # 6) Schwa
    if "ə" in gold and "ə" not in pred:
        return "schwa manquant"
    if "ə" not in gold and "ə" in pred:
        return "schwa parasite"

    # 7) Tolérance voyelles moyennes
    if _normalize_mid_vowels(pred) == _normalize_mid_vowels(gold):
        return "voyelle mi-ouverte (ɛ/e, ɔ/o)"

    return "autre"


# ── Chargement données ──────────────────────────────────────────────────


def load_lexique_freqs(path: Path) -> list[tuple[str, float]]:
    word_freq: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            freq_str = row.get("freqlivres", "").strip()
            if not ortho or " " in ortho or "'" in ortho or "-" in ortho:
                continue
            try:
                freq = float(freq_str)
            except (ValueError, TypeError):
                continue
            word_freq[ortho] = word_freq.get(ortho, 0.0) + freq
    return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)


def compute_coverage_buckets(
    sorted_words: list[tuple[str, float]],
    thresholds: list[float],
) -> list[tuple[float, list[tuple[str, float]]]]:
    total_freq = sum(f for _, f in sorted_words)
    if total_freq == 0:
        return []
    cumul = 0.0
    prev_cutoff = 0
    buckets: list[tuple[float, list[tuple[str, float]]]] = []
    for threshold in thresholds:
        target = total_freq * threshold
        cutoff = prev_cutoff
        while cutoff < len(sorted_words) and cumul < target:
            cumul += sorted_words[cutoff][1]
            cutoff += 1
        buckets.append((threshold, sorted_words[prev_cutoff:cutoff]))
        prev_cutoff = cutoff
    return buckets


def load_dico(path: Path) -> dict[str, set[str]]:
    word_phones: dict[str, set[str]] = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()
            if not ortho or not phone or " " in ortho:
                continue
            if ortho not in word_phones:
                word_phones[ortho] = set()
            word_phones[ortho].add(phone)
    return word_phones


def load_homographes(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.keys())


# ── Moteur d'inférence ──────────────────────────────────────────────────


CORRECTIONS_JSON = MODEL_DIR / "g2p_corrections_unifie.json"


def load_engine(backend: str):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    if backend == "onnx":
        from lectura_nlp.inference_onnx import OnnxInferenceEngine
        return OnnxInferenceEngine(str(ONNX_MODEL), str(VOCAB_JSON))
    elif backend == "numpy":
        from lectura_nlp.inference_numpy import NumpyInferenceEngine
        return NumpyInferenceEngine(str(NUMPY_WEIGHTS), str(VOCAB_JSON))
    elif backend == "pure":
        from lectura_nlp.inference_pure import PurePythonInferenceEngine
        return PurePythonInferenceEngine(str(NUMPY_WEIGHTS), str(VOCAB_JSON))
    else:
        raise ValueError(f"Backend inconnu : {backend}")


def _init_corrections() -> None:
    """Charge la table de corrections + règles via posttraitement."""
    from lectura_nlp.posttraitement import charger_corrections
    if CORRECTIONS_JSON.exists():
        charger_corrections(CORRECTIONS_JSON)


def predict_word(engine, word: str) -> str:
    from lectura_nlp.posttraitement import corriger_g2p
    try:
        result = engine.analyser([word])
        ipa = result["g2p"][0] if result["g2p"] else ""
    except Exception:
        ipa = ""
    return corriger_g2p(word, ipa)


# ── Évaluation ──────────────────────────────────────────────────────────


def evaluate_words(
    engine,
    words: list[tuple[str, float]],
    dico: dict[str, set[str]],
    homographes: set[str],
    exclude_homographes: bool = False,
) -> tuple[dict, list[dict]]:
    """Retourne (metrics, errors) avec catégorisation de chaque erreur."""
    total = 0
    correct = 0
    correct_o = 0
    correct_oe = 0
    no_gold = 0
    skipped_homo = 0
    errors: list[dict] = []

    for ortho, freq in words:
        if exclude_homographes and ortho in homographes:
            skipped_homo += 1
            continue

        phones_set = dico.get(ortho)
        if not phones_set:
            no_gold += 1
            continue

        total += 1
        pred = predict_word(engine, ortho)

        is_correct = pred in phones_set
        pred_o = _normalize_o(pred)
        is_correct_o = any(_normalize_o(g) == pred_o for g in phones_set)
        pred_oe = _normalize_mid_vowels(pred)
        is_correct_oe = any(_normalize_mid_vowels(g) == pred_oe for g in phones_set)

        if is_correct:
            correct += 1
        if is_correct_o:
            correct_o += 1
        if is_correct_oe:
            correct_oe += 1

        if not is_correct:
            pred_ph = iter_phonemes(pred) if pred else []
            best_gold = min(
                phones_set,
                key=lambda g: _levenshtein(pred_ph, iter_phonemes(g)),
            )
            cat = categorize_error(ortho, pred, best_gold)
            errors.append({
                "ortho": ortho, "freq": freq, "pred": pred,
                "best_gold": best_gold, "all_golds": phones_set,
                "category": cat,
                "correct_with_tol": is_correct_oe,
            })

    return {
        "total": total,
        "correct": correct,
        "correct_o": correct_o,
        "correct_oe": correct_oe,
        "no_gold": no_gold,
        "skipped_homo": skipped_homo,
        "word_acc": correct / total if total else 0,
        "word_acc_o": correct_o / total if total else 0,
        "word_acc_oe": correct_oe / total if total else 0,
    }, errors


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation G2P du modèle unifié sur mots isolés"
    )
    parser.add_argument(
        "--backend", choices=["onnx", "numpy", "pure"], default="onnx",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, name in [
        (LEXIQUE_TSV, "Lexique383.tsv"),
        (DICO_CSV, "dico.csv"),
        (HOMOGRAPHES_JSON, "homographes.json"),
        (ONNX_MODEL, "unifie_int8.onnx"),
        (VOCAB_JSON, "unifie_vocab.json"),
    ]:
        if not path.exists():
            print(f"ERREUR : {path} ({name}) non trouvé")
            sys.exit(1)

    # ── Chargement ──

    print("Chargement de Lexique383.tsv...")
    sorted_words = load_lexique_freqs(LEXIQUE_TSV)
    print(f"  {len(sorted_words):,} formes uniques")

    print("Chargement de dico.csv...")
    dico = load_dico(DICO_CSV)
    print(f"  {len(dico):,} mots avec prononciation")

    print("Chargement de homographes.json...")
    homographes = load_homographes(HOMOGRAPHES_JSON)
    print(f"  {len(homographes):,} homographes")

    print(f"Chargement du modèle unifié ({args.backend})...")
    engine = load_engine(args.backend)
    print("  OK")

    print("Chargement des corrections G2P...")
    _init_corrections()
    n_corr = "table chargée" if CORRECTIONS_JSON.exists() else "pas de table"
    print(f"  {n_corr}")

    # ── Tranches de couverture ──

    buckets = compute_coverage_buckets(sorted_words, COVERAGE_THRESHOLDS)
    labels = ["0→50%", "50→80%", "80→90%", "90→95%", "95→99%"]
    cumul_labels = ["50%", "80%", "90%", "95%", "99%"]

    report_path = OUTPUT_DIR / "g2p_mots_isoles_complet.md"
    corrections_path = OUTPUT_DIR / "corrections_candidates.json"

    t0 = time.time()

    all_errors: list[dict] = []
    cumul = {"total": 0, "correct": 0, "correct_o": 0, "correct_oe": 0}
    all_metrics: list[dict] = []

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Évaluation G2P — Modèle unifié — Mots isolés (rapport complet)\n\n")
        f.write(f"- **Modèle** : unifié BiLSTM multi-tâche ({args.backend})\n")
        f.write("- **Mode** : mot isolé (sans contexte phrastique)\n")
        f.write("- **Fréquences** : Lexique 3.83 (freqlivres)\n")
        f.write("- **Gold** : dico.csv (match si pred ∈ prononciations valides)\n")
        f.write("- **Filtres** : composés (apostrophe/tiret) exclus, "
                "homographes inclus\n\n")

        # ── Évaluation par tranche ──

        for i, (threshold, words) in enumerate(buckets):
            label = labels[i]
            print(f"\nTrache {label} ({len(words):,} formes)...")

            metrics, errors = evaluate_words(
                engine, words, dico, homographes, exclude_homographes=False,
            )
            cumul["total"] += metrics["total"]
            cumul["correct"] += metrics["correct"]
            cumul["correct_o"] += metrics["correct_o"]
            cumul["correct_oe"] += metrics["correct_oe"]

            ca = cumul
            cumul_acc = ca["correct"] / ca["total"] if ca["total"] else 0

            all_metrics.append({
                "label": label, "cumul_label": cumul_labels[i],
                "words": len(words), **metrics,
                "cumul_total": ca["total"], "cumul_acc": cumul_acc,
                "n_errors": len(errors),
            })
            all_errors.extend(errors)

            print(f"  Évalués: {metrics['total']:,} | "
                  f"Acc: {metrics['word_acc']:.1%} | Erreurs: {len(errors):,}")
            print(f"  Cumulé (→{cumul_labels[i]}): {ca['total']:,}, "
                  f"Acc: {cumul_acc:.1%}")

            # Écrire les erreurs de cette tranche
            f.write(f"## Tranche {label} (couverture cumulée → {cumul_labels[i]})\n\n")
            f.write(f"- Formes Lexique : {len(words):,}\n")
            f.write(f"- Évaluées : {metrics['total']:,}\n")
            f.write(f"- **Word Acc** : {metrics['word_acc']:.2%} | "
                    f"o/ɔ-tol : {metrics['word_acc_o']:.2%} | "
                    f"o/ɔ+e/ɛ-tol : {metrics['word_acc_oe']:.2%}\n")
            f.write(f"- **Erreurs** : {len(errors):,}\n\n")

            if errors:
                # Catégories pour cette tranche
                cats = Counter(e["category"] for e in errors)
                f.write("**Répartition des erreurs :**\n\n")
                f.write("| Catégorie | Nb | % |\n")
                f.write("|-----------|-----|---|\n")
                for cat, cnt in cats.most_common():
                    f.write(f"| {cat} | {cnt} | {cnt/len(errors):.0%} |\n")
                f.write("\n")

                # Toutes les erreurs, triées par fréquence
                errors.sort(key=lambda x: x["freq"], reverse=True)
                f.write("| Mot | Fréq | Prédiction | Gold | Cat. |\n")
                f.write("|-----|------|------------|------|------|\n")
                for err in errors:
                    golds = ", ".join(sorted(err["all_golds"]))
                    f.write(
                        f"| {err['ortho']} | {err['freq']:.1f} | "
                        f"{err['pred']} | {err['best_gold']} | "
                        f"{err['category']} |\n"
                    )
                f.write("\n")

        # ── Tableau récapitulatif ──

        f.write("---\n\n## Récapitulatif\n\n")
        f.write("| Tranche | Couv. | Évaluées | Acc (exacte) | Acc (o/ɔ) | "
                "Acc (o/ɔ+e/ɛ) | Erreurs |\n")
        f.write("|---------|-------|----------|-------------|-----------|"
                "---------------|--------|\n")
        for bm in all_metrics:
            f.write(
                f"| {bm['label']} | {bm['cumul_label']} | "
                f"{bm['total']:,} | {bm['word_acc']:.2%} | "
                f"{bm['word_acc_o']:.2%} | {bm['word_acc_oe']:.2%} | "
                f"{bm['n_errors']:,} |\n"
            )
        f.write("\n")

        # ── Catégories globales ──

        f.write("## Répartition globale des erreurs\n\n")
        global_cats = Counter(e["category"] for e in all_errors)
        f.write("| Catégorie | Nb | % |\n")
        f.write("|-----------|-----|---|\n")
        for cat, cnt in global_cats.most_common():
            f.write(f"| {cat} | {cnt:,} | {cnt/len(all_errors):.1%} |\n")
        f.write(f"\n**Total erreurs : {len(all_errors):,}**\n\n")

        # ── Section élisions ──

        f.write("---\n\n## Élisions\n\n")
        f.write("Test des formes élidées (j', l', d', etc.) en mode isolé :\n\n")
        f.write("| Forme | Gold | Prédiction | Correct |\n")
        f.write("|-------|------|------------|--------|\n")

        n_elision_ok = 0
        n_elision_total = len(ELISIONS)
        elision_corrections: list[tuple[str, str, str]] = []

        for form, gold in sorted(ELISIONS.items()):
            pred = predict_word(engine, form)
            ok = pred == gold
            if ok:
                n_elision_ok += 1
            else:
                elision_corrections.append((form, pred, gold))
            mark = "OK" if ok else f"**KO** ({pred})"
            f.write(f"| {form} | {gold} | {pred} | {mark} |\n")

        f.write(f"\n**Élisions : {n_elision_ok}/{n_elision_total} correctes**\n\n")

        if elision_corrections:
            f.write("Corrections nécessaires :\n\n")
            for form, pred, gold in elision_corrections:
                f.write(f"- `{form}` : {pred} → **{gold}**\n")
            f.write("\n")

        # ── Temps ──

        elapsed = time.time() - t0
        f.write(f"\n*Temps total : {elapsed:.1f}s*\n")

    # ── Générer le fichier corrections_candidates.json ──

    # Regrouper les erreurs par mot (dédupliquer)
    corrections: dict[str, dict] = {}

    # 1) Élisions
    for form, pred, gold in elision_corrections:
        corrections[form] = {"pred": pred, "gold": gold, "source": "elision"}

    # 2) Erreurs par fréquence (les plus fréquentes d'abord)
    all_errors.sort(key=lambda x: x["freq"], reverse=True)
    for err in all_errors:
        w = err["ortho"]
        if w not in corrections:
            corrections[w] = {
                "pred": err["pred"],
                "gold": err["best_gold"],
                "freq": err["freq"],
                "category": err["category"],
            }

    with open(corrections_path, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=1)

    # ── Console ──

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("RÉSUMÉ")
    print(f"{'=' * 60}")

    ca = cumul
    all_acc = ca["correct"] / ca["total"] if ca["total"] else 0
    print(f"\nCouverture 99% ({ca['total']:,} mots) :")
    print(f"  Word Acc (exacte)    : {all_acc:.2%}")
    print(f"  Word Acc (o/ɔ+e/ɛ)  : {ca['correct_oe'] / ca['total']:.2%}")
    print(f"  Erreurs : {ca['total'] - ca['correct']:,}")

    print(f"\nRépartition erreurs :")
    for cat, cnt in global_cats.most_common():
        print(f"  {cat:35s} : {cnt:>5,} ({cnt/len(all_errors):.1%})")

    print(f"\nÉlisions : {n_elision_ok}/{n_elision_total} correctes")

    print(f"\nRapport : {report_path}")
    print(f"Corrections candidates : {corrections_path}")
    print(f"  ({len(corrections):,} entrées)")
    print(f"Temps : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
