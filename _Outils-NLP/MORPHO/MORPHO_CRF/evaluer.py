#!/usr/bin/env python3
"""Évaluation de l'analyseur morphologique Lectura (CRF) sur corpus UD.

Métriques :
  1. Accuracy tag complet (match exact du tag composite)
  2. Accuracy POS seul (partie POS uniquement)
  3. P/R/F1 par tag composite
  4. Accuracy par trait (genre, nombre, temps, mode, personne)
  5. Accuracy lemmatisation (vs colonne 3 CoNLL-U)

Usage :
    python evaluer.py --test chemin/vers/pos_test_merged.conllu
    python evaluer.py --test ../../../POS/POS_CRF/entrainement/donnees/pos_test_merged.conllu
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from lectura_morpho import MorphoTagger, _decompose_tag, _lemmatize_by_rules

# ── Mapping UD → tags projet ────────────────────────────────────────────────

_UPOS_MAP: dict[str, str] = {
    "NOUN": "NOM", "PROPN": "NOM", "VERB": "VER", "AUX": "AUX",
    "ADJ": "ADJ", "ADV": "ADV", "ADP": "PRE", "CCONJ": "CON",
    "SCONJ": "CON", "INTJ": "INTJ", "NUM": "NOM", "SYM": "NOM", "X": "NOM",
}

_FEATURE_MAP: list[tuple[str, str, str, str]] = [
    ("DET", "Definite", "Def", "ART:def"),
    ("DET", "Definite", "Ind", "ART:ind"),
    ("DET", "PronType", "Art", "ART:def"),
    ("DET", "Poss", "Yes", "ADJ:pos"),
    ("DET", "PronType", "Dem", "ADJ:dem"),
    ("DET", "PronType", "Int", "ADJ:int"),
    ("PRON", "PronType", "Prs", "PRO:per"),
    ("PRON", "PronType", "Rel", "PRO:rel"),
    ("PRON", "PronType", "Dem", "PRO:dem"),
    ("PRON", "Poss", "Yes", "PRO:pos"),
    ("PRON", "PronType", "Int", "PRO:int"),
    ("PRON", "PronType", "Ind", "PRO:ind"),
]

_UPOS_FALLBACK: dict[str, str] = {"DET": "ART:ind", "PRON": "PRO:per"}
_IGNORE_UPOS = {"PUNCT", "SPACE"}

_GENDER_MAP = {"Masc": "Masc", "Fem": "Fem"}
_NUMBER_MAP = {"Sing": "Sing", "Plur": "Plur"}
_TENSE_MAP = {"Pres": "Pres", "Imp": "Imp", "Past": "Past", "Fut": "Fut"}
_MOOD_MAP = {"Ind": "Ind", "Sub": "Sub", "Cnd": "Cnd", "Imp": "Imp"}
_PERSON_MAP = {"1": "1", "2": "2", "3": "3"}
_VERBFORM_MAP = {"Part": "Part", "Inf": "Inf", "Fin": "Fin", "Ger": "Ger"}


def _parse_ud_feats(feat_string: str) -> dict[str, str]:
    if not feat_string or feat_string == "_":
        return {}
    result: dict[str, str] = {}
    for part in feat_string.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def _ud_to_project_tag(upos: str, feats: dict[str, str] | None = None) -> str | None:
    if upos in _IGNORE_UPOS:
        return None
    if feats:
        for rule_upos, feat_key, feat_val, tag in _FEATURE_MAP:
            if upos == rule_upos and feats.get(feat_key) == feat_val:
                return tag
    if upos in _UPOS_FALLBACK:
        return _UPOS_FALLBACK[upos]
    return _UPOS_MAP.get(upos, "NOM")


def _build_composite_tag(base_pos: str, feats: dict[str, str]) -> str:
    """Même logique que dans entrainer_crf.py."""
    verbform = _VERBFORM_MAP.get(feats.get("VerbForm", ""), "")
    gender = _GENDER_MAP.get(feats.get("Gender", ""), "")
    number = _NUMBER_MAP.get(feats.get("Number", ""), "")
    tense = _TENSE_MAP.get(feats.get("Tense", ""), "")
    mood = _MOOD_MAP.get(feats.get("Mood", ""), "")
    person = _PERSON_MAP.get(feats.get("Person", ""), "")

    core_pos = base_pos.split(":")[0]

    if core_pos in ("VER", "AUX"):
        if verbform == "Inf":
            return f"{base_pos}|Inf"
        if verbform == "Ger":
            return f"{base_pos}|Ger"
        if verbform == "Part":
            parts = [base_pos, "Part"]
            if gender:
                parts.append(gender)
            if number:
                parts.append(number)
            return "|".join(parts)
        if mood:
            parts = [base_pos, mood]
            if tense:
                parts.append(tense)
            if person:
                parts.append(person)
            if number:
                parts.append(number)
            return "|".join(parts)
        parts = [base_pos]
        if number:
            parts.append(number)
        return "|".join(parts)

    if core_pos in ("NOM", "ADJ"):
        parts = [base_pos]
        if gender:
            parts.append(gender)
        if number:
            parts.append(number)
        return "|".join(parts)

    if ":" in base_pos:
        parts = [base_pos]
        if gender:
            parts.append(gender)
        if number:
            parts.append(number)
        if person:
            parts.append(person)
        return "|".join(parts)

    return base_pos


# ── Parsing CoNLL-U ─────────────────────────────────────────────────────────

def parse_conllu(path: Path) -> list[list[tuple[str, str, str, dict[str, str]]]]:
    """Parse un fichier CoNLL-U → [(word, lemma, composite_tag, feats), ...] par phrase.

    Applique le même mapping UD → composite que l'entraînement.
    """
    sentences: list[list[tuple[str, str, str, dict[str, str]]]] = []
    current: list[tuple[str, str, str, dict[str, str]]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            if "-" in fields[0] or "." in fields[0]:
                continue
            word = fields[1]
            lemma = fields[2]
            upos = fields[3]
            feats = _parse_ud_feats(fields[5])
            base_tag = _ud_to_project_tag(upos, feats)
            if base_tag is None:
                continue
            composite = _build_composite_tag(base_tag, feats)
            current.append((word, lemma, composite, feats))
    if current:
        sentences.append(current)
    return sentences


# ── Métriques ────────────────────────────────────────────────────────────────

def evaluate_tags(
    pred_all: list[list[str]],
    gold_all: list[list[str]],
) -> dict:
    """Calcule accuracy globale, POS-only accuracy, P/R/F1 par tag, confusion."""
    total = 0
    correct = 0
    pos_correct = 0
    tag_stats: dict[str, dict[str, int]] = {}
    confusion: Counter[tuple[str, str]] = Counter()

    for pred_seq, gold_seq in zip(pred_all, gold_all):
        for pred, gold in zip(pred_seq, gold_seq):
            total += 1
            if gold == pred:
                correct += 1

            # POS-only
            pred_pos = pred.split("|")[0]
            gold_pos = gold.split("|")[0]
            if pred_pos == gold_pos:
                pos_correct += 1

            for t in (gold, pred):
                if t not in tag_stats:
                    tag_stats[t] = {"tp": 0, "fp": 0, "fn": 0}

            if gold == pred:
                tag_stats[gold]["tp"] += 1
            else:
                tag_stats[gold]["fn"] += 1
                tag_stats[pred]["fp"] += 1
                confusion[(gold, pred)] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "pos_correct": pos_correct,
        "pos_accuracy": pos_correct / total if total else 0,
        "tag_stats": tag_stats,
        "confusion": confusion,
    }


def evaluate_traits(
    pred_tags: list[list[str]],
    gold_tags: list[list[str]],
) -> dict[str, dict[str, int]]:
    """Calcule l'accuracy par trait morphologique."""
    trait_stats: dict[str, dict[str, int]] = {}
    for trait_name in ("genre", "nombre", "temps", "mode", "personne"):
        trait_stats[trait_name] = {"total": 0, "correct": 0}

    for pred_seq, gold_seq in zip(pred_tags, gold_tags):
        for pred, gold in zip(pred_seq, gold_seq):
            pred_d = _decompose_tag(pred)
            gold_d = _decompose_tag(gold)

            for trait_name in ("genre", "nombre", "temps", "mode", "personne"):
                gold_val = gold_d[trait_name]
                if gold_val is not None:
                    trait_stats[trait_name]["total"] += 1
                    if pred_d[trait_name] == gold_val:
                        trait_stats[trait_name]["correct"] += 1

    return trait_stats


def evaluate_lemmas(
    pred_lemmas: list[list[str]],
    gold_lemmas: list[list[str]],
) -> dict:
    """Calcule l'accuracy de lemmatisation."""
    total = 0
    correct = 0
    for pred_seq, gold_seq in zip(pred_lemmas, gold_lemmas):
        for pred, gold in zip(pred_seq, gold_seq):
            total += 1
            if pred == gold.lower():
                correct += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
    }


# ── Affichage ────────────────────────────────────────────────────────────────

def print_results(name: str, results: dict, trait_stats: dict, lemma_stats: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {name}")
    print(f"{'=' * 78}")

    print(f"\n  Accuracy tag complet : {results['accuracy']:.2%}"
          f"  ({results['correct']}/{results['total']})")
    print(f"  Accuracy POS seul    : {results['pos_accuracy']:.2%}"
          f"  ({results['pos_correct']}/{results['total']})")
    print(f"  Accuracy lemmes      : {lemma_stats['accuracy']:.2%}"
          f"  ({lemma_stats['correct']}/{lemma_stats['total']})")

    # Trait accuracy
    print(f"\n  Accuracy par trait :")
    print(f"  {'Trait':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 42}")
    for trait_name in ("genre", "nombre", "temps", "mode", "personne"):
        ts = trait_stats[trait_name]
        if ts["total"] > 0:
            acc = ts["correct"] / ts["total"]
            print(f"  {trait_name:<12} {ts['correct']:>8} {ts['total']:>8} {acc:>9.2%}")

    # Per-tag table (top 40 by support)
    tag_stats = results["tag_stats"]
    tags_sorted = sorted(tag_stats.keys(),
                         key=lambda t: tag_stats[t]["tp"] + tag_stats[t]["fn"],
                         reverse=True)

    print(f"\n  {'Tag':<30} {'Prec':>7} {'Rappel':>7} {'F1':>7}"
          f"  {'Support':>8}")
    print(f"  {'-' * 65}")

    for tag in tags_sorted[:40]:
        s = tag_stats[tag]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        support = tp + fn
        print(f"  {tag:<30} {prec:>7.1%} {rec:>7.1%} {f1:>7.1%}"
              f"  {support:>8}")

    # Top confusions
    confusion = results["confusion"]
    if confusion:
        print(f"\n  Top 20 confusions :")
        print(f"  {'Vrai':<30} {'Prédit':<30} {'Nb':>6}")
        print(f"  {'-' * 70}")
        for (gold, pred), count in confusion.most_common(20):
            print(f"  {gold:<30} {pred:<30} {count:>6}")


# ── Export JSON ──────────────────────────────────────────────────────────────

def export_results(
    results: dict,
    trait_stats: dict,
    lemma_stats: dict,
    name: str,
    path: Path,
) -> None:
    tag_stats = results["tag_stats"]
    export = {
        "name": name,
        "accuracy_complete": round(results["accuracy"], 4),
        "accuracy_pos": round(results["pos_accuracy"], 4),
        "accuracy_lemma": round(lemma_stats["accuracy"], 4),
        "total_tokens": results["total"],
        "correct_tokens": results["correct"],
        "trait_accuracy": {},
        "per_tag": {},
    }

    for trait_name in ("genre", "nombre", "temps", "mode", "personne"):
        ts = trait_stats[trait_name]
        if ts["total"] > 0:
            export["trait_accuracy"][trait_name] = {
                "correct": ts["correct"],
                "total": ts["total"],
                "accuracy": round(ts["correct"] / ts["total"], 4),
            }

    for tag, s in sorted(tag_stats.items()):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        export["per_tag"][tag] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"\n  Résultats exportés : {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    here = Path(__file__).parent

    # Chercher le test set
    default_test = (here / ".." / ".." / ".." / "POS" / "POS_CRF" / "entrainement" / "donnees" / "pos_test_merged.conllu").resolve()

    parser = argparse.ArgumentParser(
        description="Évaluation de l'analyseur morphologique CRF Lectura"
    )
    parser.add_argument(
        "--test", default=str(default_test),
        help="Fichier CoNLL-U de test (ex: pos_test_merged.conllu)",
    )
    parser.add_argument(
        "--model", default=str(here / "modele" / "morpho_model_crf.json"),
        help="Chemin du modèle CRF JSON",
    )
    parser.add_argument(
        "--lexicon", default=str(here / "modele" / "glaff_lookup.json"),
        help="Chemin du lexique GLAFF (optionnel)",
    )
    parser.add_argument(
        "--mini-lexicon",
        default=str((here / ".." / ".." / ".." / "POS" / "POS_CRF" / "modele" / "mini_lexique.json").resolve()),
        help="Chemin du mini-lexique POS (optionnel)",
    )
    parser.add_argument(
        "--export", default=str(here / "EVALUATION.json"),
        help="Fichier JSON de sortie pour les résultats",
    )
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"ERREUR : fichier test introuvable : {test_path}", file=sys.stderr)
        sys.exit(1)

    # Charger le test set
    print("Chargement du test set...")
    sentences = parse_conllu(test_path)
    n_tokens = sum(len(s) for s in sentences)
    print(f"  {len(sentences)} phrases, {n_tokens} tokens (hors PUNCT/SPACE)")

    # Préparer les données gold
    gold_tags_all = [[ctag for _, _, ctag, _ in sent] for sent in sentences]
    gold_lemmas_all = [[lemma for _, lemma, _, _ in sent] for sent in sentences]
    words_all = [[word for word, _, _, _ in sent] for sent in sentences]

    # Charger le modèle pour connaître les tags valides
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERREUR : modèle introuvable : {model_path}", file=sys.stderr)
        sys.exit(1)

    with open(model_path, encoding="utf-8") as f:
        model_data = json.load(f)
    valid_tags = set(model_data.get("tags", []))

    # Replier les gold tags inconnus du modèle sur leur POS de base
    if valid_tags:
        for i, seq in enumerate(gold_tags_all):
            for j, tag in enumerate(seq):
                if tag not in valid_tags:
                    gold_tags_all[i][j] = tag.split("|")[0]

    # ── Config 1 : CRF seul ──
    print("\nChargement du modèle CRF morpho...")
    mini_lex = Path(args.mini_lexicon)
    mini_lex_path = mini_lex if mini_lex.exists() else None
    tagger_base = MorphoTagger(args.model, mini_lexicon_path=mini_lex_path)

    print("Évaluation CRF seul...")
    pred_tags_all = []
    pred_lemmas_all = []
    for words in words_all:
        results = tagger_base.tag_words(words)
        pred_tags_all.append([r["tag_complet"] for r in results])
        pred_lemmas_all.append([r["lemme"] for r in results])

    results_base = evaluate_tags(pred_tags_all, gold_tags_all)
    trait_stats_base = evaluate_traits(pred_tags_all, gold_tags_all)
    lemma_stats_base = evaluate_lemmas(pred_lemmas_all, gold_lemmas_all)
    print_results("CRF morpho (sans GLAFF)", results_base, trait_stats_base, lemma_stats_base)

    # ── Config 2 : CRF + GLAFF ──
    lexicon_path = Path(args.lexicon)
    if lexicon_path.exists():
        print("\nChargement du lexique GLAFF...")
        tagger_lex = MorphoTagger(
            args.model, lexicon_path=args.lexicon,
            mini_lexicon_path=mini_lex_path,
        )
        n_entries = len(tagger_lex.lexicon)
        print(f"  {n_entries} entrées chargées")

        print("Évaluation CRF + GLAFF...")
        pred_tags_lex = []
        pred_lemmas_lex = []
        for words in words_all:
            results = tagger_lex.tag_words(words)
            pred_tags_lex.append([r["tag_complet"] for r in results])
            pred_lemmas_lex.append([r["lemme"] for r in results])

        results_lex = evaluate_tags(pred_tags_lex, gold_tags_all)
        trait_stats_lex = evaluate_traits(pred_tags_lex, gold_tags_all)
        lemma_stats_lex = evaluate_lemmas(pred_lemmas_lex, gold_lemmas_all)
        print_results("CRF morpho + GLAFF", results_lex, trait_stats_lex, lemma_stats_lex)

        # Comparaison lemmes
        print(f"\n  Comparaison lemmatisation :")
        print(f"    Sans GLAFF : {lemma_stats_base['accuracy']:.2%}")
        print(f"    Avec GLAFF : {lemma_stats_lex['accuracy']:.2%}")

        export_results(results_lex, trait_stats_lex, lemma_stats_lex,
                       "CRF morpho + GLAFF", Path(args.export))
    else:
        print(f"\n  (lexique GLAFF non trouvé : {lexicon_path}, évaluation simple)")
        export_results(results_base, trait_stats_base, lemma_stats_base,
                       "CRF morpho", Path(args.export))


if __name__ == "__main__":
    main()
