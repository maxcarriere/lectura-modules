#!/usr/bin/env python3
"""Entraîne un modèle CRF morphologique et l'exporte en JSON — Script autonome.

Utilise sklearn-crfsuite pour l'entraînement et exporte au format JSON
compatible avec lectura_morpho.py (MORPHO_CRF).

Le modèle prédit un tag composite unique par token, encodant POS + Genre + Nombre
+ Temps + Mode + Personne. Exemples : VER|Ind|Pres|3|Plur, NOM|Masc|Sing, ADV.

Pré-requis :
    pip install sklearn-crfsuite

Usage :
    # Entraînement standard (données POS_CRF partagées)
    python entrainer_crf.py

    # Avec corpus personnalisé
    python entrainer_crf.py \\
        --corpus donnees/pos_train_merged.conllu \\
        --dev donnees/pos_dev_merged.conllu

    # Ajuster la régularisation
    python entrainer_crf.py --c1 0.3 --c2 0.1 --max-iter 200

    # Changer le seuil de repli / filtrage des poids
    python entrainer_crf.py --min-tag-count 10 --weight-threshold 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ── Mapping UD → tags projet (base POS) ────────────────────────────────────

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

# Traits morphologiques — mapping UD → abréviations
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
    """Convertit UPOS → tag projet de base (POS seul)."""
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
    """Construit un tag composite POS|trait1|trait2|... depuis le POS et les traits UD.

    Règles de composition :
    - VER/AUX fini   : POS|Mood|Tense|Person|Number  (ex: VER|Ind|Pres|3|Plur)
    - VER/AUX part.  : POS|Part|Gender|Number         (ex: VER|Part|Masc|Sing)
    - VER/AUX inf.   : POS|Inf                        (ex: VER|Inf)
    - VER/AUX gér.   : POS|Ger                        (ex: VER|Ger)
    - NOM             : POS[|Gender][|Number]          (ex: NOM|Masc|Plur)
    - ADJ             : POS[|Gender][|Number]          (ex: ADJ|Fem|Sing)
    - ART:*/ADJ:*/PRO:* : POS[|Gender][|Number][|Person] (ex: PRO:per|Masc|Sing|3)
    - Invariables     : POS seul                       (ex: PRE, ADV, CON)
    """
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

    if ":" in base_pos:  # ART:def, PRO:per, ADJ:pos, etc.
        parts = [base_pos]
        if gender:
            parts.append(gender)
        if number:
            parts.append(number)
        if person:
            parts.append(person)
        return "|".join(parts)

    # Invariables: PRE, CON, ADV, INTJ
    return base_pos


# ── Parsing CoNLL-U ─────────────────────────────────────────────────────────

def parse_conllu(path: str) -> list[list[tuple[str, str, str, dict[str, str]]]]:
    """Parse un fichier CoNLL-U → [(word, lemma, upos, feats), ...] par phrase."""
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
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue
            word = fields[1]
            lemma = fields[2]
            upos = fields[3]
            feats = _parse_ud_feats(fields[5])
            current.append((word, lemma, upos, feats))

    if current:
        sentences.append(current)
    return sentences


def convert_sentences(
    raw_sentences: list[list[tuple[str, str, str, dict[str, str]]]],
    min_tag_count: int = 5,
    tag_counts: Counter | None = None,
) -> tuple[list[tuple[list[str], list[str], list[str]]], Counter]:
    """Convertit en (words, composite_tags, lemmas).

    Si tag_counts est fourni, utilise ces comptages pour le repli des tags rares.
    Sinon, les calcule depuis les données.
    """
    if tag_counts is None:
        tag_counts = Counter()
        for sent in raw_sentences:
            for word, lemma, upos, feats in sent:
                base_pos = _ud_to_project_tag(upos, feats)
                if base_pos is None:
                    continue
                composite = _build_composite_tag(base_pos, feats)
                tag_counts[composite] += 1

    valid_tags = {tag for tag, count in tag_counts.items() if count >= min_tag_count}

    result = []
    for sent in raw_sentences:
        words: list[str] = []
        tags: list[str] = []
        lemmas: list[str] = []
        for word, lemma, upos, feats in sent:
            base_pos = _ud_to_project_tag(upos, feats)
            if base_pos is None:
                continue
            composite = _build_composite_tag(base_pos, feats)
            if composite not in valid_tags:
                composite = base_pos
            words.append(word)
            tags.append(composite)
            lemmas.append(lemma)
        if words:
            result.append((words, tags, lemmas))
    return result, tag_counts


# ── Extraction de features ──────────────────────────────────────────────────

def extract_word_features(
    words: list[str], idx: int,
) -> dict[str, str]:
    """Extrait les features CRF pour le mot à la position idx.

    Features identiques à POS_CRF + suf4/suf5 pour capter les terminaisons
    verbales longues (-aient, -ions, -ement).
    """
    word = words[idx]
    low = word.lower()
    feats: dict[str, str] = {
        "bias": "1.0",
        "word": low,
        "len": str(float(len(low))),
        "suf2": low[-2:] if len(low) >= 2 else low,
        "suf3": low[-3:] if len(low) >= 3 else low,
        "suf4": low[-4:] if len(low) >= 4 else low,
        "suf5": low[-5:] if len(low) >= 5 else low,
        "pre2": low[:2] if len(low) >= 2 else low,
        "pre3": low[:3] if len(low) >= 3 else low,
        "is_upper": "1.0" if word.isupper() else "0.0",
        "is_title": "1.0" if word.istitle() else "0.0",
        "is_digit": "1.0" if word.isdigit() else "0.0",
    }

    feats["BOS"] = "1.0" if idx == 0 else "0.0"
    feats["EOS"] = "1.0" if idx == len(words) - 1 else "0.0"
    feats["w-1"] = words[idx - 1].lower() if idx > 0 else "__BOS__"
    feats["w+1"] = words[idx + 1].lower() if idx < len(words) - 1 else "__EOS__"

    return feats


def sentences_to_crfsuite_data(
    sentences: list[tuple[list[str], list[str], list[str]]],
) -> tuple[list[list[dict]], list[list[str]]]:
    """Convertit en format sklearn-crfsuite (X, y)."""
    X: list[list[dict]] = []
    y: list[list[str]] = []

    for words, tags, _lemmas in sentences:
        feats = [extract_word_features(words, i) for i in range(len(words))]
        X.append(feats)
        y.append(tags)

    return X, y


# ── Export du modèle ─────────────────────────────────────────────────────────

def export_crfsuite_model(
    crf_model, tags: list[str], output_path: str,
    weight_threshold: float = 1e-4,
) -> None:
    """Exporte un modèle sklearn-crfsuite vers le format JSON Lectura.

    Le seuil weight_threshold filtre les poids négligeables pour contenir
    la taille du modèle (~3-5 Mo au lieu de ~15+ Mo).
    """
    state_features: dict[str, dict[str, float]] = {}
    n_total = 0
    n_kept = 0
    for (attr, label), weight in crf_model.state_features_.items():
        n_total += 1
        if abs(weight) < weight_threshold:
            continue
        n_kept += 1
        if attr not in state_features:
            state_features[attr] = {}
        state_features[attr][label] = round(weight, 6)

    transitions: dict[str, dict[str, float]] = {}
    for (label_from, label_to), weight in crf_model.transition_features_.items():
        if abs(weight) < weight_threshold:
            continue
        if label_from not in transitions:
            transitions[label_from] = {}
        transitions[label_from][label_to] = round(weight, 6)

    model_data = {
        "state_features": state_features,
        "transitions": transitions,
        "tags": tags,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=1)

    size_kb = output.stat().st_size / 1024
    print(f"Modèle sauvegardé : {output} ({size_kb:.0f} Ko)")
    print(f"  State features : {n_kept}/{n_total} conservés (seuil={weight_threshold})")
    print(f"  Transitions : {len(transitions)} entrées")


# ── Évaluation ───────────────────────────────────────────────────────────────

def evaluate(crf_model, X_test, y_test) -> dict[str, float]:
    """Évalue le modèle sur un jeu de test (accuracy complète + POS seul)."""
    y_pred = crf_model.predict(X_test)

    total = 0
    correct = 0
    pos_correct = 0
    tag_stats: dict[str, dict[str, int]] = {}

    for pred_seq, gold_seq in zip(y_pred, y_test):
        for p, g in zip(pred_seq, gold_seq):
            total += 1
            if p == g:
                correct += 1
            # POS-only
            p_pos = p.split("|")[0]
            g_pos = g.split("|")[0]
            if p_pos == g_pos:
                pos_correct += 1
            if g not in tag_stats:
                tag_stats[g] = {"total": 0, "correct": 0}
            tag_stats[g]["total"] += 1
            if p == g:
                tag_stats[g]["correct"] += 1

    acc = correct / total if total else 0
    pos_acc = pos_correct / total if total else 0
    print(f"  Accuracy tag complet : {acc:.2%} ({correct}/{total})")
    print(f"  Accuracy POS seul    : {pos_acc:.2%} ({pos_correct}/{total})")

    # Top 20 tags by support
    sorted_tags = sorted(tag_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    print("  Par tag (top 20) :")
    for tag, stats in sorted_tags[:20]:
        tag_acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"    {tag:30s} : {tag_acc:.1%} ({stats['correct']}/{stats['total']})")

    return {"accuracy": acc, "pos_accuracy": pos_acc, "total": total}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    here = Path(__file__).parent
    default_corpus = here / "donnees" / "pos_train_merged.conllu"
    default_dev = here / "donnees" / "pos_dev_merged.conllu"

    # Fallback : chercher dans POS_CRF
    if not default_corpus.exists():
        crf_donnees = here.parent.parent.parent / "POS" / "POS_CRF" / "entrainement" / "donnees"
        if (crf_donnees / "pos_train_merged.conllu").exists():
            default_corpus = crf_donnees / "pos_train_merged.conllu"
            default_dev = crf_donnees / "pos_dev_merged.conllu"

    parser = argparse.ArgumentParser(
        description="Entraîne un modèle CRF morphologique pour le français"
    )
    parser.add_argument("--corpus", type=Path, default=default_corpus,
                        help="Fichier CoNLL-U d'entraînement")
    parser.add_argument("--dev", type=Path, default=default_dev,
                        help="Fichier CoNLL-U de validation")
    parser.add_argument("--extra",
                        help="Fichier CoNLL-U additionnel (ajouté au corpus)")
    parser.add_argument("--output",
                        default=str(here / ".." / "modele" / "morpho_model_crf.json"),
                        help="Fichier de sortie du modèle JSON")
    parser.add_argument("--c1", type=float, default=0.3,
                        help="Régularisation L1 (défaut: 0.3)")
    parser.add_argument("--c2", type=float, default=0.1,
                        help="Régularisation L2 (défaut: 0.1)")
    parser.add_argument("--max-iter", type=int, default=150,
                        help="Itérations max (défaut: 150)")
    parser.add_argument("--min-tag-count", type=int, default=5,
                        help="Seuil minimum d'occurrences pour garder un tag composite")
    parser.add_argument("--weight-threshold", type=float, default=1e-4,
                        help="Seuil de filtrage des poids JSON (défaut: 1e-4)")
    args = parser.parse_args()

    try:
        import sklearn_crfsuite
    except ImportError:
        print("ERREUR : sklearn-crfsuite requis.")
        print("  pip install sklearn-crfsuite")
        sys.exit(1)

    # ── Parsing du corpus ──
    print(f"Parsing du corpus : {args.corpus}")
    if not args.corpus.exists():
        print(f"ERREUR : fichier introuvable : {args.corpus}")
        sys.exit(1)
    raw_train = parse_conllu(str(args.corpus))
    train_data, tag_counts = convert_sentences(
        raw_train, min_tag_count=args.min_tag_count
    )
    print(f"  {len(train_data)} phrases")

    valid_tags = {t for t, c in tag_counts.items() if c >= args.min_tag_count}
    folded_tags = {t for t, c in tag_counts.items() if c < args.min_tag_count}
    print(f"  Tags composites : {len(valid_tags)} retenus, {len(folded_tags)} repliés "
          f"(seuil={args.min_tag_count})")

    # Corpus additionnel
    if args.extra:
        extra_path = Path(args.extra)
        if extra_path.exists():
            print(f"Parsing du corpus additionnel : {args.extra}")
            raw_extra = parse_conllu(args.extra)
            extra_data, _ = convert_sentences(
                raw_extra, min_tag_count=args.min_tag_count, tag_counts=tag_counts
            )
            print(f"  {len(extra_data)} phrases ajoutées")
            train_data.extend(extra_data)
        else:
            print(f"ATTENTION : corpus additionnel introuvable : {args.extra}")

    X_train, y_train = sentences_to_crfsuite_data(train_data)

    all_tags = sorted(set(tag for tags in y_train for tag in tags))
    print(f"  Total : {len(train_data)} phrases, {len(all_tags)} tags")

    # ── Entraînement ──
    print(f"\nEntraînement CRF (c1={args.c1}, c2={args.c2}, "
          f"max_iter={args.max_iter})...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iter,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    print("\n--- Évaluation sur train ---")
    evaluate(crf, X_train, y_train)

    # ── Évaluation sur dev ──
    if args.dev.exists():
        print(f"\nParsing dev : {args.dev}")
        raw_dev = parse_conllu(str(args.dev))
        dev_data, _ = convert_sentences(
            raw_dev, min_tag_count=args.min_tag_count, tag_counts=tag_counts
        )
        X_dev, y_dev = sentences_to_crfsuite_data(dev_data)
        print(f"  {len(dev_data)} phrases")
        print("\n--- Évaluation sur dev ---")
        evaluate(crf, X_dev, y_dev)

    # ── Export ──
    print(f"\nExport vers {args.output}...")
    export_crfsuite_model(crf, all_tags, args.output,
                          weight_threshold=args.weight_threshold)
    print("Terminé.")


if __name__ == "__main__":
    main()
