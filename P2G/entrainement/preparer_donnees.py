#!/usr/bin/env python3
"""Prépare les données d'entraînement pour le modèle unifié P2G.

Sources :
- CoNLL-U enrichis (Phone dans MISC) → phrases avec toutes les tâches
- Lexique G2P CSV → mots isolés pour le pré-entraînement P2G

Sortie : données préparées en JSON dans entrainement/donnees/

Différences vs G2P complet :
- char2idx = caractères IPA (pas orthographiques)
- p2g_label2idx = graphèmes multi-car (pas phonèmes)
- Pas de liaison/denas
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Ajouter src/ au path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from lectura_p2g.utils.aligneur import aligner
from lectura_p2g.utils.p2g_labels import _CONT, labels_from_p2g_alignment, reconstruct_ortho
from lectura_p2g.utils.ipa import iter_phonemes

# ── Tags du projet ──────────────────────────────────────────────────

_UPOS_MAP = {
    "NOUN": "NOM", "PROPN": "NOM", "VERB": "VER", "AUX": "AUX",
    "ADJ": "ADJ", "ADV": "ADV", "ADP": "PRE", "CCONJ": "CON",
    "SCONJ": "CON", "INTJ": "INTJ", "NUM": "NOM", "SYM": "NOM", "X": "NOM",
}

_FEATURE_MAP = [
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

_UPOS_FALLBACK = {"DET": "ART:ind", "PRON": "PRO:per"}
_IGNORE_UPOS = {"PUNCT", "SPACE"}


def ud_to_project_tag(upos: str, feats: dict[str, str] | None) -> str | None:
    if upos in _IGNORE_UPOS:
        return None
    if feats:
        for rule_upos, feat_key, feat_val, tag in _FEATURE_MAP:
            if upos == rule_upos and feats.get(feat_key) == feat_val:
                return tag
    if upos in _UPOS_FALLBACK:
        return _UPOS_FALLBACK[upos]
    return _UPOS_MAP.get(upos, "NOM")


def parse_ud_feats(feat_string: str) -> dict[str, str]:
    if not feat_string or feat_string == "_":
        return {}
    result = {}
    for part in feat_string.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def parse_misc(misc_string: str) -> dict[str, str]:
    if not misc_string or misc_string == "_":
        return {}
    result = {}
    for part in misc_string.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


# ── Classes de données ──────────────────────────────────────────────

# Labels morpho (features factorisées)
MORPHO_FEATURES = {
    "Number": ["_", "Sing", "Plur"],
    "Gender": ["_", "Masc", "Fem"],
    "VerbForm": ["_", "Fin", "Inf", "Part", "Ger", "Conv"],
    "Mood": ["_", "Ind", "Sub", "Imp", "Cnd"],
    "Tense": ["_", "Pres", "Past", "Imp", "Fut"],
    "Person": ["_", "1", "2", "3"],
}


@dataclass
class TokenData:
    form: str
    pos_tag: str
    phone: str
    morpho: dict[str, str]


@dataclass
class SentenceData:
    sent_id: str
    text: str
    tokens: list[TokenData]


@dataclass
class WordP2GData:
    ipa: str
    labels: list[str]  # Labels graphèmes par car. IPA (_CONT ou graphème)


# ── Parsing CoNLL-U enrichi ────────────────────────────────────────

def parse_conllu(path: Path) -> list[SentenceData]:
    """Parse un fichier CoNLL-U enrichi avec Phone dans MISC."""
    sentences: list[SentenceData] = []
    current_tokens: list[TokenData] = []
    sent_id = ""
    text = ""

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                if current_tokens:
                    sentences.append(SentenceData(sent_id, text, current_tokens))
                    current_tokens = []
                    sent_id = ""
                    text = ""
                continue

            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text"):
                    text = line.split("=", 1)[1].strip()
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            tok_id = parts[0]
            if "-" in tok_id or "." in tok_id:
                continue

            form = parts[1]
            upos = parts[3]
            feats_str = parts[5]
            misc_str = parts[9]

            feats = parse_ud_feats(feats_str)
            pos_tag = ud_to_project_tag(upos, feats)
            if pos_tag is None:
                continue

            misc = parse_misc(misc_str)
            phone = misc.get("Phone", "")

            # Morpho features
            morpho = {}
            for feat_name in MORPHO_FEATURES:
                val = feats.get(feat_name, "_")
                if val not in MORPHO_FEATURES[feat_name]:
                    val = "_"
                morpho[feat_name] = val

            current_tokens.append(TokenData(
                form=form, pos_tag=pos_tag, phone=phone, morpho=morpho,
            ))

    if current_tokens:
        sentences.append(SentenceData(sent_id, text, current_tokens))

    return sentences


# ── Alignement P2G mots isolés ─────────────────────────────────────

def load_lexique_pairs(path: Path) -> list[tuple[str, str]]:
    """Charge les paires (ortho, phone) depuis un CSV."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ortho = row.get("ortho", "").strip().lower()
            phone = row.get("phone", "").strip()
            if not ortho or not phone:
                continue
            if " " in ortho or "'" in ortho or "\u2019" in ortho or "-" in ortho:
                continue
            key = (ortho, phone)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


def align_lexique_p2g(
    pairs: list[tuple[str, str]],
    phone_to_graphs: dict[str, list[str]],
    max_items: int = 0,
) -> list[WordP2GData]:
    """Aligne les paires et retourne les données P2G par mot IPA."""
    data: list[WordP2GData] = []
    failed = 0

    items = pairs[:max_items] if max_items > 0 else pairs

    total = len(items)
    for idx, (ortho, phone) in enumerate(items):
        if idx > 0 and idx % 10000 == 0:
            print(f"    ... {idx}/{total} ({len(data)} ok, {failed} échecs)",
                  flush=True)

        dec_ph, dec_gr, dec_spans, ok = aligner(ortho, phone, phone_to_graphs)
        if not ok:
            failed += 1
            continue

        # Generate P2G labels: graphème per IPA Unicode character
        labels = labels_from_p2g_alignment(phone, dec_ph, dec_gr)
        if len(labels) != len(phone):
            failed += 1
            continue

        # Validate: reconstructed ortho should match the aligned graphemes
        reconstructed = reconstruct_ortho(labels)
        expected = "".join(
            g.replace("°", "").replace("²", "") for g in dec_gr
        )
        if reconstructed != expected:
            failed += 1
            continue

        data.append(WordP2GData(ipa=phone, labels=labels))

    print(f"  Lexique P2G aligné : {len(data)} / {total} "
          f"(échecs : {failed})", flush=True)
    return data


# ── Construction des vocabulaires ──────────────────────────────────

def build_vocabs(
    sentences: list[SentenceData],
    lexique_data: list[WordP2GData],
) -> dict:
    """Construit tous les vocabulaires nécessaires au modèle."""

    # 1. Vocabulaire caractères IPA (pas orthographiques !)
    ipa_chars: set[str] = set()
    for sent in sentences:
        for tok in sent.tokens:
            if tok.phone:
                ipa_chars.update(tok.phone)
    for wd in lexique_data:
        ipa_chars.update(wd.ipa)

    char2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4}
    for ch in sorted(ipa_chars):
        if ch not in char2idx:
            char2idx[ch] = len(char2idx)

    # 2. Vocabulaire P2G labels (graphèmes multi-car)
    p2g_labels: set[str] = set()
    p2g_labels.add(_CONT)
    for wd in lexique_data:
        for lab in wd.labels:
            p2g_labels.add(lab)

    p2g_label2idx = {"<PAD>": 0}
    for lab in sorted(p2g_labels):
        p2g_label2idx[lab] = len(p2g_label2idx)

    # 3. Vocabulaire POS
    pos_tags: set[str] = set()
    for sent in sentences:
        for tok in sent.tokens:
            pos_tags.add(tok.pos_tag)

    pos2idx = {"<PAD>": 0}
    for tag in sorted(pos_tags):
        pos2idx[tag] = len(pos2idx)

    # 4. Vocabulaires morpho (un par feature)
    morpho_vocabs = {}
    for feat_name, values in MORPHO_FEATURES.items():
        feat2idx = {"<PAD>": 0}
        for v in values:
            feat2idx[v] = len(feat2idx)
        morpho_vocabs[feat_name] = feat2idx

    return {
        "char2idx": char2idx,
        "p2g_label2idx": p2g_label2idx,
        "pos2idx": pos2idx,
        "morpho_vocabs": morpho_vocabs,
    }


# ── Statistiques ───────────────────────────────────────────────────

def print_stats(sentences: list[SentenceData], label: str) -> None:
    n_tokens = sum(len(s.tokens) for s in sentences)
    n_with_phone = sum(
        1 for s in sentences for t in s.tokens if t.phone
    )
    pos_counts = Counter(
        t.pos_tag for s in sentences for t in s.tokens
    )

    print(f"\n── {label} ──")
    print(f"  Phrases : {len(sentences)}")
    print(f"  Tokens  : {n_tokens}")
    print(f"  Avec phone : {n_with_phone}")
    print(f"  POS top 10 : {dict(pos_counts.most_common(10))}")


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare les données du modèle P2G unifié")
    parser.add_argument(
        "--donnees", type=Path,
        default=Path(__file__).parent / "donnees",
        help="Répertoire des données sources",
    )
    parser.add_argument(
        "--sortie", type=Path,
        default=Path(__file__).parent / "donnees",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--max-lexique", type=int, default=0,
        help="Limite sur le lexique (0 = tout)",
    )
    args = parser.parse_args()

    donnees = args.donnees
    sortie = args.sortie
    sortie.mkdir(parents=True, exist_ok=True)

    # Charger phone_to_graphemes
    p2g_path = donnees / "phone_to_graphemes.csv"
    print(f"Chargement phone_to_graphemes : {p2g_path}")
    phone_to_graphs: dict[str, list[str]] = {}
    with open(p2g_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            phone = row[0].strip()
            graphies = [g.strip() for g in row[1].split(";") if g.strip()]
            if phone:
                phone_to_graphs[phone] = graphies
    print(f"  {len(phone_to_graphs)} entrées")

    # ── 1. CoNLL-U enrichis ──
    print("\n═══ CoNLL-U enrichis ═══")
    train_sents = parse_conllu(donnees / "train.conllu")
    dev_sents = parse_conllu(donnees / "dev.conllu")
    test_sents = parse_conllu(donnees / "test.conllu")

    print_stats(train_sents, "Train")
    print_stats(dev_sents, "Dev")
    print_stats(test_sents, "Test")

    # ── 2. Lexique P2G ──
    print("\n═══ Lexique P2G ═══")
    lex_train_path = donnees / "lexique_train.csv"
    if lex_train_path.exists():
        print(f"Chargement : {lex_train_path}")
        lex_pairs = load_lexique_pairs(lex_train_path)
        print(f"  {len(lex_pairs)} paires")

        print("Alignement lexique P2G...")
        t0 = time.time()
        lexique_data = align_lexique_p2g(lex_pairs, phone_to_graphs, args.max_lexique)
        print(f"  Temps : {time.time() - t0:.1f}s")
    else:
        print(f"  Pas de lexique trouvé à {lex_train_path}")
        lexique_data = []

    # Lexique eval
    lex_eval_path = donnees / "lexique_eval.csv"
    lexique_eval: list[WordP2GData] = []
    if lex_eval_path.exists():
        eval_pairs = load_lexique_pairs(lex_eval_path)
        print(f"\nLexique eval : {len(eval_pairs)} paires")
        lexique_eval = align_lexique_p2g(eval_pairs, phone_to_graphs)

    # ── 3. Vocabulaires ──
    print("\n═══ Construction des vocabulaires ═══")
    vocabs = build_vocabs(train_sents, lexique_data)

    for name, v in vocabs.items():
        if isinstance(v, dict) and name != "morpho_vocabs":
            print(f"  {name}: {len(v)} entrées")
    for feat_name, feat_vocab in vocabs["morpho_vocabs"].items():
        print(f"  morpho.{feat_name}: {len(feat_vocab)} entrées")

    # ── 4. Sauvegarder ──
    print("\n═══ Sauvegarde ═══")

    # phone_to_graphs (pour entrainer.py et evaluer.py)
    p2g_out = sortie / "phone_to_graphs.json"
    with open(p2g_out, "w", encoding="utf-8") as f:
        json.dump(phone_to_graphs, f, ensure_ascii=False, indent=1)
    print(f"  phone_to_graphs : {p2g_out}")

    # Vocabulaires
    vocab_path = sortie / "vocabs.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=1)
    print(f"  Vocabs : {vocab_path}")

    # Sentences (train/dev/test) en JSON (sans liaison/denas)
    def serialize_sentences(sents: list[SentenceData]) -> list[dict]:
        result = []
        for s in sents:
            result.append({
                "sent_id": s.sent_id,
                "text": s.text,
                "tokens": [
                    {
                        "form": t.form,
                        "pos_tag": t.pos_tag,
                        "phone": t.phone,
                        "morpho": t.morpho,
                    }
                    for t in s.tokens
                ],
            })
        return result

    for name, sents in [("train", train_sents), ("dev", dev_sents), ("test", test_sents)]:
        path = sortie / f"sentences_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialize_sentences(sents), f, ensure_ascii=False)
        print(f"  {name} sentences : {path} ({len(sents)} phrases)")

    # Lexique P2G
    if lexique_data:
        lex_path = sortie / "lexique_p2g_train.json"
        with open(lex_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"ipa": d.ipa, "labels": d.labels} for d in lexique_data],
                f, ensure_ascii=False,
            )
        print(f"  Lexique train : {lex_path} ({len(lexique_data)} mots)")

    if lexique_eval:
        lex_path = sortie / "lexique_p2g_eval.json"
        with open(lex_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"ipa": d.ipa, "labels": d.labels} for d in lexique_eval],
                f, ensure_ascii=False,
            )
        print(f"  Lexique eval : {lex_path} ({len(lexique_eval)} mots)")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
