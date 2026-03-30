#!/usr/bin/env python3
"""Prépare les données d'entraînement pour le modèle unifié.

Sources :
- CoNLL-U enrichis (Phone, Liaison, Denas dans MISC) → phrases avec toutes les tâches
- Lexique G2P CSV → mots isolés pour le pré-entraînement G2P

Sortie : données préparées en JSON dans entrainement/donnees/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Ajouter src/ au path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from lectura_nlp.utils.aligneur import aligner
from lectura_nlp.utils.g2p_labels import _CONT, labels_from_alignment, reconstruct_ipa
from lectura_nlp.utils.ipa import iter_phonemes

# ── Tags du projet ──────────────────────────────────────────────────

# Mapping UD → projet (même logique que ud_tag_mapping.py)
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
    """Parse la colonne MISC du CoNLL-U."""
    if not misc_string or misc_string == "_":
        return {}
    result = {}
    for part in misc_string.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


# ── Classes de données ──────────────────────────────────────────────

# Labels liaison
LIAISON_LABELS = ["none", "Lz", "Lt", "Ln", "Lr", "Lp"]
LIAISON_MAP = {"(z)": "Lz", "(t)": "Lt", "(n)": "Ln", "(r)": "Lr", "(p)": "Lp"}

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
    pos_tag: str  # Tag projet (ART:def, NOM, VER, ...)
    phone: str  # IPA
    liaison: str  # none / Lz / Lt / Ln / Lr / Lp
    denas: str  # ex: "ɔ̃>ɔ" ou ""
    morpho: dict[str, str]  # {Number: Sing, Gender: Masc, ...}


@dataclass
class SentenceData:
    sent_id: str
    text: str
    tokens: list[TokenData]


@dataclass
class WordG2PData:
    word: str
    labels: list[str]  # Labels par caractère (_CONT ou phonème)


# ── Parsing CoNLL-U enrichi ────────────────────────────────────────

def parse_conllu(path: Path) -> list[SentenceData]:
    """Parse un fichier CoNLL-U enrichi avec Phone/Liaison/Denas dans MISC."""
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

            # Ignorer les lignes de contraction (8-9)
            tok_id = parts[0]
            if "-" in tok_id or "." in tok_id:
                continue

            form = parts[1]
            upos = parts[3]
            feats_str = parts[5]
            misc_str = parts[9]

            # POS tag projet
            feats = parse_ud_feats(feats_str)
            pos_tag = ud_to_project_tag(upos, feats)
            if pos_tag is None:
                continue  # Ignorer PUNCT, SPACE

            # Parse MISC pour Phone, Liaison, Denas
            misc = parse_misc(misc_str)
            phone = misc.get("Phone", "")
            liaison_raw = misc.get("Liaison", "")
            denas = misc.get("Denas", "")

            # Convertir liaison
            liaison = LIAISON_MAP.get(liaison_raw, "none")

            # Morpho features
            morpho = {}
            for feat_name in MORPHO_FEATURES:
                val = feats.get(feat_name, "_")
                if val not in MORPHO_FEATURES[feat_name]:
                    val = "_"
                morpho[feat_name] = val

            current_tokens.append(TokenData(
                form=form, pos_tag=pos_tag, phone=phone,
                liaison=liaison, denas=denas, morpho=morpho,
            ))

    if current_tokens:
        sentences.append(SentenceData(sent_id, text, current_tokens))

    return sentences


# ── Alignement G2P mots isolés ─────────────────────────────────────

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


def align_lexique(
    pairs: list[tuple[str, str]],
    phone_to_graphs: dict[str, list[str]],
    max_items: int = 0,
) -> list[WordG2PData]:
    """Aligne les paires lexique et retourne les données G2P par mot."""
    data: list[WordG2PData] = []
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

        labels = labels_from_alignment(ortho, dec_ph, dec_spans)
        if len(labels) != len(ortho):
            failed += 1
            continue

        reconstructed = reconstruct_ipa(labels)
        expected = "".join(dec_ph)
        if reconstructed != expected:
            failed += 1
            continue

        data.append(WordG2PData(word=ortho, labels=labels))

    print(f"  Lexique aligné : {len(data)} / {total} "
          f"(échecs : {failed})", flush=True)
    return data


def align_sentence_tokens(
    sentences: list[SentenceData],
    phone_to_graphs: dict[str, list[str]],
) -> tuple[list[SentenceData], int, int]:
    """Aligne les phones des tokens de phrase et filtre ceux sans phone valide."""
    aligned = 0
    failed = 0

    for sent in sentences:
        for tok in sent.tokens:
            if not tok.phone:
                continue
            form_lower = tok.form.lower()
            # Ignorer les formes avec caractères spéciaux
            if any(c in form_lower for c in " '-\u2019"):
                continue

            dec_ph, dec_gr, dec_spans, ok = aligner(
                form_lower, tok.phone, phone_to_graphs
            )
            if ok:
                labels = labels_from_alignment(form_lower, dec_ph, dec_spans)
                if len(labels) == len(form_lower):
                    reconstructed = reconstruct_ipa(labels)
                    expected = "".join(dec_ph)
                    if reconstructed == expected:
                        aligned += 1
                        continue
            failed += 1

    return sentences, aligned, failed


# ── Construction des vocabulaires ──────────────────────────────────

def build_vocabs(
    sentences: list[SentenceData],
    lexique_data: list[WordG2PData],
) -> dict:
    """Construit tous les vocabulaires nécessaires au modèle."""

    # 1. Vocabulaire caractères
    chars: set[str] = set()
    for sent in sentences:
        for tok in sent.tokens:
            chars.update(tok.form.lower())
    for wd in lexique_data:
        chars.update(wd.word)

    char2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4}
    for ch in sorted(chars):
        if ch not in char2idx:
            char2idx[ch] = len(char2idx)

    # 2. Vocabulaire G2P labels
    g2p_labels: set[str] = set()
    g2p_labels.add(_CONT)
    for sent in sentences:
        for tok in sent.tokens:
            if tok.phone:
                for ph in iter_phonemes(tok.phone):
                    g2p_labels.add(ph)
    for wd in lexique_data:
        for lab in wd.labels:
            g2p_labels.add(lab)

    g2p_label2idx = {"<PAD>": 0}
    for lab in sorted(g2p_labels):
        g2p_label2idx[lab] = len(g2p_label2idx)

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

    # 5. Vocabulaire liaison
    liaison2idx = {"<PAD>": 0}
    for lab in LIAISON_LABELS:
        liaison2idx[lab] = len(liaison2idx)

    return {
        "char2idx": char2idx,
        "g2p_label2idx": g2p_label2idx,
        "pos2idx": pos2idx,
        "morpho_vocabs": morpho_vocabs,
        "liaison2idx": liaison2idx,
    }


# ── Statistiques ───────────────────────────────────────────────────

def print_stats(sentences: list[SentenceData], label: str) -> None:
    n_tokens = sum(len(s.tokens) for s in sentences)
    n_with_phone = sum(
        1 for s in sentences for t in s.tokens if t.phone
    )
    liaison_counts = Counter(
        t.liaison for s in sentences for t in s.tokens
    )
    pos_counts = Counter(
        t.pos_tag for s in sentences for t in s.tokens
    )

    print(f"\n── {label} ──")
    print(f"  Phrases : {len(sentences)}")
    print(f"  Tokens  : {n_tokens}")
    print(f"  Avec phone : {n_with_phone}")
    print(f"  Liaison : {dict(liaison_counts.most_common())}")
    print(f"  POS top 10 : {dict(pos_counts.most_common(10))}")


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare les données du modèle unifié")
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

    # Vérifier l'alignement G2P sur un échantillon (le corpus complet est trop lent)
    print("\nVérification alignement G2P (échantillon 500 phrases)...", flush=True)
    t0 = time.time()
    sample = train_sents[:500]
    _, al_ok, al_fail = align_sentence_tokens(sample, phone_to_graphs)
    total = al_ok + al_fail
    pct = al_ok / total * 100 if total else 0
    print(f"  Alignés : {al_ok}, échecs : {al_fail} "
          f"({pct:.1f}%) en {time.time() - t0:.1f}s")

    # ── 2. Lexique G2P ──
    print("\n═══ Lexique G2P ═══")
    lex_train_path = donnees / "lexique_train.csv"
    if lex_train_path.exists():
        print(f"Chargement : {lex_train_path}")
        lex_pairs = load_lexique_pairs(lex_train_path)
        print(f"  {len(lex_pairs)} paires")

        print("Alignement lexique...")
        t0 = time.time()
        lexique_data = align_lexique(lex_pairs, phone_to_graphs, args.max_lexique)
        print(f"  Temps : {time.time() - t0:.1f}s")
    else:
        print(f"  Pas de lexique trouvé à {lex_train_path}")
        lexique_data = []

    # Lexique eval
    lex_eval_path = donnees / "lexique_eval.csv"
    lexique_eval: list[WordG2PData] = []
    if lex_eval_path.exists():
        eval_pairs = load_lexique_pairs(lex_eval_path)
        print(f"\nLexique eval : {len(eval_pairs)} paires")
        lexique_eval = align_lexique(eval_pairs, phone_to_graphs)

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

    # Sentences (train/dev/test) en JSON
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
                        "liaison": t.liaison,
                        "denas": t.denas,
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

    # Lexique G2P
    if lexique_data:
        lex_path = sortie / "lexique_g2p_train.json"
        with open(lex_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"word": d.word, "labels": d.labels} for d in lexique_data],
                f, ensure_ascii=False,
            )
        print(f"  Lexique train : {lex_path} ({len(lexique_data)} mots)")

    if lexique_eval:
        lex_path = sortie / "lexique_g2p_eval.json"
        with open(lex_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"word": d.word, "labels": d.labels} for d in lexique_eval],
                f, ensure_ascii=False,
            )
        print(f"  Lexique eval : {lex_path} ({len(lexique_eval)} mots)")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
