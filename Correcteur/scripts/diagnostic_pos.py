#!/usr/bin/env python3
"""Diagnostic POS : identifie les mots avec POS suspects qui causent des FN/FP.

Pour chaque phrase du benchmark, tague via LexiqueTagger et compare
le POS avec les attentes des regles grammaticales.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
CORPUS_PATH = os.path.join(_PROJECT_ROOT, "data", "corpus", "corpus_10000.jsonl")


def main():
    from lectura_lexique import Lexique
    from lectura_correcteur._tagger_lexique import LexiqueTagger
    from lectura_correcteur._utils import LexiqueNormalise

    lexique = Lexique(LEXIQUE_DB)
    lex_norm = LexiqueNormalise(lexique)
    tagger = LexiqueTagger(lex_norm)

    # Load 500 phrases
    phrases = []
    with open(CORPUS_PATH) as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            phrases.append(json.loads(line))

    # POS distribution for key words
    word_pos_counts: dict[str, Counter] = defaultdict(Counter)
    # Mots grammaticaux frequents et leur POS assigne
    gram_words = {
        "il", "elle", "on", "ils", "elles", "je", "tu", "nous", "vous",
        "le", "la", "les", "un", "une", "des", "du", "au", "aux",
        "ce", "se", "son", "sa", "ses", "leur", "leurs",
        "et", "est", "a", "ou", "ont", "on", "sont",
        "qui", "que", "dont", "en", "y",
        "me", "te", "ne", "de",
        "mon", "ton", "ma", "ta", "mes", "tes", "nos", "vos",
    }

    # Track POS for all words
    all_word_pos: dict[str, Counter] = defaultdict(Counter)

    for p in phrases:
        fautif = p["fautif"]
        tokens = tagger.tokenize(fautif)
        words = [t for t, is_w in tokens if is_w]
        tags = tagger.tag_words(words)

        for w, t in zip(words, tags):
            pos = t.get("pos", "")
            low = w.lower()
            if low in gram_words:
                word_pos_counts[low][pos] += 1
            all_word_pos[low][pos] += 1

    # Report 1: POS distribution for grammatical words
    print("=" * 70)
    print("  POS distribution for key grammatical words")
    print("=" * 70)
    for w in sorted(gram_words):
        if w in word_pos_counts:
            total = sum(word_pos_counts[w].values())
            items = word_pos_counts[w].most_common(5)
            items_str = ", ".join(f"{pos}:{cnt}" for pos, cnt in items)
            print(f"  {w:12s} ({total:>4d}x) : {items_str}")

    # Report 2: Words that get NOM when they should be grammatical
    print()
    print("=" * 70)
    print("  Grammatical words tagged as NOM (suspect)")
    print("=" * 70)
    for w in sorted(gram_words):
        if w in word_pos_counts:
            nom_count = word_pos_counts[w].get("NOM", 0)
            total = sum(word_pos_counts[w].values())
            if nom_count > 0:
                pct = 100 * nom_count / total
                best_pos = word_pos_counts[w].most_common(1)[0]
                print(f"  {w:12s} : NOM={nom_count}/{total} ({pct:.0f}%)  best={best_pos[0]}:{best_pos[1]}")

    # Report 3: All cgrams returned by lexique for key words
    print()
    print("=" * 70)
    print("  Lexique v4 entries for key grammatical words")
    print("=" * 70)
    for w in sorted(gram_words):
        infos = lex_norm.info(w)
        if infos:
            entries = [(e.get("cgram", "?"), float(e.get("freq", 0))) for e in infos]
            entries.sort(key=lambda x: -x[1])
            items_str = ", ".join(f"{cg}:{f:.1f}" for cg, f in entries[:5])
            print(f"  {w:12s} : {items_str}")
        else:
            print(f"  {w:12s} : (not in lexique)")

    # Report 4: Short words (<=3 chars) tagged as NOM but with grammatical entries
    print()
    print("=" * 70)
    print("  Short words (<=3) tagged NOM but have gram entries in lexique")
    print("=" * 70)
    for w in sorted(all_word_pos.keys()):
        if len(w) > 3:
            continue
        nom_count = all_word_pos[w].get("NOM", 0)
        if nom_count == 0:
            continue
        total = sum(all_word_pos[w].values())
        infos = lex_norm.info(w)
        gram_entries = [e for e in infos if e.get("cgram", "") in (
            "ART:def", "ART:ind", "ART", "PRE", "CON", "PRO:per",
            "PRO:dem", "PRO:rel", "PRO:ind", "DET", "DET:dem",
            "ADJ:pos", "ADV",
        )]
        if gram_entries:
            best_gram = max(gram_entries, key=lambda e: float(e.get("freq", 0)))
            print(f"  {w:8s} NOM={nom_count}/{total}  best_gram={best_gram.get('cgram')}:{float(best_gram.get('freq', 0)):.1f}")

    # Report 5: Analyze FN from last benchmark
    print()
    print("=" * 70)
    print("  FN analysis: POS at error positions")
    print("=" * 70)
    results_path = os.path.join(_PROJECT_ROOT, "benchmark", "iterations", "baseline_pre_fix.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        fn_pos_by_type: dict[str, Counter] = defaultdict(Counter)
        fn_word_pos: dict[str, list] = defaultdict(list)

        for r in results:
            for e in r.get("erreurs_detail", []):
                if e.get("type") == "FP":
                    continue
                # This is a FN - find the POS at this position
                fautif = r["fautif"]
                tokens = tagger.tokenize(fautif)
                words = [t for t, is_w in tokens if is_w]
                tags = tagger.tag_words(words)
                pos_idx = e.get("pos", -1)
                if 0 <= pos_idx < len(tags):
                    pos = tags[pos_idx].get("pos", "?")
                    err_type = e.get("type", "?")
                    fn_pos_by_type[err_type][pos] += 1
                    orig_word = e.get("orig", "?")
                    gold_word = e.get("gold", "?")
                    fn_word_pos[f"{orig_word}->{gold_word}"].append(pos)

        for err_type in sorted(fn_pos_by_type.keys()):
            counts = fn_pos_by_type[err_type]
            total = sum(counts.values())
            items = counts.most_common(10)
            items_str = ", ".join(f"{pos}:{cnt}" for pos, cnt in items)
            print(f"  {err_type:<8s} ({total:>3d} FN) : {items_str}")

        # Top word->gold pairs with their POS
        print()
        print("  Top FN word pairs and their POS:")
        for pair, pos_list in sorted(fn_word_pos.items(), key=lambda x: -len(x[1]))[:30]:
            pos_counter = Counter(pos_list)
            pos_str = ", ".join(f"{p}:{c}" for p, c in pos_counter.most_common(3))
            print(f"    {len(pos_list):>3d}x  {pair:30s}  POS: {pos_str}")
    else:
        print("  (No baseline results found, run benchmark first)")


if __name__ == "__main__":
    main()
