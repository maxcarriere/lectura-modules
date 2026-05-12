#!/usr/bin/env python3
"""Analyze ACC false negatives from benchmark to find fixable patterns."""

import json
from pathlib import Path
from collections import defaultdict, Counter

# Prepositions that often require singular (de, en, par, etc.)
PREPOSITIONS = {"de", "d'", "en", "par", "pour", "sans", "avec", "sous", "sur", "dans"}

# Singular determiners
SINGULAR_DETS = {"le", "la", "l'", "un", "une", "ce", "cette", "cet", "mon", "ma", "ton", "ta", "son", "sa"}

# Plural determiners
PLURAL_DETS = {"les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "leurs", "quelques", "plusieurs", "certains", "certaines"}


def classify_edit_type(original: str, gold: str) -> str:
    """Classify the type of edit needed."""
    orig_lower = original.lower()
    gold_lower = gold.lower()

    # RM_S: remove plural s
    if orig_lower.endswith('s') and not gold_lower.endswith('s'):
        if orig_lower[:-1] == gold_lower:
            return "RM_S"

    # ADD_S: add plural s
    if not orig_lower.endswith('s') and gold_lower.endswith('s'):
        if orig_lower + 's' == gold_lower:
            return "ADD_S"

    # GENDER: likely gender change (e/es/a/as endings)
    if (orig_lower.endswith('e') and gold_lower.endswith('es')) or \
       (orig_lower.endswith('es') and gold_lower.endswith('e')) or \
       (orig_lower.endswith('a') and gold_lower.endswith('as')) or \
       (orig_lower.endswith('as') and gold_lower.endswith('a')):
        return "GENDER"

    return "OTHER"


def simple_tokenize(text: str) -> list[str]:
    """Very simple word tokenizer for analysis."""
    # Replace common punctuation with spaces
    for char in ',.;:!?()[]{}«»"\'':
        text = text.replace(char, ' ')
    return [w for w in text.split() if w]


def get_context_window(words: list, error_idx: int, window: int = 5):
    """Get context around error position."""
    start = max(0, error_idx - window)
    end = min(len(words), error_idx + window + 1)

    context_words = words[start:end]
    error_pos_in_context = error_idx - start

    return context_words, error_pos_in_context


def find_preceding_det(words: list, pos: int):
    """Find the preceding determiner if any."""
    for i in range(pos - 1, max(-1, pos - 3), -1):
        if i < 0:
            break
        word_lower = words[i].lower()
        if word_lower in SINGULAR_DETS or word_lower in PLURAL_DETS:
            return i, words[i]
    return None, None


def analyze_acc_fn():
    """Main analysis function."""
    benchmark_file = Path("/data/work/projets/lectura/workspace/Modules/Correcteur/benchmark/iterations/conj_fixes_v1.json")

    if not benchmark_file.exists():
        print(f"ERROR: Benchmark file not found: {benchmark_file}")
        return

    print("Loading benchmark JSON...")
    with open(benchmark_file) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} sentence results\n")

    # Collect ACC FN cases
    acc_fn_cases = []

    print("Processing ACC FN cases...")
    for i, sentence_result in enumerate(results):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(results)}...")

        if "erreurs_detail" not in sentence_result:
            continue

        # Check if this result has ACC FN (fn > 0 and has ACC error)
        has_acc_fn = sentence_result.get("fn", 0) > 0 and any(
            e.get("type") == "ACC" for e in sentence_result["erreurs_detail"]
        )

        if not has_acc_fn:
            continue

        sentence = sentence_result.get("fautif", "")  # Use fautif (original incorrect)

        # Simple tokenization
        words = simple_tokenize(sentence)

        for err in sentence_result["erreurs_detail"]:
            # FN is when sys == orig (not corrected)
            if err.get("type") == "ACC" and err.get("sys") == err.get("orig"):
                pos = err.get("pos", -1)
                # Find the word in our tokenized list
                # We'll search around the position for a match
                found_pos = -1
                orig_word = err.get("orig", "")
                for j in range(max(0, pos - 2), min(len(words), pos + 3)):
                    if words[j].lower() == orig_word.lower():
                        found_pos = j
                        break

                if found_pos == -1:
                    # Fallback: just use position if within bounds
                    if 0 <= pos < len(words):
                        found_pos = pos

                acc_fn_cases.append({
                    "sentence": sentence,
                    "words": words,
                    "original": orig_word,
                    "gold": err.get("gold", ""),
                    "position": found_pos,
                })

    print(f"\nFound {len(acc_fn_cases)} ACC FN cases\n")

    # Classify by edit type
    edit_type_stats = Counter()
    categorized_cases = defaultdict(list)

    for case in acc_fn_cases:
        edit_type = classify_edit_type(case["original"], case["gold"])
        edit_type_stats[edit_type] += 1
        categorized_cases[edit_type].append(case)

    print("=" * 80)
    print("EDIT TYPE DISTRIBUTION")
    print("=" * 80)
    for edit_type, count in edit_type_stats.most_common():
        print(f"{edit_type:15s}: {count:4d} cases ({count/len(acc_fn_cases)*100:.1f}%)")
    print()

    # Analyze RM_S cases
    print("=" * 80)
    print("RM_S ANALYSIS (Remove plural 's')")
    print("=" * 80)

    rm_s_with_det_mismatch = []
    rm_s_after_prep = []
    rm_s_other = []

    for case in categorized_cases["RM_S"]:
        words = case["words"]
        pos = case["position"]

        if pos < 0 or pos >= len(words):
            rm_s_other.append(case)
            continue

        # Get context
        context_words, error_pos = get_context_window(words, pos)
        case["context_words"] = context_words
        case["error_pos"] = error_pos

        # Check for preceding DET
        det_idx, det_word = find_preceding_det(words, pos)

        if det_idx is not None:
            case["det_word"] = det_word

            # Check if DET is singular
            if det_word.lower() in SINGULAR_DETS:
                rm_s_with_det_mismatch.append(case)
                continue

        # Check for preposition before
        if pos > 0 and words[pos - 1].lower() in PREPOSITIONS:
            case["prep"] = words[pos - 1]
            rm_s_after_prep.append(case)
            continue

        rm_s_other.append(case)

    print(f"With singular DET + plural NOM mismatch: {len(rm_s_with_det_mismatch)}")
    print(f"After preposition: {len(rm_s_after_prep)}")
    print(f"Other: {len(rm_s_other)}")
    print()

    # Analyze ADD_S cases
    print("=" * 80)
    print("ADD_S ANALYSIS (Add plural 's')")
    print("=" * 80)

    add_s_with_det_mismatch = []
    add_s_other = []

    for case in categorized_cases["ADD_S"]:
        words = case["words"]
        pos = case["position"]

        if pos < 0 or pos >= len(words):
            add_s_other.append(case)
            continue

        # Get context
        context_words, error_pos = get_context_window(words, pos)
        case["context_words"] = context_words
        case["error_pos"] = error_pos

        # Check for preceding DET
        det_idx, det_word = find_preceding_det(words, pos)

        if det_idx is not None:
            case["det_word"] = det_word

            # Check if DET is plural
            if det_word.lower() in PLURAL_DETS:
                add_s_with_det_mismatch.append(case)
                continue

        add_s_other.append(case)

    print(f"With plural DET + singular NOM mismatch: {len(add_s_with_det_mismatch)}")
    print(f"Other: {len(add_s_other)}")
    print()

    # Print samples
    def print_samples(cases, title, n=10):
        print("=" * 80)
        print(f"{title} (showing {min(n, len(cases))} of {len(cases)})")
        print("=" * 80)
        for i, case in enumerate(cases[:n], 1):
            print(f"\n{i}. {case['original']} → {case['gold']}")

            # Highlight the error position
            ctx_words = case.get("context_words", [])
            err_pos = case.get("error_pos", -1)

            if ctx_words and 0 <= err_pos < len(ctx_words):
                highlighted = []
                for j, w in enumerate(ctx_words):
                    if j == err_pos:
                        highlighted.append(f">>>{w}<<<")
                    else:
                        highlighted.append(w)
                print(f"   Context: {' '.join(highlighted)}")

            if "det_word" in case:
                print(f"   DET: {case['det_word']}")
            if "prep" in case:
                print(f"   PREP: {case['prep']}")
        print()

    print_samples(rm_s_with_det_mismatch, "RM_S: Singular DET + Plural NOM", n=15)
    print_samples(rm_s_after_prep, "RM_S: After Preposition", n=15)
    print_samples(add_s_with_det_mismatch, "ADD_S: Plural DET + Singular NOM", n=15)

    # Summary
    print("=" * 80)
    print("SUMMARY - RECOVERABLE PATTERNS")
    print("=" * 80)
    if len(acc_fn_cases) > 0:
        print(f"RM_S with DET mismatch:    {len(rm_s_with_det_mismatch):4d} ({len(rm_s_with_det_mismatch)/len(acc_fn_cases)*100:.1f}%)")
        print(f"RM_S after preposition:    {len(rm_s_after_prep):4d} ({len(rm_s_after_prep)/len(acc_fn_cases)*100:.1f}%)")
        print(f"ADD_S with DET mismatch:   {len(add_s_with_det_mismatch):4d} ({len(add_s_with_det_mismatch)/len(acc_fn_cases)*100:.1f}%)")
        print()
        total_recoverable = len(rm_s_with_det_mismatch) + len(rm_s_after_prep) + len(add_s_with_det_mismatch)
        print(f"Total recoverable:         {total_recoverable:4d}")
        print(f"Total ACC FN:              {len(acc_fn_cases):4d}")
        print(f"Recovery rate potential:   {total_recoverable/len(acc_fn_cases)*100:.1f}%")
    else:
        print("No ACC FN cases found!")


if __name__ == "__main__":
    analyze_acc_fn()
