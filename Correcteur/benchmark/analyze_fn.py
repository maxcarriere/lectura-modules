#!/usr/bin/env python3
"""Analyze False Negatives and False Positives across iter6i evaluation samples."""

import json
from collections import Counter, defaultdict
from pathlib import Path

FILES = [
    Path("/data/work/projets/lectura/workspace/Modules/Correcteur/benchmark/iterations/iter6i_s1.json"),
    Path("/data/work/projets/lectura/workspace/Modules/Correcteur/benchmark/iterations/iter6i_s2.json"),
    Path("/data/work/projets/lectura/workspace/Modules/Correcteur/benchmark/iterations/iter6i_s3.json"),
    Path("/data/work/projets/lectura/workspace/Modules/Correcteur/benchmark/iterations/iter6i_s4.json"),
]

def load_all():
    results = []
    for f in FILES:
        with open(f) as fh:
            data = json.load(fh)
            for item in data:
                item["_source"] = f.stem
            results.extend(data)
    return results


def main():
    results = load_all()
    print(f"Total sentences loaded: {len(results)}")

    # Collect all FN and FP entries
    all_fn = []  # list of (error_detail, item)
    all_fp = []  # list of (error_detail, item)

    for item in results:
        for err in item.get("erreurs_detail", []):
            if err["type"] == "FP":
                all_fp.append((err, item))
            else:
                # It's an expected correction. Check if system got it right.
                sys_val = (err.get("sys") or "").rstrip(".,;:!?").lower()
                gold_val = (err.get("gold") or "").lower()
                if sys_val != gold_val:
                    # FN: system did NOT produce the correct gold
                    all_fn.append((err, item))
                # else: TP

    print(f"\nTotal FN: {len(all_fn)}")
    print(f"Total FP: {len(all_fp)}")

    # ====================================================================
    # 1. TOP 30 most frequent FN patterns (orig -> gold)
    # ====================================================================
    fn_pattern_counter = Counter()
    fn_by_pattern = defaultdict(list)  # (orig, gold) -> list of (err, item)
    fn_type_counter = Counter()
    fn_by_type = defaultdict(list)  # type -> list of (err, item)

    for err, item in all_fn:
        orig = err.get("orig") or ""
        gold = err.get("gold") or ""
        pattern = (orig, gold)
        fn_pattern_counter[pattern] += 1
        fn_by_pattern[pattern].append((err, item))
        fn_type_counter[err.get("type", "?")] += 1
        fn_by_type[err.get("type", "?")].append((err, item))

    print("\n" + "=" * 80)
    print("1. TOP 30 MOST FREQUENT FN PATTERNS (orig -> gold) [what system should fix]")
    print("=" * 80)
    for i, ((orig, gold), count) in enumerate(fn_pattern_counter.most_common(30), 1):
        # Find the type(s)
        types_for_pattern = set()
        sys_outputs = set()
        for err, item in fn_by_pattern[(orig, gold)]:
            types_for_pattern.add(err["type"])
            sys_outputs.add(err["sys"])
        type_str = "/".join(sorted(types_for_pattern))
        sys_str = ", ".join(sorted(sys_outputs)[:3])
        print(f"  {i:2d}. {orig:25s} -> {gold:25s} [{type_str:8s}] x{count:3d}  (sys: {sys_str})")

    # ====================================================================
    # 2. FN grouped by type
    # ====================================================================
    print("\n" + "=" * 80)
    print("2. FN GROUPED BY TYPE")
    print("=" * 80)
    for typ, count in fn_type_counter.most_common():
        pct = count / len(all_fn) * 100
        print(f"\n  {typ:10s}: {count:4d} FN ({pct:.1f}%)")
        # Show top patterns for this type
        pattern_counts = Counter()
        for err, item in fn_by_type[typ]:
            pattern_counts[(err.get("orig") or "", err.get("gold") or "")] += 1
        for (orig, gold), c in pattern_counts.most_common(8):
            print(f"      {orig:25s} -> {gold:25s}  x{c}")

    # ====================================================================
    # 3. Top 10 HOMO FN patterns with 2-3 example sentences
    # ====================================================================
    print("\n" + "=" * 80)
    print("3. TOP 10 HOMO FN PATTERNS WITH EXAMPLE SENTENCES")
    print("=" * 80)
    homo_patterns = Counter()
    homo_by_pattern = defaultdict(list)
    for err, item in fn_by_type.get("HOMO", []):
        pattern = (err["orig"], err["gold"])
        homo_patterns[pattern] += 1
        homo_by_pattern[pattern].append((err, item))

    for i, ((orig, gold), count) in enumerate(homo_patterns.most_common(10), 1):
        print(f"\n  {i}. '{orig}' -> '{gold}' (x{count})")
        examples = homo_by_pattern[(orig, gold)]
        for j, (err, item) in enumerate(examples[:3]):
            print(f"     [{item['_source']}] Fautif:  {item['fautif']}")
            print(f"     [{item['_source']}] Correct: {item['correct']}")
            print(f"     [{item['_source']}] Obtenu:  {item['obtenu']}")
            print(f"     [{item['_source']}] sys='{err['sys']}'")
            print()

    # ====================================================================
    # 4. Top 5 "est" -> "et" FN cases with context + POS analysis
    # ====================================================================
    print("\n" + "=" * 80)
    print("4. TOP 5 'est' -> 'et' FN CASES WITH CONTEXT + POS")
    print("=" * 80)
    est_et_cases = homo_by_pattern.get(("est", "et"), [])
    if not est_et_cases:
        # Check in all FN
        est_et_cases = [(err, item) for err, item in all_fn if err["orig"] == "est" and err["gold"] == "et"]
    print(f"  Total 'est'->'et' FN cases: {len(est_et_cases)}")
    for i, (err, item) in enumerate(est_et_cases[:5], 1):
        fautif = item["fautif"]
        correct = item["correct"]
        words_f = fautif.split()
        words_c = correct.split()
        pos = err["pos"]
        print(f"\n  Case {i} (source: {item['_source']}, idx: {item['idx']}):")
        print(f"    Fautif:  {fautif}")
        print(f"    Correct: {correct}")
        print(f"    Obtenu:  {item['obtenu']}")
        # Show surrounding context
        before = words_f[max(0, pos-3):pos]
        word_at = words_f[pos] if pos < len(words_f) else "?"
        after = words_f[pos+1:pos+4]
        print(f"    Context: ...{' '.join(before)} [{word_at}->{err['gold']}] {' '.join(after)}...")
        print(f"    Likely POS:")
        for w in before:
            print(f"      '{w}' = {guess_pos(w)}")
        print(f"      '{word_at}' (error: should be '{err['gold']}') = CONJ_COORD")
        for w in after:
            print(f"      '{w}' = {guess_pos(w)}")

    # ====================================================================
    # 5. Top 5 "et" -> "est" FN cases with context
    # ====================================================================
    print("\n" + "=" * 80)
    print("5. TOP 5 'et' -> 'est' FN CASES WITH CONTEXT")
    print("=" * 80)
    et_est_cases = homo_by_pattern.get(("et", "est"), [])
    if not et_est_cases:
        et_est_cases = [(err, item) for err, item in all_fn if err["orig"] == "et" and err["gold"] == "est"]
    print(f"  Total 'et'->'est' FN cases: {len(et_est_cases)}")
    for i, (err, item) in enumerate(et_est_cases[:5], 1):
        fautif = item["fautif"]
        correct = item["correct"]
        words_f = fautif.split()
        pos = err["pos"]
        print(f"\n  Case {i} (source: {item['_source']}, idx: {item['idx']}):")
        print(f"    Fautif:  {fautif}")
        print(f"    Correct: {correct}")
        print(f"    Obtenu:  {item['obtenu']}")
        before = words_f[max(0, pos-3):pos]
        word_at = words_f[pos] if pos < len(words_f) else "?"
        after = words_f[pos+1:pos+4]
        print(f"    Context: ...{' '.join(before)} [{word_at}->{err['gold']}] {' '.join(after)}...")

    # ====================================================================
    # 6. FP patterns appearing 2+ times
    # ====================================================================
    print("\n" + "=" * 80)
    print("6. FALSE POSITIVE PATTERNS APPEARING 2+ TIMES")
    print("=" * 80)
    fp_pattern_counter = Counter()
    fp_by_pattern = defaultdict(list)

    for err, item in all_fp:
        # For FP: orig==gold (it was correct), sys changed it wrongly
        pattern = (err["orig"], err["sys"])
        fp_pattern_counter[pattern] += 1
        fp_by_pattern[pattern].append((err, item))

    fp_repeated = [(p, c) for p, c in fp_pattern_counter.most_common() if c >= 2]
    if not fp_repeated:
        print("  No FP patterns appear 2+ times.")
    else:
        for (orig, sys_out), count in fp_repeated:
            print(f"\n  '{orig}' -> '{sys_out}' (WRONG)  x{count}")
            for err, item in fp_by_pattern[(orig, sys_out)][:2]:
                print(f"      [{item['_source']}] Fautif:  {item['fautif']}")
                print(f"      [{item['_source']}] Correct: {item['correct']}")
                print(f"      [{item['_source']}] Obtenu:  {item['obtenu']}")
                print()

    # ====================================================================
    # ADDITIONAL: FN where system output != orig (system tried but got wrong)
    # ====================================================================
    print("\n" + "=" * 80)
    print("7. FN WHERE SYSTEM ATTEMPTED CORRECTION BUT GOT IT WRONG")
    print("=" * 80)
    wrong_corrections = [(err, item) for err, item in all_fn
                         if (err.get("sys") or "").rstrip(".,;:!?").lower() != err["orig"].lower()]
    print(f"  Total: {len(wrong_corrections)} (out of {len(all_fn)} FN)")
    pattern_counter = Counter()
    for err, item in wrong_corrections:
        pattern_counter[(err["orig"], err["gold"], err["sys"])] += 1
    for (orig, gold, sys_out), count in pattern_counter.most_common(20):
        print(f"    {orig:20s} -> should be '{gold:20s}' but got '{sys_out}'  x{count}")

    # ====================================================================
    # SUMMARY STATS
    # ====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_expected_errors = 0
    total_tp = 0
    total_fn_count = 0
    total_fp_count = 0
    for item in results:
        total_tp += item.get("tp", 0)
        total_fn_count += item.get("fn", 0)
        total_fp_count += item.get("fp", 0)

    total_corrections = total_tp + total_fp_count
    total_expected = total_tp + total_fn_count
    precision = total_tp / total_corrections if total_corrections else 0
    recall = total_tp / total_expected if total_expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"  TP: {total_tp}  FN: {total_fn_count}  FP: {total_fp_count}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Total expected errors: {total_expected}")
    print(f"  Total system corrections: {total_corrections}")


def guess_pos(word):
    """Naive POS tagger for French context analysis."""
    w = word.lower().rstrip(".,;:!?'\"")

    determiners = {"le", "la", "les", "l", "un", "une", "des", "du", "de", "d",
                   "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses",
                   "notre", "votre", "leur", "nos", "vos", "leurs",
                   "ce", "cet", "cette", "ces", "quel", "quelle", "quels", "quelles",
                   "chaque", "aucun", "aucune"}
    pronouns = {"je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
                "me", "te", "se", "lui", "y", "en", "qui", "que", "qu",
                "celui", "celle", "ceux", "celles", "dont", "où",
                "c", "s", "j", "l", "n", "m", "t"}
    prepositions = {"à", "a", "de", "en", "dans", "sur", "sous", "par", "pour", "avec",
                    "sans", "entre", "vers", "chez", "contre", "depuis", "jusqu",
                    "pendant", "après", "avant", "devant", "derrière", "dès"}
    conjunctions = {"et", "ou", "mais", "donc", "or", "ni", "car", "que", "quand",
                    "si", "comme", "lorsque", "puisque", "quoique"}
    auxiliaries_etre = {"est", "sont", "était", "étaient", "sera", "seront", "soit",
                       "suis", "es", "sommes", "êtes", "fut", "fût", "serait"}
    auxiliaries_avoir = {"a", "ont", "avait", "avaient", "aura", "auront", "ait",
                        "ai", "as", "avons", "avez", "eut", "eût", "aurait"}
    adverbs = {"ne", "pas", "plus", "très", "bien", "mal", "aussi", "encore",
               "toujours", "jamais", "souvent", "déjà", "trop", "peu", "beaucoup",
               "vraiment", "alors", "puis", "ensuite", "notamment", "également"}

    if w in determiners:
        return "DET"
    if w in pronouns:
        return "PRON"
    if w in prepositions:
        return "PREP"
    if w in conjunctions:
        return "CONJ"
    if w in auxiliaries_etre:
        return "V_ETRE"
    if w in auxiliaries_avoir:
        return "V_AVOIR"
    if w in adverbs:
        return "ADV"
    if w.endswith(("er", "ir", "re", "oir")):
        return "V_INF/N?"
    if w.endswith(("tion", "sion", "ment", "eur", "ence", "ance", "ité")):
        return "N"
    if w.endswith(("ait", "ais", "aient", "ions", "iez")):
        return "V_IMP?"
    if w.endswith(("ons", "ez")):
        return "V_PRES?"
    if w.endswith(("é", "ée", "és", "ées")):
        return "PP/ADJ"
    if w.endswith(("ant", "ante", "ants", "antes")):
        return "PRES_PART/ADJ"
    return "N/ADJ/V?"


if __name__ == "__main__":
    main()
