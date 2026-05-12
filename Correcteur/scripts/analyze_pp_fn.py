#!/usr/bin/env python3
"""
Analyze PP (past participle) false negatives from benchmark results.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

def get_context_window(texte: str, word_pos: int, window: int = 3) -> str:
    """Extract context around a word position (n words before and after)."""
    words = texte.split()

    start = max(0, word_pos - window)
    end = min(len(words), word_pos + window + 1)
    context_words = words[start:end]

    # Mark the target word
    target_idx = word_pos - start
    if 0 <= target_idx < len(context_words):
        context_words[target_idx] = f">>>{context_words[target_idx]}<<<"

    return " ".join(context_words)

def get_preceding_words(texte: str, word_pos: int, n: int = 3) -> list[str]:
    """Get n words before word position."""
    words = texte.split()
    if word_pos <= 0:
        return []
    start = max(0, word_pos - n)
    return words[start:word_pos]

def classify_pp_error(erreur: dict, texte: str) -> tuple[str, dict]:
    """
    Classify PP error type.
    Returns: (category, context_info)

    Benchmark format:
    - pos: word position in fautif text (0-indexed)
    - orig: original word in fautif text
    - gold: expected correct word
    - sys: what the system produced
    """
    word_pos = erreur["pos"]
    gold = erreur["gold"]  # Expected correct word
    orig = erreur["orig"]  # Original faulty word (what we found)

    # Use "fautif" text from parent item to get context
    # (texte will be passed as "fautif")

    # Get preceding words
    preceding = get_preceding_words(texte, word_pos, 5)
    preceding_lower = [w.lower().strip(",.;:!?") for w in preceding]

    gold_lower = gold.lower()
    orig_lower = orig.lower()

    # Category classification
    category = "other"
    context = {
        "attendu": gold,
        "trouve": orig,
        "preceding_3": " ".join(preceding[-3:]) if preceding else "",
        "preceding_all": " ".join(preceding) if preceding else "",
        "full_context": get_context_window(texte, word_pos, 4)
    }

    # er→é (infinitive→participle)
    if orig_lower.endswith("er") and re.match(r".*é(?:e|s|es)?$", gold_lower):
        category = "er_to_e"
        # Check if auxiliary present
        auxiliaries = {"être", "avoir", "est", "sont", "a", "ont", "ai", "as", "avons", "avez",
                      "était", "étaient", "avait", "avaient", "été", "eu", "eus", "eut"}
        context["has_auxiliary"] = any(w in auxiliaries for w in preceding_lower)
        context["auxiliary_found"] = [w for w in preceding_lower if w in auxiliaries]

    # é→er (participle→infinitive)
    elif re.match(r".*é(?:e|s|es)?$", orig_lower) and gold_lower.endswith("er"):
        category = "e_to_er"
        # Check if modal/preposition present
        modals_prep = {"pour", "de", "à", "sans", "avant", "après", "aller", "vais", "vas",
                      "va", "allons", "allez", "vont", "dois", "doit", "devons", "devez",
                      "doivent", "peux", "peut", "pouvons", "pouvez", "peuvent", "veux",
                      "veut", "voulons", "voulez", "veulent", "faut", "falloir"}
        context["has_modal_prep"] = any(w in modals_prep for w in preceding_lower)
        context["modal_prep_found"] = [w for w in preceding_lower if w in modals_prep]

    # é→ée/ées (masculine→feminine PP)
    elif (orig_lower.endswith("é") and (gold_lower.endswith("ée") or
          gold_lower.endswith("és") or gold_lower.endswith("ées"))):
        category = "masc_to_fem"
        auxiliaries = {"être", "est", "sont", "était", "étaient", "été", "fut", "furent"}
        context["has_etre"] = any(w in auxiliaries for w in preceding_lower)
        context["auxiliary_found"] = [w for w in preceding_lower if w in auxiliaries]

    # ée/ées→é (feminine→masculine PP)
    elif ((orig_lower.endswith("ée") or orig_lower.endswith("ées")) and
          gold_lower.endswith("é")):
        category = "fem_to_masc"

    return category, context

def main():
    # Load benchmark results
    benchmark_path = Path("/data/work/projets/lectura/workspace/Modules/Correcteur/benchmark/iterations/conj_fixes_v1.json")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Collect PP false negatives
    pp_fn_by_category = defaultdict(list)

    for item in results:
        # Use "fautif" as the original text to get context
        texte = item.get("fautif", "")
        erreurs_detail = item.get("erreurs_detail", [])

        for erreur in erreurs_detail:
            if erreur.get("type") == "PP":
                category, context = classify_pp_error(erreur, texte)
                pp_fn_by_category[category].append(context)

    # Report results
    print("=" * 80)
    print("PP FALSE NEGATIVE ANALYSIS")
    print("=" * 80)
    print()

    total_pp_fn = sum(len(v) for v in pp_fn_by_category.values())
    print(f"Total PP FN: {total_pp_fn}")
    print()

    # Category: er→é (infinitive→participle)
    category = "er_to_e"
    if category in pp_fn_by_category:
        items = pp_fn_by_category[category]
        with_aux = [x for x in items if x.get("has_auxiliary")]
        print(f"📌 CATEGORY A: er→é/ée/és/ées (infinitive→participle)")
        print(f"   Count: {len(items)}")
        print(f"   With auxiliary (être/avoir/est/sont/ont/etc.): {len(with_aux)} ({100*len(with_aux)/len(items):.1f}%)")
        print()
        print("   Examples (with auxiliary):")
        for i, ex in enumerate(with_aux[:5], 1):
            print(f"   {i}. {ex['trouve']} → {ex['attendu']}")
            print(f"      Aux: {ex['auxiliary_found']}")
            print(f"      Context: {ex['full_context']}")
            print()

        without_aux = [x for x in items if not x.get("has_auxiliary")]
        if without_aux:
            print(f"   Examples (NO auxiliary - harder cases):")
            for i, ex in enumerate(without_aux[:5], 1):
                print(f"   {i}. {ex['trouve']} → {ex['attendu']}")
                print(f"      Preceding: {ex['preceding_all']}")
                print(f"      Context: {ex['full_context']}")
                print()
        print()

    # Category: é→er (participle→infinitive)
    category = "e_to_er"
    if category in pp_fn_by_category:
        items = pp_fn_by_category[category]
        with_modal = [x for x in items if x.get("has_modal_prep")]
        print(f"📌 CATEGORY B: é/ée/és/ées→er (participle→infinitive)")
        print(f"   Count: {len(items)}")
        print(f"   With modal/prep (pour/de/à/aller/devoir/etc.): {len(with_modal)} ({100*len(with_modal)/len(items):.1f}%)")
        print()
        print("   Examples (with modal/prep):")
        for i, ex in enumerate(with_modal[:5], 1):
            print(f"   {i}. {ex['trouve']} → {ex['attendu']}")
            print(f"      Modal/Prep: {ex['modal_prep_found']}")
            print(f"      Context: {ex['full_context']}")
            print()

        without_modal = [x for x in items if not x.get("has_modal_prep")]
        if without_modal:
            print(f"   Examples (NO modal/prep - harder cases):")
            for i, ex in enumerate(without_modal[:5], 1):
                print(f"   {i}. {ex['trouve']} → {ex['attendu']}")
                print(f"      Preceding: {ex['preceding_all']}")
                print(f"      Context: {ex['full_context']}")
                print()
        print()

    # Category: é→ée/ées (masculine→feminine PP)
    category = "masc_to_fem"
    if category in pp_fn_by_category:
        items = pp_fn_by_category[category]
        with_etre = [x for x in items if x.get("has_etre")]
        print(f"📌 CATEGORY C: é→ée/és/ées (masculine→feminine PP agreement)")
        print(f"   Count: {len(items)}")
        print(f"   With être auxiliary: {len(with_etre)} ({100*len(with_etre)/len(items):.1f}%)")
        print()
        print("   Examples:")
        for i, ex in enumerate(items[:5], 1):
            print(f"   {i}. {ex['trouve']} → {ex['attendu']}")
            if ex.get("has_etre"):
                print(f"      Être aux: {ex['auxiliary_found']}")
            print(f"      Context: {ex['full_context']}")
            print()
        print()

    # Category: ée/ées→é (feminine→masculine PP)
    category = "fem_to_masc"
    if category in pp_fn_by_category:
        items = pp_fn_by_category[category]
        print(f"📌 CATEGORY D: ée/ées→é (feminine→masculine PP)")
        print(f"   Count: {len(items)}")
        print()
        print("   Examples:")
        for i, ex in enumerate(items[:5], 1):
            print(f"   {i}. {ex['trouve']} → {ex['attendu']}")
            print(f"      Context: {ex['full_context']}")
            print()
        print()

    # Other PP patterns
    category = "other"
    if category in pp_fn_by_category:
        items = pp_fn_by_category[category]
        print(f"📌 OTHER PP PATTERNS")
        print(f"   Count: {len(items)}")
        print()
        print("   Examples:")
        for i, ex in enumerate(items[:10], 1):
            print(f"   {i}. {ex['trouve']} → {ex['attendu']}")
            print(f"      Context: {ex['full_context']}")
            print()
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Category':<40} {'Count':<10} {'% of PP FN':<15}")
    print("-" * 80)
    for cat, items in sorted(pp_fn_by_category.items(), key=lambda x: -len(x[1])):
        cat_name = {
            "er_to_e": "A. er→é (infinitive→participle)",
            "e_to_er": "B. é→er (participle→infinitive)",
            "masc_to_fem": "C. é→ée/ées (masc→fem agreement)",
            "fem_to_masc": "D. ée→é (fem→masc)",
            "other": "Other PP patterns"
        }.get(cat, cat)
        print(f"{cat_name:<40} {len(items):<10} {100*len(items)/total_pp_fn:>6.1f}%")
    print("-" * 80)
    print(f"{'TOTAL':<40} {total_pp_fn:<10} {'100.0%':>6}")
    print()

if __name__ == "__main__":
    main()
