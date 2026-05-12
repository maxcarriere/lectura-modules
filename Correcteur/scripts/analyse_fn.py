#!/usr/bin/env python3
"""Analyse detaillee des 1129 FN (faux negatifs) du benchmark session3_baseline."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
BENCHMARK = Path(__file__).resolve().parent.parent / "benchmark" / "iterations" / "session3_baseline.json"

with open(BENCHMARK, encoding="utf-8") as f:
    data = json.load(f)

# ---------------------------------------------------------------------------
# Extraction des FN individuels
# ---------------------------------------------------------------------------
# Un FN = une erreur dans erreurs_detail ou type != "FP" et:
#   - sys == orig (pas corrige du tout) => "missed"
#   - sys != orig et sys != gold (corrige mais mal) => "wrong"
fn_items = []
fn_missed = []
fn_wrong = []
for item in data:
    for err in item.get("erreurs_detail", []):
        if err["type"] == "FP":
            continue
        orig_l = (err["orig"] or "").lower()
        gold_l = (err["gold"] or "").lower()
        sys_l = (err["sys"] or "").lower()

        if sys_l == gold_l:
            continue  # TP, pas un FN

        entry = {
            "idx": item["idx"],
            "orig": err["orig"],
            "gold": err["gold"],
            "sys": err["sys"],
            "type_benchmark": err["type"],
            "fautif": item["fautif"],
            "correct": item["correct"],
            "obtenu": item["obtenu"],
        }

        if sys_l == orig_l:
            fn_items.append(entry)
            fn_missed.append(entry)
        else:
            # sys != orig and sys != gold: wrong correction
            entry["fn_kind"] = "wrong"
            fn_items.append(entry)
            fn_wrong.append(entry)

print(f"Total FN extraits: {len(fn_items)}")
print(f"  - missed (sys==orig, not corrected): {len(fn_missed)}")
print(f"  - wrong  (sys!=orig!=gold, bad correction): {len(fn_wrong)}")
print(f"  (item[fn] sum from benchmark: {sum(item['fn'] for item in data)})")
print()

# ---------------------------------------------------------------------------
# Homophones connus
# ---------------------------------------------------------------------------
HOMO_PAIRS = {
    # paire: set de formes qui se confondent
    "a/à": {"a", "à"},
    "est/et": {"est", "et"},
    "ou/où": {"ou", "où"},
    "on/ont": {"on", "ont"},
    "son/sont": {"son", "sont"},
    "ses/ces": {"ses", "ces"},
    "se/ce": {"se", "ce"},
    "sa/ça": {"sa", "ça"},
    "la/là": {"la", "là"},
    "peu/peut/peux": {"peu", "peut", "peux"},
    "ma/m'a": {"ma", "m'a"},
    "ta/t'a": {"ta", "t'a"},
    "mes/mais": {"mes", "mais"},
    "mon/m'ont": {"mon", "m'ont"},
    "ton/t'ont": {"ton", "t'ont"},
    "quel/quelle": {"quel", "quelle"},
    "quels/quelles": {"quels", "quelles"},
    "ni/n'y": {"ni", "n'y"},
    "si/s'y": {"si", "s'y"},
    "dans/d'en": {"dans", "d'en"},
    "sans/s'en": {"sans", "s'en"},
    "sens/sent/s'en": {"sens", "sent", "s'en"},
    "sais/sait/c'est/s'est": {"sais", "sait", "c'est", "s'est"},
    "leur/leurs": {"leur", "leurs"},
    "tout/tous": {"tout", "tous"},
    "quel/qu'elle": {"quel", "qu'elle"},
    "quand/quant/qu'en": {"quand", "quant", "qu'en"},
    "près/prêt": {"près", "prêt", "prêts"},
    "fois/foi/foie": {"fois", "foi", "foie"},
    "voie/voix": {"voie", "voix"},
    "ver/vers/vert/verre": {"ver", "vers", "vert", "verre"},
    "du/dû": {"du", "dû"},
    "sur/sûr": {"sur", "sûr", "sûre"},
    "ou/ou": set(),  # handled above
}

def find_homo_pair(orig_low, gold_low):
    """Retourne le nom de la paire d'homophones si applicable."""
    for name, forms in HOMO_PAIRS.items():
        if orig_low in forms and gold_low in forms:
            return name
    return None

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_fn(item):
    """Classifie un FN en categorie + sous-categorie."""
    orig = item["orig"] or ""
    gold = item["gold"] or ""
    if not orig or not gold:
        return "OTHER", f"{orig}→{gold}", "empty"
    orig_low = orig.lower()
    gold_low = gold.lower()
    if orig_low == gold_low:
        return "OTHER", f"{orig_low}→{gold_low}", "case_only"

    # --- HOMO ---
    pair = find_homo_pair(orig_low, gold_low)
    if pair:
        return "HOMO", f"{orig_low}→{gold_low}", pair

    # --- ACC_S (pluriel) ---
    # ajouter ou retirer -s/-x
    if orig_low + "s" == gold_low or orig_low + "x" == gold_low:
        return "ACC_S", "need_add_s", f"+s ({orig_low}→{gold_low})"
    if gold_low + "s" == orig_low or gold_low + "x" == orig_low:
        return "ACC_S", "need_rm_s", f"-s ({orig_low}→{gold_low})"
    # -aux/-al, -eaux/-eau
    if re.match(r'^(.+)aux$', orig_low) and re.match(r'^(.+)al$', gold_low):
        stem_o = orig_low[:-3]
        stem_g = gold_low[:-2]
        if stem_o == stem_g:
            return "ACC_S", "need_rm_s", f"-aux→-al ({orig_low}→{gold_low})"
    if re.match(r'^(.+)al$', orig_low) and re.match(r'^(.+)aux$', gold_low):
        stem_o = orig_low[:-2]
        stem_g = gold_low[:-3]
        if stem_o == stem_g:
            return "ACC_S", "need_add_s", f"-al→-aux ({orig_low}→{gold_low})"

    # --- ACC_E (genre -e) ---
    if orig_low + "e" == gold_low:
        return "ACC_E", "need_add_e", f"+e ({orig_low}→{gold_low})"
    if gold_low + "e" == orig_low:
        return "ACC_E", "need_rm_e", f"-e ({orig_low}→{gold_low})"
    # -ée/-é
    if orig_low.endswith("é") and gold_low == orig_low + "e":
        return "ACC_E", "need_add_e", f"+e ({orig_low}→{gold_low})"
    if orig_low.endswith("ée") and gold_low == orig_low[:-1]:
        return "ACC_E", "need_rm_e", f"-e ({orig_low}→{gold_low})"

    # --- ACC_ES (genre+nombre -es) ---
    if orig_low + "es" == gold_low:
        return "ACC_E", "need_add_es", f"+es ({orig_low}→{gold_low})"
    if gold_low + "es" == orig_low:
        return "ACC_E", "need_rm_es", f"-es ({orig_low}→{gold_low})"

    # --- PP (participe passe / infinitif) ---
    # -er/-é, -é/-er, -er/-ée, -ée/-er, -és/-er, -er/-és, etc.
    pp_patterns = [
        (r'(.+)er$', r'\1é'),   # infinitif → pp masc sg
        (r'(.+)é$', r'\1er'),   # pp masc sg → infinitif
        (r'(.+)er$', r'\1ée'),  # infinitif → pp fem sg
        (r'(.+)ée$', r'\1er'),  # pp fem sg → infinitif
        (r'(.+)er$', r'\1és'),  # infinitif → pp masc pl
        (r'(.+)és$', r'\1er'),  # pp masc pl → infinitif
        (r'(.+)er$', r'\1ées'), # infinitif → pp fem pl
        (r'(.+)ées$', r'\1er'), # pp fem pl → infinitif
        (r'(.+)é$', r'\1ée'),   # pp masc → pp fem
        (r'(.+)ée$', r'\1é'),   # pp fem → pp masc
        (r'(.+)é$', r'\1és'),   # pp sg → pp pl
        (r'(.+)és$', r'\1é'),   # pp pl → pp sg
    ]
    for pat_o, pat_g in pp_patterns:
        m = re.match(pat_o, orig_low)
        if m:
            expected = re.sub(pat_o, pat_g, orig_low)
            if expected == gold_low:
                return "PP", f"{orig_low}→{gold_low}", ""

    # --- CONJ (conjugaison) ---
    # (suffix_orig, suffix_gold, label)
    conj_suffixes = [
        ("ent", "e", "3pl→3sg"),
        ("e", "ent", "3sg→3pl"),
        ("es", "e", "2sg→3sg"),
        ("e", "es", "3sg→2sg"),
        ("ons", "ont", "1pl→3pl"),
        ("ont", "ons", "3pl→1pl"),
        ("ait", "ais", "3sg→1sg_imp"),
        ("ais", "ait", "1sg→3sg_imp"),
        ("aient", "ait", "3pl→3sg_imp"),
        ("ait", "aient", "3sg→3pl_imp"),
        ("aient", "ent", "3pl_imp→3pl_pres"),
        ("ent", "aient", "3pl_pres→3pl_imp"),
        ("ons", "ent", "1pl→3pl"),
        ("ent", "ons", "3pl→1pl"),
        ("ez", "ent", "2pl→3pl"),
        ("ent", "ez", "3pl→2pl"),
        ("ais", "aient", "1sg_imp→3pl_imp"),
        ("aient", "ais", "3pl_imp→1sg_imp"),
        ("a", "e", "3sg_ps→3sg_pres"),
        ("e", "a", "3sg_pres→3sg_ps"),
    ]
    for suf_o, suf_g, label in conj_suffixes:
        if orig_low.endswith(suf_o) and gold_low.endswith(suf_g):
            stem_o = orig_low[:-len(suf_o)] if suf_o else orig_low
            stem_g = gold_low[:-len(suf_g)] if suf_g else gold_low
            if stem_o == stem_g and len(stem_o) >= 2:
                return "CONJ", label, f"{orig_low}→{gold_low}"

    # Broader CONJ detection: same benchmark type
    if item["type_benchmark"] == "CONJ":
        return "CONJ", "other", f"{orig_low}→{gold_low}"

    # --- ACCENT ---
    # Normalize: remove all accents and compare
    import unicodedata
    def strip_accents(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    if strip_accents(orig_low) == strip_accents(gold_low) and orig_low != gold_low:
        return "ACCENT", f"{orig_low}→{gold_low}", ""

    # --- PHON (erreur phonetique) ---
    # Use the benchmark type as a hint
    if item["type_benchmark"] == "PHON":
        return "PHON", f"{orig_low}→{gold_low}", ""

    # --- TYPO ---
    # Levenshtein distance 1-2 and same length +/- 1
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev_row[j + 1] + 1
                dele = curr_row[j] + 1
                sub = prev_row[j] + (c1 != c2)
                curr_row.append(min(ins, dele, sub))
            prev_row = curr_row
        return prev_row[-1]

    dist = levenshtein(orig_low, gold_low)
    if dist <= 2 and abs(len(orig_low) - len(gold_low)) <= 1:
        return "TYPO", f"{orig_low}→{gold_low}", f"dist={dist}"

    # --- OTHER ---
    return "OTHER", f"{orig_low}→{gold_low}", item["type_benchmark"]

# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
categories = Counter()
homo_pairs = Counter()     # paire homophone
homo_directions = Counter() # orig→gold
acc_s_sub = Counter()
acc_e_sub = Counter()
conj_sub = Counter()
pp_sub = Counter()
accent_sub = Counter()
phon_sub = Counter()
other_sub = Counter()

# Toutes les corrections individuelles
all_corrections = Counter()  # (orig, gold) → count

classified = []

for item in fn_items:
    cat, sub, extra = classify_fn(item)
    categories[cat] += 1
    o = (item["orig"] or "").lower()
    g = (item["gold"] or "").lower()
    all_corrections[(o, g)] += 1

    classified.append({**item, "category": cat, "sub": sub, "extra": extra})

    if cat == "HOMO":
        homo_pairs[extra] += 1
        homo_directions[sub] += 1
    elif cat == "ACC_S":
        acc_s_sub[sub] += 1
    elif cat == "ACC_E":
        acc_e_sub[sub] += 1
    elif cat == "CONJ":
        conj_sub[sub] += 1
    elif cat == "PP":
        pp_sub[sub] += 1
    elif cat == "ACCENT":
        accent_sub[sub] += 1
    elif cat == "PHON":
        phon_sub[sub] += 1
    elif cat == "OTHER":
        other_sub[sub] += 1

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SEP = "=" * 80

print(SEP)
print("1. CATEGORY COUNTS")
print(SEP)
total = sum(categories.values())
for cat, cnt in categories.most_common():
    print(f"  {cat:10s}  {cnt:5d}  ({cnt/total*100:5.1f}%)")
print(f"  {'TOTAL':10s}  {total:5d}")
print()

print(SEP)
print("2. HOMO — sub-counts by pair")
print(SEP)
for pair, cnt in homo_pairs.most_common():
    print(f"  {pair:25s}  {cnt:4d}")
print()
print("  HOMO — sub-counts by direction (orig→gold)")
for direction, cnt in homo_directions.most_common(30):
    print(f"  {direction:25s}  {cnt:4d}")
print()

print(SEP)
print("3. ACC_S — sub-counts by direction")
print(SEP)
for sub, cnt in acc_s_sub.most_common():
    print(f"  {sub:20s}  {cnt:4d}")
print()

print(SEP)
print("3b. ACC_E — sub-counts by direction")
print(SEP)
for sub, cnt in acc_e_sub.most_common():
    print(f"  {sub:20s}  {cnt:4d}")
print()

print(SEP)
print("4. CONJ — sub-counts by pattern")
print(SEP)
for sub, cnt in conj_sub.most_common():
    print(f"  {sub:25s}  {cnt:4d}")
print()

print(SEP)
print("4b. PP — sub-counts")
print(SEP)
for sub, cnt in pp_sub.most_common(20):
    print(f"  {sub:30s}  {cnt:4d}")
print()

print(SEP)
print("4c. ACCENT — sub-counts")
print(SEP)
for sub, cnt in accent_sub.most_common(20):
    print(f"  {sub:30s}  {cnt:4d}")
print()

print(SEP)
print("5. TOP 20 MOST FREQUENT INDIVIDUAL FN CORRECTIONS")
print(SEP)
for (orig, gold), cnt in all_corrections.most_common(20):
    print(f"  {orig:20s} → {gold:20s}  {cnt:4d}")
print()

print(SEP)
print("6. PHON details (top 30)")
print(SEP)
for sub, cnt in phon_sub.most_common(30):
    print(f"  {sub:40s}  {cnt:4d}")
print()

print(SEP)
print("7. OTHER details (top 30)")
print(SEP)
for sub, cnt in other_sub.most_common(30):
    print(f"  {sub:40s}  {cnt:4d}")
print()

# ---------------------------------------------------------------------------
# 6. Recoverable clusters
# ---------------------------------------------------------------------------
print(SEP)
print("8. RECOVERABLE CLUSTERS — rule changes that could capture many FN")
print(SEP)
print()

# Group by category and compute "recoverable" score
print("  A) Homophones — easiest to add rules for:")
print("     Each pair below is a self-contained rule.")
print()
for pair, cnt in homo_pairs.most_common(10):
    directions = [(d, c) for d, c in homo_directions.most_common()
                  if any(p == pair for p2, _ in [(pair, None)]
                         for p in [pair])]
    # Filter directions that belong to this pair
    pair_forms = HOMO_PAIRS.get(pair, set())
    pair_dirs = [(d, c) for d, c in homo_directions.items()
                 if d.split("→")[0] in pair_forms and d.split("→")[1] in pair_forms]
    pair_dirs.sort(key=lambda x: -x[1])
    print(f"     {pair} ({cnt} FN total)")
    for d, c in pair_dirs:
        print(f"       {d}: {c}")
    print()

print("  B) Accord pluriel (ACC_S) — {sum(acc_s_sub.values())} FN")
print(f"     need_add_s: {acc_s_sub.get('need_add_s', 0)}")
print(f"     need_rm_s:  {acc_s_sub.get('need_rm_s', 0)}")
print()

print(f"  C) Accord genre (ACC_E) — {sum(acc_e_sub.values())} FN")
for sub, cnt in acc_e_sub.most_common():
    print(f"     {sub}: {cnt}")
print()

print(f"  D) Participe passe (PP) — {sum(pp_sub.values())} FN")
pp_er_to_e = sum(c for s, c in pp_sub.items() if "er→" in s and s.endswith("é"))
pp_e_to_er = sum(c for s, c in pp_sub.items() if "é→" in s and s.endswith("er"))
print(f"     -er→-é type:  check all PP sub-items above")
print(f"     Most frequent PP corrections:")
for sub, cnt in pp_sub.most_common(10):
    print(f"       {sub}: {cnt}")
print()

print(f"  E) Conjugaison (CONJ) — {sum(conj_sub.values())} FN")
for sub, cnt in conj_sub.most_common():
    print(f"     {sub}: {cnt}")
print()

print(f"  F) Accents (ACCENT) — {sum(accent_sub.values())} FN")
print(f"     These are pure accent additions/changes, potentially fixable by dictionary lookup.")
for sub, cnt in accent_sub.most_common(10):
    print(f"       {sub}: {cnt}")
print()

# Summary: recoverable by effort
print(SEP)
print("SUMMARY: RECOVERABLE FN BY EFFORT")
print(SEP)
clusters = [
    ("HOMO rules (a/à, est/et, etc.)", sum(homo_pairs.values())),
    ("ACC_S plural agreement", sum(acc_s_sub.values())),
    ("ACC_E gender agreement", sum(acc_e_sub.values())),
    ("PP participe/infinitif", sum(pp_sub.values())),
    ("CONJ conjugation", sum(conj_sub.values())),
    ("ACCENT diacritics", sum(accent_sub.values())),
    ("PHON phonetic", sum(phon_sub.values())),
    ("TYPO/OTHER", categories.get("TYPO", 0) + categories.get("OTHER", 0)),
]
clusters.sort(key=lambda x: -x[1])
cumul = 0
for name, cnt in clusters:
    cumul += cnt
    print(f"  {name:40s}  {cnt:5d}  (cumul: {cumul:5d} / {total} = {cumul/total*100:.1f}%)")

# ---------------------------------------------------------------------------
# CONJ "other" detail
# ---------------------------------------------------------------------------
print()
print(SEP)
print("9. CONJ 'other' details (top 30)")
print(SEP)
conj_other_detail = Counter()
for c in classified:
    if c["category"] == "CONJ" and c["sub"] == "other":
        o = (c["orig"] or "").lower()
        g = (c["gold"] or "").lower()
        conj_other_detail[f"{o}→{g}"] += 1
for key, cnt in conj_other_detail.most_common(30):
    print(f"  {key:40s}  {cnt:4d}")
print()

# ---------------------------------------------------------------------------
# Wrong corrections (sys != orig != gold)
# ---------------------------------------------------------------------------
print(SEP)
print("10. WRONG CORRECTIONS (sys!=orig, sys!=gold) — top 30")
print(SEP)
wrong_detail = Counter()
for item in fn_wrong:
    o = (item["orig"] or "").lower()
    g = (item["gold"] or "").lower()
    s = (item["sys"] or "").lower()
    wrong_detail[f"{o}→{s} (gold: {g})"] += 1
for key, cnt in wrong_detail.most_common(30):
    print(f"  {key:55s}  {cnt:4d}")
print()

# Wrong corrections by category
print("  Wrong corrections by benchmark type:")
wrong_by_type = Counter()
for item in fn_wrong:
    wrong_by_type[item["type_benchmark"]] += 1
for t, cnt in wrong_by_type.most_common():
    print(f"    {t:15s}  {cnt:4d}")
print()

# ---------------------------------------------------------------------------
# ACC_S detail: most frequent words
# ---------------------------------------------------------------------------
print(SEP)
print("11. ACC_S — most frequent individual words (top 20)")
print(SEP)
acc_s_words = Counter()
for c in classified:
    if c["category"] == "ACC_S":
        o = (c["orig"] or "").lower()
        g = (c["gold"] or "").lower()
        acc_s_words[f"{o}→{g}"] += 1
for key, cnt in acc_s_words.most_common(20):
    print(f"  {key:40s}  {cnt:4d}")
print()

# ---------------------------------------------------------------------------
# TYPO detail
# ---------------------------------------------------------------------------
print(SEP)
print("12. TYPO — most frequent individual corrections (top 20)")
print(SEP)
typo_detail = Counter()
for c in classified:
    if c["category"] == "TYPO":
        o = (c["orig"] or "").lower()
        g = (c["gold"] or "").lower()
        typo_detail[f"{o}→{g}"] += 1
for key, cnt in typo_detail.most_common(20):
    print(f"  {key:40s}  {cnt:4d}")
