#!/usr/bin/env python3
"""Analyse des faux positifs (FP) du benchmark pp_pronoun_v4.

Objectif : identifier les FP causes par le module orthographique qui
"corrige" des noms propres, mots etrangers, et mots OOV vers des
mots francais proches.

Categories analysees :
  1. ACCENT_ADD  : ajout d'accent (ocean -> ocean, halevi -> halevi)
  2. ACCENT_STRIP: retrait d'accent (hyderabab -> hyderabad, sarria -> sarria)
  3. ACCENT_CROSS: changement d'accent (decedra -> decedra)
  4. EDIT_1_2    : changement de 1-2 lettres (yari -> mari, boyd -> bord)
  5. PLURAL_AGREE: ajout/retrait de -s/-es/-e (finales, convaincues)
  6. OTHER       : autres

Pour chaque FP, on verifie :
  - L'original est-il un nom propre probable (OOV + majuscule dans gold) ?
  - L'original est-il un mot etranger (OOV dans le lexique) ?
  - Y a-t-il d'autres mots OOV a proximite (contexte etranger) ?
  - Une heuristique "N consecutifs OOV" pourrait-elle aider ?
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Charger le benchmark ─────────────────────────────────────────────

BENCH_PATH = Path(__file__).resolve().parent.parent / "benchmark" / "iterations" / "pp_pronoun_v4.json"

with open(BENCH_PATH, encoding="utf-8") as f:
    data = json.load(f)

print(f"Benchmark: {BENCH_PATH.name}")
print(f"Phrases: {len(data)}")
print()

# ── Essayer de charger le lexique pour tester l'existence des mots ───

lexique = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Lexique" / "src"))
    from lectura_lexique import Lexique
    # Chercher la base de donnees
    # Chercher dans plusieurs emplacements possibles
    _base = Path(__file__).resolve().parent.parent.parent
    _workspace = Path(__file__).resolve().parent.parent.parent.parent
    db_candidates = [
        _workspace / "Lexique" / "lexique_lectura.db",
        _workspace / "Lexique" / "lexique_lectura_v3.db",
        _base / "Lexique" / "lexique_lectura_v4.db",
        _base / "Lexique" / "lexique_lectura.db",
    ]
    for db_path in db_candidates:
        if db_path.exists() and db_path.stat().st_size > 0:
            lexique = Lexique(str(db_path))
            print(f"Lexique charge: {db_path} ({db_path.stat().st_size // 1024}KB)")
            break
    if lexique is None:
        print("WARN: aucune base lexique trouvee, analyse OOV basee sur heuristiques")
except Exception as e:
    print(f"WARN: impossible de charger le lexique ({e}), analyse basee sur heuristiques")

print()


# ── Classification helpers ───────────────────────────────────────────

_DESACCENTUER = str.maketrans(
    "aaaaeeeeiiouuuycoa",
    "aaaaeeeeiiouuuycoa",
)
# Meilleure table
_DEACCENT = str.maketrans(
    "àâäéèêëïîôùûüÿçœæ",
    "aaaeeeeiioouuycoa",
)

def strip_accents(s: str) -> str:
    return s.translate(_DEACCENT)


def est_variante_accent(a: str, b: str) -> bool:
    """Vrai si a et b ne different que par les accents."""
    if len(a) != len(b):
        return False
    return strip_accents(a.lower()) == strip_accents(b.lower())


def edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[lb]


def classify_fp(orig: str, sys_: str | None) -> str:
    """Classifie le type de FP."""
    if sys_ is None:
        return "UNKNOWN"
    o, s = orig.lower(), sys_.lower()

    # Accent changes
    if est_variante_accent(o, s):
        n_acc_orig = sum(1 for c in o if c in "àâäéèêëïîôùûüÿç")
        n_acc_sys = sum(1 for c in s if c in "àâäéèêëïîôùûüÿç")
        if n_acc_sys > n_acc_orig:
            return "ACCENT_ADD"
        elif n_acc_sys < n_acc_orig:
            return "ACCENT_STRIP"
        else:
            return "ACCENT_CROSS"

    # Plural/agreement (differs only by trailing -s, -es, -e)
    if o.rstrip("s") == s.rstrip("s"):
        return "PLURAL_AGREE"
    if (o + "s" == s) or (s + "s" == o):
        return "PLURAL_AGREE"
    if (o + "es" == s) or (s + "es" == o):
        return "PLURAL_AGREE"
    if (o + "e" == s) or (s + "e" == o):
        return "PLURAL_AGREE"
    # Feminine forms: -e ending changes
    if strip_accents(o).rstrip("es") == strip_accents(s).rstrip("es"):
        return "PLURAL_AGREE"

    # Edit distance 1-2
    ed = edit_distance(o, s)
    if ed <= 2:
        return f"EDIT_{ed}"

    return f"EDIT_{ed}"


def est_oov(mot: str) -> bool:
    """Verifie si le mot est OOV (hors lexique)."""
    if lexique is not None:
        return not lexique.existe(mot) and not lexique.existe(mot.lower())
    # Heuristique sans lexique : considerer tous les mots courts
    # non-francais comme OOV
    return False


def est_nom_propre_probable(mot: str, pos_in_sent: int) -> bool:
    """Heuristique pour detecter un nom propre."""
    if len(mot) <= 1:
        return False
    # Commence par majuscule et pas en debut de phrase
    if pos_in_sent > 0 and mot[0].isupper() and mot[1:].islower():
        return True
    # Mot OOV en minuscule entoure d'autres OOV (nom propre non capitalise)
    if est_oov(mot):
        return True  # dans le corpus Wikipedia, beaucoup de NP en minuscule
    return False


# ── Extraire et classifier les FP ────────────────────────────────────

fps: list[dict] = []

for item in data:
    words_fautif = item["fautif"].split()
    words_gold = item["correct"].split()

    for e in item["erreurs_detail"]:
        if e["type"] != "FP":
            continue

        pos = e["pos"]
        orig = e["orig"]
        gold = e["gold"]
        sys_ = e["sys"]

        # Contexte: mots voisins dans la phrase fautive
        ctx_before = words_fautif[max(0, pos-3):pos]
        ctx_after = words_fautif[pos+1:pos+4]

        # Analyser les mots voisins pour OOV
        neighbors = ctx_before + ctx_after
        oov_neighbors = [w for w in neighbors if est_oov(w) and w.isalpha() and len(w) > 1]
        n_oov_neighbors = len(oov_neighbors)

        # Detecter les sequences OOV consecutives autour du mot
        # (indique un contexte etranger ou sequence de noms propres)
        oov_run = 0
        for offset in range(-3, 4):
            idx = pos + offset
            if 0 <= idx < len(words_fautif):
                w = words_fautif[idx]
                if est_oov(w) and w.isalpha() and len(w) > 1:
                    oov_run += 1

        category = classify_fp(orig, sys_)

        # Est-ce un nom propre ?
        is_proper = False
        # Verifier dans la phrase gold si le mot est capitalise
        if pos < len(words_gold):
            gold_word = words_gold[pos]
            if gold_word[0].isupper() and len(gold_word) > 1:
                is_proper = True
        # Verifier si OOV
        is_oov = est_oov(orig)

        # Verifier si mot etranger (anglais, allemand, etc.)
        # Heuristique: OOV + patterns typiques
        is_foreign = False
        if is_oov:
            lo = orig.lower()
            # Patterns courants de mots etrangers
            foreign_endings = ("ing", "tion", "ness", "ment", "ive", "ous",
                             "burg", "berg", "stadt", "stein", "heim",
                             "sky", "ski", "owski", "enko",
                             "abad", "pur", "nagar")
            if any(lo.endswith(e) for e in foreign_endings):
                is_foreign = True
            # Lettres rares en francais
            if any(c in lo for c in "wk") and (lexique is None or not lexique.existe(lo)):
                is_foreign = True

        fps.append({
            "idx": item["idx"],
            "pos": pos,
            "orig": orig,
            "gold": gold,
            "sys": sys_,
            "category": category,
            "is_proper": is_proper,
            "is_oov": is_oov,
            "is_foreign": is_foreign,
            "n_oov_neighbors": n_oov_neighbors,
            "oov_run": oov_run,
            "oov_neighbors": oov_neighbors,
            "ctx_before": ctx_before,
            "ctx_after": ctx_after,
            "fautif": item["fautif"],
        })


# ── Rapports ─────────────────────────────────────────────────────────

print("=" * 80)
print(f"TOTAL FP: {len(fps)}")
print("=" * 80)

# 1. Classification par type de correction
print("\n### 1. Type de correction FP ###\n")
cat_counts = Counter(fp["category"] for fp in fps)
for cat, n in cat_counts.most_common():
    pct = 100 * n / len(fps)
    print(f"  {cat:16s}: {n:3d}  ({pct:5.1f}%)")

# 2. FP causes par OOV (noms propres + mots etrangers)
print("\n### 2. FP impliquant des mots OOV ###\n")
oov_fps = [fp for fp in fps if fp["is_oov"]]
proper_fps = [fp for fp in fps if fp["is_proper"]]
foreign_fps = [fp for fp in fps if fp["is_foreign"]]
with_oov_ctx = [fp for fp in fps if fp["n_oov_neighbors"] >= 1]
with_oov_run = [fp for fp in fps if fp["oov_run"] >= 2]

print(f"  FP dont orig est OOV:                 {len(oov_fps):3d}  ({100*len(oov_fps)/len(fps):5.1f}%)")
print(f"  FP dont orig est nom propre probable:  {len(proper_fps):3d}  ({100*len(proper_fps)/len(fps):5.1f}%)")
print(f"  FP dont orig est mot etranger:         {len(foreign_fps):3d}  ({100*len(foreign_fps)/len(fps):5.1f}%)")
print(f"  FP avec >= 1 voisin OOV:               {len(with_oov_ctx):3d}  ({100*len(with_oov_ctx)/len(fps):5.1f}%)")
print(f"  FP avec >= 2 mots OOV consecutifs:     {len(with_oov_run):3d}  ({100*len(with_oov_run)/len(fps):5.1f}%)")

# 3. Croisement type x OOV
print("\n### 3. Croisement type de correction x OOV ###\n")
print(f"  {'Category':16s}  {'Total':>5s}  {'OOV':>5s}  {'Proper':>6s}  {'OOVctx':>6s}  {'OOVrun2+':>8s}")
print(f"  {'-'*16}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}")
for cat, _ in cat_counts.most_common():
    cat_fps = [fp for fp in fps if fp["category"] == cat]
    n_total = len(cat_fps)
    n_oov = sum(1 for fp in cat_fps if fp["is_oov"])
    n_proper = sum(1 for fp in cat_fps if fp["is_proper"])
    n_ctx = sum(1 for fp in cat_fps if fp["n_oov_neighbors"] >= 1)
    n_run = sum(1 for fp in cat_fps if fp["oov_run"] >= 2)
    print(f"  {cat:16s}  {n_total:5d}  {n_oov:5d}  {n_proper:6d}  {n_ctx:6d}  {n_run:8d}")

# 4. Exemples detailles par categorie
print("\n### 4. Exemples detailles par categorie ###\n")
for cat, _ in cat_counts.most_common():
    cat_fps = [fp for fp in fps if fp["category"] == cat]
    print(f"\n--- {cat} ({len(cat_fps)} FP) ---\n")
    for fp in cat_fps[:8]:
        oov_tag = " [OOV]" if fp["is_oov"] else ""
        proper_tag = " [PROPER]" if fp["is_proper"] else ""
        foreign_tag = " [FOREIGN]" if fp["is_foreign"] else ""
        ctx_tag = f" [OOVctx={fp['n_oov_neighbors']}]" if fp["n_oov_neighbors"] > 0 else ""

        ctx_b = " ".join(fp["ctx_before"])
        ctx_a = " ".join(fp["ctx_after"])
        print(f"  idx={fp['idx']:4d}  {fp['orig']!r:20s} -> {fp['sys']!r:20s}  (gold={fp['gold']!r}){oov_tag}{proper_tag}{foreign_tag}{ctx_tag}")
        print(f"           ...{ctx_b} [{fp['orig']}] {ctx_a}...")
        if fp["oov_neighbors"]:
            print(f"           OOV voisins: {fp['oov_neighbors']}")
        print()

# 5. Analyse specifique: "correction" de noms propres
print("\n### 5. Noms propres OOV corriges par le module ortho ###\n")
proper_oov = [fp for fp in fps if fp["is_oov"] and not fp["is_proper"]]
proper_real = [fp for fp in fps if fp["is_oov"]]
# Filtrer les FP ou l'orig est vraiment un nom propre ou mot etranger
# (pas une vraie faute d'orthographe qui serait corrigee)
# Un FP = gold == orig, donc le systeme n'aurait PAS du corriger
true_np_foreign = [fp for fp in fps if fp["is_oov"] and fp["gold"] == fp["orig"]]
print(f"  FP OOV ou gold==orig (correction inutile d'un mot correct): {len(true_np_foreign)}")
print()
for fp in true_np_foreign[:20]:
    ctx_b = " ".join(fp["ctx_before"])
    ctx_a = " ".join(fp["ctx_after"])
    oov_tag = f"[OOV ctx={fp['n_oov_neighbors']}, run={fp['oov_run']}]"
    print(f"  idx={fp['idx']:4d}  {fp['orig']!r:20s} -> {fp['sys']!r:20s}  {oov_tag}")
    print(f"           ...{ctx_b} [{fp['orig']}] {ctx_a}...")
    if fp["oov_neighbors"]:
        print(f"           OOV voisins: {fp['oov_neighbors']}")
    print()

# 6. Heuristique proposee: "bloquer si >= 2 OOV consecutifs"
print("\n### 6. Evaluation heuristique: bloquer correction si OOV run >= 2 ###\n")
preventable_2 = [fp for fp in fps if fp["oov_run"] >= 2]
not_preventable_2 = [fp for fp in fps if fp["oov_run"] < 2]
print(f"  FP qui seraient bloques:   {len(preventable_2):3d}  ({100*len(preventable_2)/len(fps):5.1f}%)")
print(f"  FP qui ne seraient PAS bloques: {len(not_preventable_2):3d}  ({100*len(not_preventable_2)/len(fps):5.1f}%)")

# 7. Heuristique: bloquer si l'orig est OOV
print("\n### 7. Evaluation heuristique: bloquer si orig est OOV ###\n")
preventable_oov = [fp for fp in fps if fp["is_oov"]]
print(f"  FP qui seraient bloques:   {len(preventable_oov):3d}  ({100*len(preventable_oov)/len(fps):5.1f}%)")
# Mais attention: combien de vrais positifs seraient aussi bloques?
# (TP = erreurs corrigees dont l'orig est aussi OOV)
tp_oov_count = 0
tp_total = 0
for item in data:
    words = item["fautif"].split()
    for e in item["erreurs_detail"]:
        if e["type"] != "FP" and e["sys"] != e["orig"]:
            # C'est un TP (le systeme a corrige et c'etait correct)
            # ou un FN (le systeme n'a pas corrige)
            pass
    # Compter les TP
    tp_total += item["tp"]

# Compter les TP qui impliquent un mot OOV corrige
tp_from_oov = 0
for item in data:
    words = item["fautif"].split()
    gold_words = item["correct"].split()
    for e in item["erreurs_detail"]:
        # TP = sys == gold != orig (correction correcte)
        if e["sys"] == e["gold"] and e["orig"] != e["gold"]:
            orig_w = e["orig"]
            if est_oov(orig_w):
                tp_from_oov += 1

print(f"  TP total (benchmark): {tp_total}")
print(f"  TP dont orig est OOV: {tp_from_oov}")
print(f"  -> Bloquer tous les OOV perdrait {tp_from_oov} TP")

# 8. Analyse des patterns specifiques
print("\n### 8. Patterns specifiques de correction abusive ###\n")

# 8a. Accent ajoute sur mot etranger/NP
accent_oov = [fp for fp in fps if fp["category"].startswith("ACCENT") and fp["is_oov"]]
print(f"  Accent ajoute/change sur mot OOV: {len(accent_oov)}")
for fp in accent_oov:
    print(f"    {fp['orig']!r:20s} -> {fp['sys']!r:20s}  (idx={fp['idx']})")

# 8b. Edit 1-2 lettres sur mot OOV (changement destructif)
edit_oov = [fp for fp in fps if fp["category"].startswith("EDIT") and fp["is_oov"]]
print(f"\n  Edit 1-2 lettres sur mot OOV: {len(edit_oov)}")
for fp in edit_oov:
    print(f"    {fp['orig']!r:20s} -> {fp['sys']!r:20s}  (idx={fp['idx']})")
    if fp["oov_neighbors"]:
        print(f"      OOV voisins: {fp['oov_neighbors']}")

# 8c. Accord/pluriel sur mot correct (pas OOV)
plural_in_lex = [fp for fp in fps if fp["category"] == "PLURAL_AGREE" and not fp["is_oov"]]
print(f"\n  Accord/pluriel sur mot in-lexique: {len(plural_in_lex)}")
for fp in plural_in_lex[:10]:
    ctx_b = " ".join(fp["ctx_before"])
    ctx_a = " ".join(fp["ctx_after"])
    print(f"    {fp['orig']!r:20s} -> {fp['sys']!r:20s}  ...{ctx_b} [{fp['orig']}] {ctx_a}...")

# 9. Resume et recommandations
print("\n" + "=" * 80)
print("RESUME ET RECOMMANDATIONS")
print("=" * 80)
print()

# Compter les FP evitables
accent_on_np_foreign = [fp for fp in fps if fp["category"].startswith("ACCENT") and (fp["is_oov"] or fp["n_oov_neighbors"] >= 1)]
edit_on_np_foreign = [fp for fp in fps if fp["category"].startswith("EDIT") and (fp["is_oov"] or fp["n_oov_neighbors"] >= 1)]
all_np_foreign = [fp for fp in fps if fp["is_oov"] or fp["n_oov_neighbors"] >= 1]

print(f"FP total: {len(fps)}")
print(f"FP lies a des mots OOV ou contexte OOV: {len(all_np_foreign)} ({100*len(all_np_foreign)/len(fps):.1f}%)")
print(f"  dont ACCENT sur OOV/NP:   {len(accent_on_np_foreign)}")
print(f"  dont EDIT sur OOV/NP:     {len(edit_on_np_foreign)}")
print()
print("Heuristiques testees:")
print(f"  A. Bloquer si orig OOV:           bloque {len(preventable_oov):3d} FP, perd {tp_from_oov} TP")
print(f"  B. Bloquer si OOV run >= 2:       bloque {len(preventable_2):3d} FP")
print(f"  C. Bloquer si >= 1 voisin OOV:    bloque {len(with_oov_ctx):3d} FP")

# Heuristique combinee: bloquer EDIT si voisin OOV, mais garder ACCENT
edit_blockable = [fp for fp in fps if fp["category"].startswith("EDIT") and fp["n_oov_neighbors"] >= 1]
print(f"  D. Bloquer EDIT si voisin OOV:    bloque {len(edit_blockable):3d} FP")
