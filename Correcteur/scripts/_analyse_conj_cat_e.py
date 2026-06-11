"""Analyse detaillee de la categorie (e) - pas de sujet identifiable.
Comprendre ce qui precede le verbe dans ces cas."""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Correcteur/src')
sys.path.insert(0, '/data/work/projets/lectura/workspace/Modules/Lexique/src')

_TOKEN_RE = re.compile(
    r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu|[Jj]usqu|[Qq]uelqu)"
    r"['\u2019]"
    r"|[\w]+(?:-[\w]+)*"
    r"|[^\s\w]+",
)

CORPUS = "/data/work/projets/lectura/workspace/Corpus/Correcteur/corpus_10000.jsonl"

corpus = []
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            corpus.append(json.loads(line))

# On recree la meme liste d'erreurs CONJ que le script principal
conj_errors = []
for idx, entry in enumerate(corpus):
    for err in entry.get("erreurs", []):
        if err.get("type") == "CONJ":
            conj_errors.append((idx, err, entry))

# Importer les memes categories
PRONOMS_SUJETS = {"je", "j'", "j\u2019", "tu", "il", "elle", "on", "ils", "elles", "nous", "vous"}
TRANSPARENTS = {
    "ne", "n'", "n\u2019", "me", "m'", "m\u2019", "te", "t'", "t\u2019",
    "se", "s'", "s\u2019", "le", "la", "les", "l'", "l\u2019",
    "lui", "leur", "nous", "vous", "y", "en",
    "pas", "plus", "jamais", "rien", "point",
}
DETERMINANTS = {
    "le", "la", "les", "l'", "l\u2019",
    "un", "une", "des", "du", "au", "aux",
    "ce", "cet", "cette", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "nos", "votre", "vos",
    "leur", "leurs",
}
AUXILIAIRES_SET = {
    "est", "sont", "a", "ont", "suis", "es", "sommes", "\u00eates",
    "avons", "avez", "\u00e9tait", "avait", "avaient", "\u00e9taient",
    "sera", "seront", "fut", "furent", "soit", "soient",
    "aura", "auront",
}

# Analyser chaque erreur cat (e) : qu'est-ce qui precede ?
patterns = Counter()
examples_by_pattern = {}

for idx, err, entry in conj_errors:
    tokens = _TOKEN_RE.findall(entry["fautif"])
    pos = err["position"]
    if pos >= len(tokens):
        continue

    # Trouver le contexte avant
    j = pos - 1
    while j >= 0 and tokens[j].lower() in TRANSPARENTS:
        j -= 1
    if j < 0:
        word_before = "(debut_phrase)"
    else:
        word_before = tokens[j].lower()

    # Classifier: est-ce un sujet nominal, pronom, etc.?
    if word_before in PRONOMS_SUJETS:
        continue  # cat (a), pas nous
    if word_before == "qui":
        continue  # cat (d)

    # Remonter pour chercher determinant
    k = j
    found_det = False
    while k >= 0:
        mk = tokens[k].lower()
        if mk in DETERMINANTS:
            found_det = True
            break
        if mk.isalpha() and len(mk) > 1:
            k -= 1
            continue
        break

    if found_det:
        continue  # cat (b)

    # Categorie (e) - pas de sujet identifiable
    # Sous-classifier
    if word_before in AUXILIAIRES_SET:
        pattern = "apres_auxiliaire"
    elif word_before == "(debut_phrase)":
        pattern = "debut_phrase"
    elif word_before.endswith(("'", "\u2019")):
        pattern = "apres_elision"
    elif tokens[j][0].isupper() if j >= 0 else False:
        pattern = "apres_nom_propre"
    elif word_before in ("et", "ou", "mais", "donc", "car", "ni", "puis"):
        pattern = "apres_conjonction"
    elif word_before in ("de", "d'", "du", "des", "a", "au", "aux", "en", "dans", "sur", "sous", "par", "pour", "avec", "sans", "chez", "entre"):
        pattern = "apres_preposition"
    elif word_before == ",":
        pattern = "apres_virgule"
    elif word_before in ("-", ".", ";", ":"):
        pattern = "apres_ponctuation"
    else:
        pattern = f"apres_autre ({word_before})"

    patterns[pattern] += 1
    if pattern not in examples_by_pattern:
        examples_by_pattern[pattern] = []
    if len(examples_by_pattern[pattern]) < 5:
        examples_by_pattern[pattern].append((idx, err, entry))

print("=" * 80)
print("SOUS-CLASSIFICATION CATEGORIE (e) : PAS DE SUJET IDENTIFIABLE")
print("=" * 80)
print(f"\nTotal cas cat (e): {sum(patterns.values())}")
print()

for pat, cnt in patterns.most_common():
    print(f"  {cnt:4d}x  {pat}")
    exs = examples_by_pattern.get(pat, [])
    for idx, err, entry in exs[:3]:
        print(f"         [{idx}] {err['perturbe']} -> {err['original']}")
        fautif = entry['fautif']
        if len(fautif) > 120:
            fautif = fautif[:120] + "..."
        print(f"         {fautif}")
    print()
