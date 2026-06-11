"""Analyse des erreurs CONJ manquees par le Correcteur V6.

Charge le corpus_10000.jsonl, execute le correcteur V6 sur chaque phrase fautive,
et classifie les erreurs CONJ non corrigees par pattern.
"""

import json
import sys
import time
from collections import Counter, defaultdict

# -- Setup paths --
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Correcteur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Phonemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Graphemiseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Tokeniseur/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/G2P-Pipeline/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/P2G-Pipeline/src")
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Formules/src")

from lectura_lexique import Lexique
from lectura_correcteur.correcteur_v6 import CorrecteurV6
from lectura_correcteur._config import CorrecteurV6Config

# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
CORPUS = "/data/work/projets/lectura/workspace/Corpus/Correcteur/corpus_10000.jsonl"
LEXIQUE_PATH = "/data/work/projets/lectura/workspace/Lexique/bases/lexique_lectura.db"

print("Chargement du lexique...")
lex = Lexique(LEXIQUE_PATH)
print("Initialisation du correcteur V6...")
config = CorrecteurV6Config()
correcteur = CorrecteurV6(lex, config=config)
print("Correcteur pret.\n")

# --------------------------------------------------------------------------
# Chargement corpus
# --------------------------------------------------------------------------
print("Chargement du corpus...")
corpus = []
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            corpus.append(json.loads(line))
print(f"  {len(corpus)} phrases chargees.\n")

# --------------------------------------------------------------------------
# Extraire toutes les erreurs CONJ
# --------------------------------------------------------------------------
conj_errors = []  # (phrase_idx, erreur_dict, phrase_dict)
for idx, entry in enumerate(corpus):
    for err in entry.get("erreurs", []):
        if err.get("type") == "CONJ":
            conj_errors.append((idx, err, entry))

print(f"Total erreurs CONJ dans le corpus : {len(conj_errors)}")

# --------------------------------------------------------------------------
# Pronoms sujets
# --------------------------------------------------------------------------
PRONOMS_SUJETS = {
    "je", "j'", "j\u2019", "tu", "il", "elle", "on",
    "ils", "elles", "nous", "vous",
}

PRONOMS_SING = {"je", "j'", "j\u2019", "tu", "il", "elle", "on"}
PRONOMS_PLUR = {"ils", "elles", "nous", "vous"}

# Mots transparents entre sujet et verbe
TRANSPARENTS = {
    "ne", "n'", "n\u2019", "me", "m'", "m\u2019", "te", "t'", "t\u2019",
    "se", "s'", "s\u2019", "le", "la", "les", "l'", "l\u2019",
    "lui", "leur", "nous", "vous", "y", "en",
    "pas", "plus", "jamais", "rien", "point",
}

AUXILIAIRES = {
    "est", "sont", "a", "ont", "suis", "es", "sommes", "etes", "\u00eates",
    "avons", "avez", "etait", "\u00e9tait", "etaient", "\u00e9taient",
    "sera", "seront", "fut", "furent", "soit", "soient",
    "avait", "avaient", "aura", "auront",
}

# Determinants
DETERMINANTS = {
    "le", "la", "les", "l'", "l\u2019",
    "un", "une", "des", "du", "au", "aux",
    "ce", "cet", "cette", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "nos", "votre", "vos",
    "leur", "leurs",
}


def tokenize_simple(phrase):
    """Tokenisation simple pour analyse de contexte."""
    import re
    return re.findall(
        r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu|[Jj]usqu|[Qq]uelqu)"
        r"['\u2019]"
        r"|[\w]+(?:-[\w]+)*"
        r"|[^\s\w]+",
        phrase,
    )


def find_subject_before(tokens, err_pos):
    """Cherche le type de sujet avant la position de l'erreur.

    Retourne un tuple (type, sujet_mot) ou type est:
      'pronom' - pronom sujet direct
      'nom' - sujet nominal (avec determinant)
      'nom_propre' - nom propre (sans determinant, majuscule)
      'relatif' - apres 'qui'
      'aucun' - pas de sujet identifiable
    """
    if err_pos <= 0 or err_pos >= len(tokens):
        return ("aucun", "")

    # Chercher en arriere depuis la position de l'erreur
    j = err_pos - 1
    # Sauter les transparents
    while j >= 0 and tokens[j].lower() in TRANSPARENTS:
        j -= 1

    if j < 0:
        return ("aucun", "")

    mot = tokens[j].lower()

    # Pronom sujet
    if mot in PRONOMS_SUJETS:
        return ("pronom", mot)

    # Pronom relatif "qui"
    if mot == "qui":
        return ("relatif", "qui")

    # Nom avec determinant ?
    # Remonter pour chercher un determinant
    k = j
    while k >= 0:
        mk = tokens[k].lower()
        if mk in DETERMINANTS:
            # On a un groupe nominal
            nom_group = " ".join(tokens[k:err_pos])
            return ("nom", nom_group)
        # Si c'est un nom/adjectif, continuer a remonter
        if mk.isalpha() and len(mk) > 1:
            k -= 1
            continue
        break

    # Nom propre (majuscule, pas de determinant)
    if tokens[j][0].isupper() and tokens[j].lower() not in PRONOMS_SUJETS:
        return ("nom_propre", tokens[j])

    return ("aucun", tokens[j])


def classify_conj_error(err, entry, tokens_fautif, corrected_text):
    """Classifie une erreur CONJ manquee."""
    pos = err["position"]
    original = err["original"]      # forme correcte attendue
    perturbe = err["perturbe"]      # forme fautive dans le texte
    perturbe_low = perturbe.lower()
    original_low = original.lower()

    # Est-ce un auxiliaire ?
    is_aux = original_low in AUXILIAIRES or perturbe_low in AUXILIAIRES

    # Trouver le type de sujet
    subj_type, subj_mot = find_subject_before(tokens_fautif, pos)

    # Sous-categories
    if subj_type == "pronom":
        return ("pronom_sujet", subj_mot, is_aux)
    elif subj_type == "relatif":
        return ("relatif_qui", subj_mot, is_aux)
    elif subj_type == "nom":
        return ("nom_sujet", subj_mot, is_aux)
    elif subj_type == "nom_propre":
        return ("nom_propre", subj_mot, is_aux)
    else:
        return ("aucun_sujet", subj_mot, is_aux)


# --------------------------------------------------------------------------
# Execution du correcteur sur toutes les phrases avec erreurs CONJ
# --------------------------------------------------------------------------
print("Execution du correcteur V6 sur les phrases avec erreurs CONJ...")
print("(Ceci peut prendre quelques minutes)\n")

# Indexer les phrases uniques a corriger
phrase_indices = set(idx for idx, _, _ in conj_errors)
results = {}  # phrase_idx -> ResultatCorrection

t0 = time.time()
done = 0
total = len(phrase_indices)
for pidx in sorted(phrase_indices):
    entry = corpus[pidx]
    fautif = entry["fautif"]
    try:
        res = correcteur.corriger(fautif)
        results[pidx] = res
    except Exception as e:
        results[pidx] = None
        print(f"  ERREUR phrase {pidx}: {e}")
    done += 1
    if done % 100 == 0:
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  {done}/{total} phrases traitees ({rate:.0f} phrases/s)")

elapsed = time.time() - t0
print(f"\n  Termine : {total} phrases en {elapsed:.1f}s ({total/elapsed:.0f} phrases/s)\n")

# --------------------------------------------------------------------------
# Analyser les resultats
# --------------------------------------------------------------------------
corrected = []
missed = []

for pidx, err, entry in conj_errors:
    res = results.get(pidx)
    if res is None:
        missed.append((pidx, err, entry, "erreur_execution", "", False))
        continue

    original_word = err["original"].lower()
    perturbe_word = err["perturbe"].lower()
    corrige_text = res.phrase_corrigee.lower()

    # Tokeniser le texte corrige pour verifier mot par mot
    tokens_corrige = tokenize_simple(res.phrase_corrigee)
    tokens_corrige_low = [t.lower() for t in tokens_corrige]

    # Tokeniser le texte fautif
    tokens_fautif = tokenize_simple(entry["fautif"])
    tokens_fautif_low = [t.lower() for t in tokens_fautif]

    # L'erreur est "corrigee" si :
    # 1. Le mot perturbe a ete remplace par le mot original dans la sortie
    # 2. Ou si le mot perturbe n'est plus present ET le mot original est present

    pos = err["position"]
    is_corrected = False

    # Methode 1 : verifier la position directe
    if pos < len(tokens_corrige_low):
        if tokens_corrige_low[pos] == original_word:
            is_corrected = True
        elif tokens_corrige_low[pos] != perturbe_word:
            # A ete change mais pas vers la bonne forme - compter comme corrige
            # si c'est plus proche de l'original
            pass

    # Methode 2 : le mot perturbe est absent ET l'original est present
    if not is_corrected:
        if (original_word in tokens_corrige_low
                and perturbe_word not in tokens_corrige_low):
            is_corrected = True

    # Methode 3 : verifier position par position (le tokenizer peut decaler)
    if not is_corrected and pos < len(tokens_corrige_low):
        # Chercher dans une fenetre autour de la position
        for delta in range(-2, 3):
            p = pos + delta
            if 0 <= p < len(tokens_corrige_low):
                if tokens_corrige_low[p] == original_word:
                    is_corrected = True
                    break

    if is_corrected:
        corrected.append((pidx, err, entry))
    else:
        # Classifier l'erreur manquee
        cat = classify_conj_error(err, entry, tokens_fautif, res.phrase_corrigee)
        missed.append((pidx, err, entry, cat[0], cat[1], cat[2]))

# --------------------------------------------------------------------------
# Statistiques globales
# --------------------------------------------------------------------------
print("=" * 80)
print("STATISTIQUES GLOBALES — ERREURS CONJ")
print("=" * 80)
print(f"Total erreurs CONJ dans le corpus : {len(conj_errors)}")
print(f"Corrigees par V6                  : {len(corrected)} ({100*len(corrected)/len(conj_errors):.1f}%)")
print(f"Manquees                          : {len(missed)} ({100*len(missed)/len(conj_errors):.1f}%)")
print()

# --------------------------------------------------------------------------
# Repartition des erreurs manquees par categorie
# --------------------------------------------------------------------------
cat_counter = Counter()
cat_aux_counter = Counter()
cat_examples = defaultdict(list)

for pidx, err, entry, cat, subj, is_aux in missed:
    cat_counter[cat] += 1
    if is_aux:
        cat_aux_counter[cat] += 1
    if len(cat_examples[cat]) < 30:  # garder assez d'exemples
        cat_examples[cat].append((pidx, err, entry, subj, is_aux))

print("-" * 80)
print("REPARTITION DES ERREURS MANQUEES PAR CATEGORIE")
print("-" * 80)
print()

cat_labels = {
    "pronom_sujet": "(a) Pronom sujet present mais non corrige",
    "nom_sujet": "(b) Sujet nominal (DET + NOM)",
    "nom_propre": "(c) Nom propre (sans determinant)",
    "relatif_qui": "(d) Pronom relatif 'qui'",
    "aucun_sujet": "(e) Pas de sujet identifiable",
    "erreur_execution": "(f) Erreur d'execution",
}

for cat, count in cat_counter.most_common():
    label = cat_labels.get(cat, cat)
    aux_count = cat_aux_counter.get(cat, 0)
    pct = 100 * count / len(missed) if missed else 0
    pct_total = 100 * count / len(conj_errors) if conj_errors else 0
    print(f"  {label}")
    print(f"    Total: {count} ({pct:.1f}% des manquees, {pct_total:.1f}% du total)")
    if aux_count:
        print(f"    Dont auxiliaires (etre/avoir): {aux_count}")
    print()

# --------------------------------------------------------------------------
# (a) Pronom sujet present mais non corrige — POURQUOI ?
# --------------------------------------------------------------------------
print("=" * 80)
print("CATEGORIE (a) : PRONOM SUJET PRESENT MAIS NON CORRIGE")
print("=" * 80)
print()

pronom_examples = cat_examples.get("pronom_sujet", [])
pronom_why = Counter()

for pidx, err, entry, subj, is_aux in pronom_examples[:30]:
    fautif = entry["fautif"]
    correct = entry["correct"]
    tokens_f = tokenize_simple(fautif)
    pos = err["position"]
    perturbe = err["perturbe"]
    original = err["original"]

    # Analyser pourquoi la regle n'a pas fire
    reasons = []

    # Verifier la distance sujet-verbe
    j = pos - 1
    distance = 0
    while j >= 0 and tokens_f[j].lower() in TRANSPARENTS:
        j -= 1
        distance += 1
    if j >= 0 and tokens_f[j].lower() in PRONOMS_SUJETS:
        actual_distance = pos - j
    else:
        actual_distance = -1
        reasons.append("pronom pas directement avant (token intermediaire non-transparent)")

    # La regle V6 cherche dans une fenetre de 3 mots
    if actual_distance > 3:
        reasons.append(f"distance sujet-verbe = {actual_distance} (fenetre max = 3)")

    # Le verbe doit exister dans le lexique
    if not lex.existe(perturbe.lower()):
        reasons.append(f"'{perturbe}' n'existe pas dans le lexique (OOV)")
    else:
        infos = lex.info(perturbe.lower())
        verb_entries = [e for e in infos if e.get("cgram", "").startswith("VER") or e.get("cgram") == "AUX"]
        conj_entries = [e for e in verb_entries if e.get("mode", "") in ("ind", "sub", "con")]
        if not verb_entries:
            reasons.append(f"'{perturbe}' pas reconnu comme VER/AUX dans le lexique")
        elif not conj_entries:
            reasons.append(f"'{perturbe}' pas en mode ind/sub/con (modes: {set(e.get('mode','') for e in verb_entries)})")

    # G2P POS pourrait bloquer
    reasons.append("(possible: G2P POS ne confirme pas VER/AUX, ou :par bloque)")

    # Le verbe est dans _MOTS_PROTEGES ? (non, accord_sujet_verbe bypasse)

    if not reasons:
        reasons.append("raison inconnue")

    pronom_why["; ".join(reasons)] += 1

    print(f"  [{pidx}] {perturbe} -> {original} (sujet: {subj})")
    print(f"    Fautif:  {fautif}")
    print(f"    Correct: {correct}")
    print(f"    Raisons possibles: {'; '.join(reasons)}")
    print()

if pronom_why:
    print("  --- Resume des raisons (pronom sujet) ---")
    for reason, cnt in pronom_why.most_common():
        print(f"    {cnt}x : {reason}")
    print()

# --------------------------------------------------------------------------
# (b) Sujet nominal — exemples et statistiques
# --------------------------------------------------------------------------
print("=" * 80)
print("CATEGORIE (b) : SUJET NOMINAL (GAIN POTENTIEL)")
print("=" * 80)
print()

nom_examples = cat_examples.get("nom_sujet", [])
nom_patterns = Counter()

for pidx, err, entry, subj, is_aux in nom_examples[:25]:
    fautif = entry["fautif"]
    perturbe = err["perturbe"]
    original = err["original"]

    # Pattern: singulier->pluriel ou pluriel->singulier ?
    orig_low = original.lower()
    pert_low = perturbe.lower()
    if pert_low.endswith("ent") and not orig_low.endswith("ent"):
        pattern = "3s->3p (ex: mange->mangent)"
    elif orig_low.endswith("ent") and not pert_low.endswith("ent"):
        pattern = "3p->3s (ex: mangent->mange)"
    elif pert_low.endswith("s") and not orig_low.endswith("s"):
        pattern = "ajoute -s"
    elif orig_low.endswith("s") and not pert_low.endswith("s"):
        pattern = "retire -s"
    else:
        pattern = f"{pert_low}->{orig_low}"

    nom_patterns[pattern] += 1

    print(f"  [{pidx}] {perturbe} -> {original}  (sujet: {subj})")
    print(f"    Fautif:  {fautif}")
    print(f"    Correct: {entry['correct']}")
    print()

print(f"  --- Patterns des erreurs NOM sujet (sur tous les {cat_counter.get('nom_sujet', 0)} cas) ---")
# Recalculer les patterns sur TOUS les cas (pas juste les exemples)
all_nom_patterns = Counter()
all_nom_aux = 0
for pidx, err, entry, cat, subj, is_aux in missed:
    if cat != "nom_sujet":
        continue
    orig_low = err["original"].lower()
    pert_low = err["perturbe"].lower()
    if is_aux:
        all_nom_aux += 1
    if pert_low.endswith("ent") and not orig_low.endswith("ent"):
        all_nom_patterns["3s->3p (mange->mangent)"] += 1
    elif orig_low.endswith("ent") and not pert_low.endswith("ent"):
        all_nom_patterns["3p->3s (mangent->mange)"] += 1
    elif pert_low.endswith("ons") and not orig_low.endswith("ons"):
        all_nom_patterns["Xp->1p"] += 1
    elif pert_low.endswith("ez") and not orig_low.endswith("ez"):
        all_nom_patterns["Xp->2p"] += 1
    else:
        all_nom_patterns[f"autre ({pert_low}->{orig_low})"] += 1

for pat, cnt in all_nom_patterns.most_common():
    print(f"    {cnt}x : {pat}")
print(f"    Dont auxiliaires: {all_nom_aux}")
print()

# --------------------------------------------------------------------------
# (c) Nom propre — exemples
# --------------------------------------------------------------------------
print("=" * 80)
print("CATEGORIE (c) : NOM PROPRE (sans determinant)")
print("=" * 80)
print()

np_examples = cat_examples.get("nom_propre", [])
for pidx, err, entry, subj, is_aux in np_examples[:15]:
    print(f"  [{pidx}] {err['perturbe']} -> {err['original']} (sujet: {subj})")
    print(f"    Fautif: {entry['fautif']}")
    print()

# --------------------------------------------------------------------------
# (d) Pronom relatif "qui"
# --------------------------------------------------------------------------
print("=" * 80)
print("CATEGORIE (d) : PRONOM RELATIF 'QUI'")
print("=" * 80)
print()

qui_examples = cat_examples.get("relatif_qui", [])
for pidx, err, entry, subj, is_aux in qui_examples[:15]:
    print(f"  [{pidx}] {err['perturbe']} -> {err['original']}")
    print(f"    Fautif: {entry['fautif']}")
    print()

# --------------------------------------------------------------------------
# (e) Pas de sujet identifiable
# --------------------------------------------------------------------------
print("=" * 80)
print("CATEGORIE (e) : PAS DE SUJET IDENTIFIABLE")
print("=" * 80)
print()

aucun_examples = cat_examples.get("aucun_sujet", [])
for pidx, err, entry, subj, is_aux in aucun_examples[:20]:
    print(f"  [{pidx}] {err['perturbe']} -> {err['original']} (dernier mot: '{subj}')")
    print(f"    Fautif: {entry['fautif']}")
    print()

# --------------------------------------------------------------------------
# Top 20 transformations manquees les plus frequentes
# --------------------------------------------------------------------------
print("=" * 80)
print("TOP 20 TRANSFORMATIONS CONJ MANQUEES (perturbe -> original)")
print("=" * 80)
print()

transform_counter = Counter()
for pidx, err, entry, cat, subj, is_aux in missed:
    pert = err["perturbe"].lower()
    orig = err["original"].lower()
    transform_counter[(pert, orig)] += 1

for (pert, orig), cnt in transform_counter.most_common(20):
    print(f"  {cnt:3d}x  {pert:20s} -> {orig}")

# --------------------------------------------------------------------------
# Top 20 verbes (lemme) les plus souvent manques
# --------------------------------------------------------------------------
print()
print("=" * 80)
print("TOP 20 VERBES (forme fautive) LES PLUS SOUVENT MANQUES")
print("=" * 80)
print()

verb_counter = Counter()
for pidx, err, entry, cat, subj, is_aux in missed:
    verb_counter[err["perturbe"].lower()] += 1

for verb, cnt in verb_counter.most_common(20):
    print(f"  {cnt:3d}x  {verb}")

# --------------------------------------------------------------------------
# Analyse detaillee: quel % des CONJ corrigees vs manquees sont des auxiliaires
# --------------------------------------------------------------------------
print()
print("=" * 80)
print("AUXILIAIRES vs VERBES LEXICAUX")
print("=" * 80)
print()

aux_missed = sum(1 for _, _, _, _, _, is_aux in missed if is_aux)
aux_corrected = 0
for pidx, err, entry in corrected:
    orig = err["original"].lower()
    pert = err["perturbe"].lower()
    if orig in AUXILIAIRES or pert in AUXILIAIRES:
        aux_corrected += 1

print(f"  Corrigees - auxiliaires: {aux_corrected}, lexicaux: {len(corrected)-aux_corrected}")
print(f"  Manquees  - auxiliaires: {aux_missed}, lexicaux: {len(missed)-aux_missed}")

# --------------------------------------------------------------------------
# Resume final compact
# --------------------------------------------------------------------------
print()
print("=" * 80)
print("RESUME FINAL")
print("=" * 80)
print()
print(f"Total CONJ: {len(conj_errors)}")
print(f"Corrigees:  {len(corrected)} ({100*len(corrected)/len(conj_errors):.1f}%)")
print(f"Manquees:   {len(missed)} ({100*len(missed)/len(conj_errors):.1f}%)")
print()
print("Repartition des manquees:")
for cat, count in cat_counter.most_common():
    label = cat_labels.get(cat, cat)
    print(f"  {count:4d} ({100*count/len(missed):5.1f}%)  {label}")
print()
print("Gain potentiel si NOM sujet corrige:", cat_counter.get("nom_sujet", 0),
      f"erreurs ({100*cat_counter.get('nom_sujet', 0)/len(conj_errors):.1f}% du total)")
print("Gain potentiel si NOM propre corrige:", cat_counter.get("nom_propre", 0),
      f"erreurs ({100*cat_counter.get('nom_propre', 0)/len(conj_errors):.1f}% du total)")
print("Gain potentiel si relatif 'qui' corrige:", cat_counter.get("relatif_qui", 0),
      f"erreurs ({100*cat_counter.get('relatif_qui', 0)/len(conj_errors):.1f}% du total)")
print()
recall_potential = len(corrected) + cat_counter.get("nom_sujet", 0) + cat_counter.get("nom_propre", 0) + cat_counter.get("relatif_qui", 0)
print(f"Recall potentiel maximal (si toutes ces categories corrigees) : {100*recall_potential/len(conj_errors):.1f}%")
print(f"  (actuel: {100*len(corrected)/len(conj_errors):.1f}%)")
