"""Post-traitement pour le modele P2G unifie V6.

Strategies :
- corriger_p2g() : correction par mot via predictions morpho
- corriger_phrase_v2() : correction contextuelle inter-mots + lexique
- forcer_coherence_ortho_morpho() : coherence ortho-morpho par regles
- corriger_phrase_v3() : pipeline V6 core (coherence morpho + accents)

Note : les formules (nombres, sigles) et noms propres sont dans
lectura-p2g (pipeline couche 2).
"""

from __future__ import annotations

from lectura_graphemiseur._chargeur import (
    homophones_pos as _load_homophones_pos,
    determinants_pluriel as _load_plur_det,
    determinants_singulier as _load_sing_det,
    invariables_pluriel as _load_no_plural_s,
)

_HOMOPHONES_POS = _load_homophones_pos()
_PLUR_DET = _load_plur_det()
_SING_DET = _load_sing_det()
_NO_PLURAL_S = _load_no_plural_s()


# ── Correction par mot (v1, morpho-based) ──────────────────────────

def corriger_p2g(
    ortho: str,
    pos: str = "",
    morpho: dict[str, str] | None = None,
) -> str:
    """Corrige l'orthographe P2G en utilisant les prédictions morpho.

    Règles appliquées :
    - Number=Plur : ajoute -s si absent (sauf si finit par s/x/z)
    - Number=Sing : retire -s final si en trop
    - Gender=Fem  : ajoute -e si participe/adj sans -e final
    - Gender=Masc : retire -e final si en trop sur participe/adj
    - Person=3 + Number=Plur + VerbForm=Fin : terminaison -ent
    """
    if not ortho:
        return ortho

    # ── Correction homophones POS-aware (priorité haute) ──
    key = (ortho.lower(), pos)
    if key in _HOMOPHONES_POS:
        return _HOMOPHONES_POS[key]

    if morpho is None:
        return ortho

    number = morpho.get("Number", "_")
    gender = morpho.get("Gender", "_")
    person = morpho.get("Person", "_")
    verbform = morpho.get("VerbForm", "_")

    # Ne pas toucher les mots fonctionnels
    if pos in ("PRE", "CON", "ART:def", "ART:ind", "PRO:rel", "PRO:dem",
               "PRO:per", "ADV", "INT"):
        return ortho

    result = ortho

    # ── Verbes 3pl : ajouter -nt sur forme en -e ──
    if (
        number == "Plur"
        and person == "3"
        and verbform == "Fin"
        and pos in ("VER", "AUX")
    ):
        if result.endswith("e") and not result.endswith(("ent", "nt")):
            result = result + "nt"
        return result

    # ── Ne pas modifier les verbes conjugués (1sg/2sg finissent en -s/-x) ──
    if verbform == "Fin" and pos in ("VER", "AUX"):
        return result

    # ── Féminin + Pluriel : mot doit finir par -es ──
    if number == "Plur" and gender == "Fem" and pos in ("ADJ", "VER", "NOM"):
        if result.endswith("es"):
            return result  # déjà correct
        if result.endswith("e"):
            return result + "s"  # ajouter -s
        if result.endswith("s"):
            return result  # finit déjà par -s, ne pas ajouter -e
        # Ni -e ni -s : ajouter -es
        return result + "es"

    # ── Pluriel seul : ajouter -s ──
    if number == "Plur" and pos in ("NOM", "ADJ"):
        if not result.endswith(("s", "x", "z")):
            return result + "s"
        return result

    # ── Féminin seul (singulier) : ajouter -e ──
    if (
        gender == "Fem"
        and number != "Plur"
        and pos in ("ADJ", "VER", "NOM")
        and verbform in ("Part", "_")
        and not result.endswith(("e", "ée"))
    ):
        return result + "e"

    # ── Masculin : retirer -ée → -é (participe) ──
    if (
        gender == "Masc"
        and pos in ("VER", "ADJ")
        and verbform == "Part"
        and result.endswith("ée")
    ):
        return result[:-1]

    return result


# ── Coherence ortho-morpho (Phase 3c) ────────────────────────────

# Suffixes qui empechent le retrait du -s final (mots finissant naturellement par -s)
_NO_STRIP_S_ENDINGS = ("ss", "is", "us", "as", "os", "ès", "ais", "ois", "urs")


def forcer_coherence_ortho_morpho(
    ortho: str,
    pos: str,
    morpho: dict[str, str],
    lexique: set[str] | frozenset[str] | None = None,
) -> str:
    """Force la coherence entre ortho predite et morpho predite.

    Ne modifie que si la morpho contredit clairement l'ortho.
    Le lexique sert de filtre : ne corriger que si la forme corrigee existe.

    Regles (par priorite) :
    1. plur_sans_s : NOM/ADJ + Number=Plur + pas de -s/-x/-z -> ajouter -s
    2. sing_avec_s : NOM/ADJ + Number=Sing + -s spurieux -> retirer -s
    3. 3pl_sans_ent : VER/AUX + Person=3 + Plur + Fin + pas -ent/-ont/-nt -> +nt
    4. fem_sans_e : ADJ + Gender=Fem + pas -e -> ajouter -e
    5. masc_avec_ee : VER/ADJ + Gender=Masc + Part + -ee -> retirer -e
    6. inf_forme : VER + VerbForm=Inf + -e final -> -er
    7. 1pl/2pl : VER + Person=1/2 + Plur + Fin -> -ons/-ez
    8. 2sg_sans_s : VER/AUX + Person=2 + Sing + Fin + Mood!=Imp + pas -s/-x -> +s
    """
    if not ortho:
        return ortho

    number = morpho.get("Number", "_")
    gender = morpho.get("Gender", "_")
    person = morpho.get("Person", "_")
    verbform = morpho.get("VerbForm", "_")
    mood = morpho.get("Mood", "_")

    # Ne pas toucher les mots fonctionnels ni les mots trop courts
    # (sauf verbes : "a" -> "as" pour 2sg)
    if len(ortho) < 2 and pos not in ("VER", "AUX"):
        return ortho
    if pos in ("PRE", "CON", "ART:def", "ART:ind", "PRO:rel", "PRO:dem",
               "PRO:per", "ADV", "INT", "PRE:det", "ADJ:pos", "ADJ:dem",
               "ADJ:ind", "ADJ:num"):
        return ortho

    def _in_lexique(form: str) -> bool:
        if lexique is None:
            return True
        return form.lower() in lexique

    # ── Regle 1 : plur_sans_s ──
    if (
        pos in ("NOM", "ADJ")
        and number == "Plur"
        and not ortho.endswith(("s", "x", "z"))
        and ortho.lower() not in _NO_PLURAL_S
    ):
        candidate = ortho + "s"
        if _in_lexique(candidate):
            return candidate

    # ── Regle 2 : sing_avec_s ──
    if (
        pos in ("NOM", "ADJ")
        and number == "Sing"
        and ortho.endswith("s")
        and not ortho.lower().endswith(_NO_STRIP_S_ENDINGS)
    ):
        candidate = ortho[:-1]
        if candidate and _in_lexique(candidate):
            return candidate

    # ── Regle 3 : 3pl_sans_ent ──
    if (
        pos in ("VER", "AUX")
        and person == "3"
        and number == "Plur"
        and verbform == "Fin"
        and not ortho.endswith(("ent", "ont", "nt"))
    ):
        # Ajouter -nt (ex: "paie" -> "paient")
        candidate = ortho + "nt"
        if _in_lexique(candidate):
            return candidate

    # ── Regle 4 : fem_sans_e (ADJ) ──
    # Exclure les mots finissant deja par -e ou -es (deja feminises)
    if (
        pos == "ADJ"
        and gender == "Fem"
        and not ortho.endswith(("e", "es", "ée", "ées"))
    ):
        candidate = ortho + "e"
        if _in_lexique(candidate):
            return candidate

    # ── Regle 4b : fem_sans_e (VER + Part) ──
    # Participe passe feminin : rattache -> rattachee
    # Exclure les mots finissant deja par -e ou -es (deja feminises)
    if (
        pos == "VER"
        and verbform == "Part"
        and gender == "Fem"
        and not ortho.endswith(("e", "es", "ée", "ées"))
    ):
        candidate = ortho + "e"
        if _in_lexique(candidate):
            return candidate

    # ── Regle 5 : masc_avec_ee ──
    if (
        pos in ("VER", "ADJ")
        and gender == "Masc"
        and verbform == "Part"
        and ortho.endswith("ée")
    ):
        candidate = ortho[:-1]  # "diffusee" -> "diffuse" ; "ée" -> "é"
        if _in_lexique(candidate):
            return candidate

    # ── Regle 6 : inf_forme ──
    if (
        pos == "VER"
        and verbform == "Inf"
        and ortho.endswith("e")
        and not ortho.endswith(("er", "re"))
    ):
        candidate = ortho[:-1] + "er"  # "note" -> "noter"
        if _in_lexique(candidate):
            return candidate

    # ── Regle 7 : 1pl/2pl ──
    if (
        pos in ("VER", "AUX")
        and number == "Plur"
        and verbform == "Fin"
    ):
        if person == "1" and not ortho.endswith("ons"):
            # Essayer plusieurs formes : "mange" -> "mangeons", "mange" -> "mangons"
            candidates = []
            if ortho.endswith("e"):
                candidates.append(ortho + "ons")    # mangeons
                candidates.append(ortho[:-1] + "ons")  # mangons
            else:
                candidates.append(ortho + "ons")
            for cand in candidates:
                if _in_lexique(cand):
                    return cand

        if person == "2" and not ortho.endswith("ez"):
            candidates = []
            if ortho.endswith("e"):
                candidates.append(ortho[:-1] + "ez")  # mangez
                candidates.append(ortho + "ez")
            else:
                candidates.append(ortho + "ez")
            for cand in candidates:
                if _in_lexique(cand):
                    return cand

    # ── Regle 8 : 2sg_sans_s ──
    # En francais, les verbes a la 2e personne du singulier finissent
    # toujours par -s ou -x (tu manges, tu peux), SAUF a l'imperatif
    # des verbes du 1er groupe (mange !, donne !).
    if (
        pos in ("VER", "AUX")
        and person == "2"
        and number == "Sing"
        and verbform == "Fin"
        and mood != "Imp"
        and not ortho.endswith(("s", "x"))
    ):
        candidate = ortho + "s"
        if _in_lexique(candidate):
            return candidate

    return ortho


# ── Correction contextuelle inter-mots (v2) ────────────────────────

def corriger_phrase_v2(
    ortho_words: list[str],
    pos_tags: list[str],
    lexique: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Corrige une phrase en exploitant le contexte inter-mots.

    Regles appliquees :
    - Det. pluriel + NOM/ADJ sans -s -> ajouter -s (si forme dans lexique)
    - Det. pluriel + ADJ + NOM sans -s -> idem
    - ils/elles + VER en -e -> ajouter -nt (si forme dans lexique)
    - Det. singulier + NOM/ADJ avec -s spurieux -> retirer -s (si forme dans lexique)
    - NOM + ADJ : propager le pluriel (si NOM finit par -s, ADJ doit aussi)
    - ADJ + NOM : propager le pluriel (si ADJ finit par -s, NOM doit aussi)
    - nous/vous + VER : accorder le verbe (nous -> -ons, vous -> -ez)

    Le lexique (set de formes en minuscules) sert de filtre de securite.
    Sans lexique, les regles sont quand meme appliquees mais sans verification.
    """
    if not ortho_words:
        return ortho_words

    result = list(ortho_words)
    n = len(result)

    for i in range(n):
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]

        # ── Regle 1 : Det. pluriel -> NOM/ADJ doit avoir -s ──
        if i > 0 and pos in ("NOM", "ADJ"):
            prev = result[i - 1].lower()
            if (
                prev in _PLUR_DET
                and not curr.endswith(("s", "x", "z"))
                and len(curr) > 1
                and curr.lower() not in _NO_PLURAL_S
            ):
                candidate = curr + "s"
                if lexique is None or candidate.lower() in lexique:
                    result[i] = candidate

        # ── Regle 2 : Det. pluriel + ADJ + NOM ──
        if (
            i > 1
            and pos == "NOM"
            and pos_tags[i - 1] == "ADJ"
        ):
            prev2 = result[i - 2].lower()
            if (
                prev2 in _PLUR_DET
                and not result[i].endswith(("s", "x", "z"))
                and len(result[i]) > 1
                and result[i].lower() not in _NO_PLURAL_S
            ):
                candidate = result[i] + "s"
                if lexique is None or candidate.lower() in lexique:
                    result[i] = candidate

        # ── Regle 3 : ils/elles + VER en -e -> -ent ──
        if (
            i > 0
            and pos in ("VER", "AUX")
            and result[i - 1].lower() in ("ils", "elles")
            and curr.endswith("e")
            and not curr.endswith(("ent", "nt"))
        ):
            candidate = curr + "nt"
            if lexique is None or candidate.lower() in lexique:
                result[i] = candidate

        # ── Regle 4 : Det. singulier + NOM/ADJ avec -s spurieux -> retirer -s ──
        if (
            i > 0
            and pos in ("NOM", "ADJ")
            and result[i - 1].lower() in _SING_DET
            and curr.endswith("s")
            and len(curr) > 2
            and not curr.lower().endswith(("ss", "is", "us", "as", "os"))
        ):
            candidate = curr[:-1]
            if candidate and (lexique is None or candidate.lower() in lexique):
                result[i] = candidate

        # ── Regle 5 : NOM + ADJ -> propager le pluriel ──
        if (
            i > 0
            and pos == "ADJ"
            and pos_tags[i - 1] in ("NOM",)
            and result[i - 1].endswith(("s", "x"))
            and not curr.endswith(("s", "x", "z"))
            and len(curr) > 1
            and curr.lower() not in _NO_PLURAL_S
        ):
            candidate = curr + "s"
            if lexique is None or candidate.lower() in lexique:
                result[i] = candidate

        # ── Regle 6 : ADJ + NOM -> propager le pluriel ──
        if (
            i > 0
            and pos == "NOM"
            and pos_tags[i - 1] == "ADJ"
            and result[i - 1].endswith(("s", "x"))
            and not curr.endswith(("s", "x", "z"))
            and len(curr) > 1
            and curr.lower() not in _NO_PLURAL_S
        ):
            candidate = curr + "s"
            if lexique is None or candidate.lower() in lexique:
                result[i] = candidate

        # ── Regle 7 : nous + VER en -e -> -ons ──
        if (
            i > 0
            and pos in ("VER", "AUX")
            and result[i - 1].lower() == "nous"
            and curr.endswith("e")
            and not curr.endswith(("ons", "ent", "ez"))
            and len(curr) > 1
        ):
            # Essayer curr+"ons" (mangeons) puis curr[:-1]+"ons" (mangons)
            for cand in (curr + "ons", curr[:-1] + "ons"):
                if lexique is None or cand.lower() in lexique:
                    result[i] = cand
                    break

        # ── Regle 8 : vous + VER en -e -> -ez ──
        if (
            i > 0
            and pos in ("VER", "AUX")
            and result[i - 1].lower() == "vous"
            and curr.endswith("e")
            and not curr.endswith(("ez", "ent", "ons"))
            and len(curr) > 1
        ):
            # Essayer curr[:-1]+"ez" (mangez) puis curr+"ez" (mangeez)
            for cand in (curr[:-1] + "ez", curr + "ez"):
                if lexique is None or cand.lower() in lexique:
                    result[i] = cand
                    break

    return result


def corriger_phrase(
    ortho_words: list[str],
    pos_tags: list[str],
    morpho_features: dict[str, list[str]],
) -> list[str]:
    """Corrige tous les mots d'une phrase (v1, morpho-based)."""
    result = []
    n = len(ortho_words)
    for i in range(n):
        word_morpho = {
            feat: vals[i] if i < len(vals) else "_"
            for feat, vals in morpho_features.items()
        }
        pos = pos_tags[i] if i < len(pos_tags) else ""
        corrected = corriger_p2g(ortho_words[i], pos=pos, morpho=word_morpho)
        result.append(corrected)
    return result


# ── Fallback ortho-lexique (Phase 3c) ──────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Distance de Levenshtein entre deux chaines."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[lb]


def _ngrams(word: str, n: int = 3) -> set[str]:
    """Extrait les n-grammes d'un mot."""
    if len(word) < n:
        return {word}
    return {word[i:i + n] for i in range(len(word) - n + 1)}


def _build_prefix_index(
    lexique: set[str] | frozenset[str],
    freq_map: dict[str, float] | None = None,
    min_freq: float = 0.0,
) -> dict[str, list[str]]:
    """Construit un index par bigramme pour recherche rapide.

    Si freq_map est fourni, ne garde que les formes avec freq >= min_freq.
    """
    index: dict[str, list[str]] = {}
    for word in lexique:
        if freq_map and freq_map.get(word, 0) < min_freq:
            continue
        if len(word) < 2:
            continue
        # Indexer par tous les bigrammes du mot
        for i in range(len(word) - 1):
            bigram = word[i:i + 2]
            if bigram not in index:
                index[bigram] = []
            index[bigram].append(word)
    return index


# Fusions courantes : mots colles dont la forme correcte n'est pas un
# simple insert de - ou ' (ex: "esse" -> "est-ce", pas "es-se").
_FUSIONS = {
    "esse": "est-ce",
    "estce": "est-ce",
    "estil": "est-il",
    "estelle": "est-elle",
    "eston": "est-on",
    "yatil": "y a-t-il",
    "peuxtu": "peux-tu",
    "peutetre": "peut-être",
    "peutêtre": "peut-être",
    "cela": "cela",  # pas de correction (mot valide)
    "audessus": "au-dessus",
    "audessous": "au-dessous",
    "audela": "au-delà",
    "audelà": "au-delà",
    "cestadire": "c'est-à-dire",
    "visavis": "vis-à-vis",
    "visàvis": "vis-à-vis",
}


def _trouver_separateur(lower: str, lexique: set[str] | frozenset[str]) -> str | None:
    """Trouve la forme composee d'un mot colle.

    1. Dictionnaire de fusions courantes (esse -> est-ce, etc.)
    2. Insertion de - ou ' a chaque position interne.
    Retourne le premier match trouve, ou None.
    """
    # Fusions connues
    fusion = _FUSIONS.get(lower)
    if fusion is not None and fusion != lower:
        return fusion

    # Insertion de separateur
    for i in range(1, len(lower)):
        gauche = lower[:i]
        droite = lower[i:]
        for sep in ("-", "'"):
            compose = gauche + sep + droite
            if compose in lexique:
                return compose
    return None


def corriger_par_lexique_ortho(
    ortho: str,
    pos: str,
    lexique: set[str] | frozenset[str],
    lexique_index: dict[str, list[str]] | None = None,
    freq_map: dict[str, float] | None = None,
    max_edit: int = 1,
) -> str:
    """Corrige un mot absent du lexique en deux etapes.

    Etape 1 : insertion de separateur (- ou ') → match exact dans le lexique.
              Rapide et sans faux positif (celuici -> celui-ci).
              S'applique a NOM/ADJ/VER/AUX.
    Etape 2 : distance d'edition + frequence via l'index bigramme.
              Corrige les fautes d'ortho proches (conquiet -> conquiert).
              Exclut NOM (trop de noms propres hors lexique → faux positifs).
              Limite a max_edit=1 par defaut pour haute precision.

    Ne s'applique que si le mot n'est pas dans le lexique et a au moins 4 chars.
    """
    lower = ortho.lower()
    if lower in lexique:
        return ortho

    if len(lower) < 4:
        return ortho

    if pos not in ("NOM", "ADJ", "VER", "AUX"):
        return ortho

    # ── Etape 1 : insertion de separateur (tous POS de contenu) ──
    sep_match = _trouver_separateur(lower, lexique)
    if sep_match is not None:
        if ortho[0].isupper():
            return sep_match[0].upper() + sep_match[1:]
        return sep_match

    # ── Etape 2 : distance d'edition (ADJ/VER/AUX seulement) ──
    # NOM exclu : les noms propres hors lexique seraient corriges a tort
    # (witsel->rituel, ptolemer->tolerer, etc.)
    if pos == "NOM":
        return ortho

    if lexique_index is None:
        return ortho

    candidate_counts: dict[str, int] = {}
    source_bigrams = set()
    for i in range(len(lower) - 1):
        bigram = lower[i:i + 2]
        source_bigrams.add(bigram)
        for w in lexique_index.get(bigram, []):
            if abs(len(w) - len(lower)) <= max_edit:
                candidate_counts[w] = candidate_counts.get(w, 0) + 1

    if not candidate_counts:
        return ortho

    n_bigrams_src = max(len(source_bigrams), 1)
    min_shared = max(1, n_bigrams_src // 2)

    best_candidate = None
    best_dist = max_edit + 1
    best_freq = -1.0

    for candidate, shared in candidate_counts.items():
        if shared < min_shared:
            continue
        dist = _levenshtein(lower, candidate)
        if dist > max_edit:
            continue
        cand_freq = freq_map.get(candidate, 0.0) if freq_map else 0.0
        if dist < best_dist or (dist == best_dist and cand_freq > best_freq):
            best_dist = dist
            best_candidate = candidate
            best_freq = cand_freq

    if best_candidate is not None:
        if ortho[0].isupper() and best_candidate[0].islower():
            return best_candidate[0].upper() + best_candidate[1:]
        return best_candidate

    return ortho


# ── Pipeline Phase 3+ ────────────────────────────────────────────


def corriger_phrase_v3(
    ortho_words: list[str],
    pos_tags: list[str],
    morpho_features: dict[str, list[str]],
    lexique: set[str] | frozenset[str] | None = None,
    lexique_index: dict[str, list[str]] | None = None,
    freq_map: dict[str, float] | None = None,
    lex_candidates: list[list[tuple[str, float]]] | None = None,
    skip_positions: set[int] | None = None,
) -> list[str]:
    """Pipeline post-traitement core pour le modele P2G v6.

    Etape 1  : forcer_coherence_ortho_morpho() (coherence ortho vs morpho predite)
    Etape 1b : corrections contextuelles inter-mots (det/pro pluriel)
    Etape 1c : homophones par POS (a/a, ou/ou)
    Etape 1d : homophones par morpho (il/ils, au/aux, etc.)

    Les etapes 1b et 1c s'appliquent uniquement quand la morpho predite
    est incoherente avec le contexte (ex: morpho=Sing mais det=pluriel),
    ce qui evite les regressions sur les cas ou morpho est deja correct.

    Parameters
    ----------
    skip_positions : set[int] | None
        Positions a ne pas modifier (ex: formules detectees par le pipeline
        couche 2 lectura-p2g). Ces positions sont protegees de toutes les
        corrections.

    Note : les formules (nombres, sigles) et les noms propres sont geres
    par lectura-p2g (pipeline couche 2).
    """
    result = list(ortho_words)
    n = len(result)

    _in_lex = (lambda w: w.lower() in lexique) if lexique is not None else (lambda w: True)

    formule_positions = skip_positions or set()

    # Etape 1 : coherence ortho-morpho (regles Phase 3c)
    # Ne s'applique qu'aux mots HORS lexique : si lex_select a produit un
    # mot existant, la morpho predite (souvent fausse) ne doit pas le casser.
    for i in range(n):
        if i in formule_positions:
            continue
        if _in_lex(result[i]):
            continue
        pos = pos_tags[i] if i < len(pos_tags) else ""
        word_morpho = {
            feat: vals[i] if i < len(vals) else "_"
            for feat, vals in morpho_features.items()
        }
        result[i] = forcer_coherence_ortho_morpho(
            result[i], pos, word_morpho, lexique=lexique,
        )

    # Etape 1b : corrections contextuelles inter-mots
    # Appliquees uniquement quand la morpho predite contredit le contexte
    for i in range(n):
        if i in formule_positions:
            continue
        pos = pos_tags[i] if i < len(pos_tags) else ""
        curr = result[i]
        morpho_number = "_"
        if "Number" in morpho_features and i < len(morpho_features["Number"]):
            morpho_number = morpho_features["Number"][i]

        # ── Regle 1b-a : Det pluriel + NOM/ADJ sans -s → +s ──
        # Seulement si la morpho ne predit PAS deja Plur (sinon la regle 1
        # de forcer_coherence l'aurait deja fait)
        if (
            i > 0
            and pos in ("NOM", "ADJ")
            and morpho_number != "Plur"
            and result[i - 1].lower() in _PLUR_DET
            and not curr.endswith(("s", "x", "z"))
            and len(curr) > 1
            and curr.lower() not in _NO_PLURAL_S
        ):
            candidate = curr + "s"
            if _in_lex(candidate):
                result[i] = candidate

        # ── Regle 1b-b : Det singulier + NOM/ADJ avec -s spurieux → -s ──
        # Seulement si la morpho ne predit PAS deja Sing
        if (
            i > 0
            and pos in ("NOM", "ADJ")
            and morpho_number != "Sing"
            and result[i - 1].lower() in _SING_DET
            and curr.endswith("s")
            and len(curr) > 2
            and not curr.lower().endswith(_NO_STRIP_S_ENDINGS)
        ):
            candidate = curr[:-1]
            if candidate and _in_lex(candidate):
                result[i] = candidate

        # ── Regle 1b-c : ils/elles + VER sans -ent → +nt ──
        # Seulement si la morpho ne predit PAS deja Plur
        if (
            i > 0
            and pos in ("VER", "AUX")
            and morpho_number != "Plur"
            and result[i - 1].lower() in ("ils", "elles")
            and curr.endswith("e")
            and not curr.endswith(("ent", "nt"))
        ):
            candidate = curr + "nt"
            if _in_lex(candidate):
                result[i] = candidate

    # Etape 1c : homophones par POS
    # Utilise directement la prediction POS du modele pour choisir la bonne forme.
    for i in range(n):
        if i in formule_positions:
            continue
        pos = pos_tags[i] if i < len(pos_tags) else ""
        lower = result[i].lower()

        # a/à : PRE → à, AUX/VER → a
        if lower == "a" and pos == "PRE":
            result[i] = "à"
        elif lower == "à" and pos in ("AUX", "VER"):
            result[i] = "a"

        # ou/où : PRO:rel/ADV → où, CON → ou
        if lower == "ou" and pos in ("PRO:rel", "ADV"):
            result[i] = "où"
        elif lower == "où" and pos == "CON":
            result[i] = "ou"

        # Note: ces/ses, ça/sa, son/sont, on/ont testes via benchmark_pos_morpho_optim
        # mais net negatif ou neutre meme avec seuils de confiance POS.
        # leur/leurs (POS+morpho) : net +1 mais trop marginal.

    # Etape 1d : homophones par morpho
    # Utilise la prediction morpho (Number) pour les paires dont la distinction
    # est le nombre (il/ils, au/aux, leur/leurs, etc.).
    from lectura_graphemiseur._homophones import _HOMOPHONES_MORPHO

    for i in range(n):
        if i in formule_positions:
            continue
        lower = result[i].lower()
        if lower not in _HOMOPHONES_MORPHO:
            continue
        feature, mapping = _HOMOPHONES_MORPHO[lower]
        feat_vals = morpho_features.get(feature, [])
        feat_val = feat_vals[i] if i < len(feat_vals) else "_"
        if feat_val == "_":
            continue
        correct = mapping.get(feat_val)
        if correct is None or correct == lower:
            continue
        # Preserver la capitalisation
        if result[i][0].isupper() and correct[0].islower():
            result[i] = correct[0].upper() + correct[1:]
        else:
            result[i] = correct

    # Note: les etapes lex_cand et fallback ortho-lexique ont ete retirees.
    # Benchmark montrait des regressions nettes (-286 et -1988 respectivement) :
    # le modele V6 avec lex_select produit des sorties suffisamment bonnes
    # pour que ces corrections post-hoc soient contre-productives.

    return result
