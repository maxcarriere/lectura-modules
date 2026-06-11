"""Viterbi POS+Morpho en aval de la grammaire.

Apres les regles de grammaire (accords, conjugaison, homophones), ce module
execute un Viterbi trigramme sur les tags POS+Morpho (ex: "NOM|Sing|Fem")
pour valider et corriger les traits morphologiques (genre, nombre).

Cas d'usage :
- "les petites enfants" → "les petits enfants" (genre corrige via sequence PM)
- "élevés" vs "élèves" : PM tag NOM|Plur|Masc vs VER|_|Masc -> le trigram tranche
- Validation des accords sujet-verbe, determinant-nom, etc.

Architecture :
1. Pour chaque mot corrige, extraire tous les PM tags possibles du lexique
2. Emission = log(freq) + bonus si le PM tag correspond au POS/morpho actuel
3. Transition = pm_trigram avec backoff pm_bigram
4. Viterbi trigramme sur les PM tags (~92 tags possibles, mais ~2-5 par position)
5. Si le meilleur PM tag implique une forme differente, corriger

L'espace d'etats est reduit au PM tag, pas aux paires (forme, PM).
Pour chaque PM tag a chaque position, on garde la meilleure forme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class CandidatMorpho:
    """Candidat forme + PM tag pour une position."""
    forme: str       # ex: "élèves"
    pm_tag: str      # ex: "NOM|Plur|Masc"
    pos: str         # ex: "NOM"
    genre: str       # "Masc", "Fem", "_"
    nombre: str      # "Sing", "Plur", "_"
    freq: float      # frequence lexique
    emission: float  # log-score


@dataclass
class ResultatMorpho:
    """Resultat du Viterbi Morpho pour une position."""
    forme: str
    pos: str
    pm_tag: str
    genre: str
    nombre: str
    changed: bool    # True si forme modifiee


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_BOS = "<BOS>"
_EPS = 1e-6
_DEFAULT_LOGP = -15.0
_BACKOFF_PM_THRESHOLD = -14.0

# Mapping lexique short codes -> PM tag long form
_GENRE_MAP = {"m": "Masc", "f": "Fem", "": "_", "_": "_"}
_NOMBRE_MAP = {"s": "Sing", "p": "Plur", "": "_", "_": "_"}


# ---------------------------------------------------------------------------
# Construction des candidats PM par position
# ---------------------------------------------------------------------------

def _build_pm_candidates(
    forme: str,
    pos_actuel: str,
    lexique,
    bonus_current: float,
) -> list[CandidatMorpho]:
    """Construit les candidats PM pour un mot a partir du lexique.

    Pour chaque entree du lexique correspondant a la forme, on genere
    un CandidatMorpho avec le PM tag correspondant.
    """
    low = forme.lower()
    infos = lexique.info(low) if hasattr(lexique, "info") else []

    candidates: list[CandidatMorpho] = []
    seen_pm: set[str] = set()

    for entry in infos:
        cgram = entry.get("cgram", "")
        if not cgram:
            continue
        genre_raw = entry.get("genre", "") or ""
        nombre_raw = entry.get("nombre", "") or ""
        freq = float(entry.get("freq") or 0)

        genre = _GENRE_MAP.get(genre_raw, "_")
        nombre = _NOMBRE_MAP.get(nombre_raw, "_")
        personne_raw = entry.get("personne", "") or ""
        personne = {"1": "1", "2": "2", "3": "3"}.get(personne_raw, "_")
        pm_tag = f"{cgram}|{nombre}|{genre}|{personne}"

        if pm_tag in seen_pm:
            # Agreger la frequence
            for c in candidates:
                if c.pm_tag == pm_tag:
                    c.freq += max(freq, 0.01)
                    break
            continue

        seen_pm.add(pm_tag)

        # Emission = log(freq) + bonus si POS correspond a l'actuel
        em = math.log(max(freq, 0.01) + _EPS)
        if cgram == pos_actuel:
            em += bonus_current

        candidates.append(CandidatMorpho(
            forme=low,
            pm_tag=pm_tag,
            pos=cgram,
            genre=genre,
            nombre=nombre,
            freq=max(freq, 0.01),
            emission=em,
        ))

    # Recalculer les emissions apres agregation
    for c in candidates:
        em = math.log(c.freq + _EPS)
        if c.pos == pos_actuel:
            em += bonus_current
        c.emission = em

    if not candidates:
        # OOV : un seul candidat generique
        pm_tag = f"{pos_actuel or 'NOM'}|_|_|_"
        candidates.append(CandidatMorpho(
            forme=low, pm_tag=pm_tag, pos=pos_actuel or "NOM",
            genre="_", nombre="_", freq=0.01, emission=0.0,
        ))

    return candidates


def _edit_distance_simple(a: str, b: str) -> int:
    """Distance d'edition Levenshtein simple."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 3:
        return abs(la - lb)
    prev = list(range(lb + 1))
    for i in range(la):
        curr = [i + 1] + [0] * lb
        for j in range(lb):
            cost = 0 if a[i] == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[lb]


def _build_morpho_variants(
    forme: str,
    pos_actuel: str,
    lexique,
    bonus_current: float,
) -> list[CandidatMorpho]:
    """Construit les candidats PM incluant les variantes morphologiques.

    Restrictions pour eviter les faux positifs :
    - Seules les variantes du meme POS sont acceptees
    - Distance d'edition <= 3 (flexions genre/nombre : -é/-ée/-és/-ées)
    - Le lemme doit correspondre a une entree du meme POS que pos_actuel
    """
    candidates = _build_pm_candidates(forme, pos_actuel, lexique, bonus_current)

    low = forme.lower()
    seen_pm: set[str] = {c.pm_tag for c in candidates}
    seen_formes: set[str] = {low}

    # Trouver le(s) lemme(s) du mot pour le POS actuel uniquement
    infos = lexique.info(low) if hasattr(lexique, "info") else []
    lemmes: set[str] = set()
    # Collecter les POS du mot actuel (pour ne garder que les variantes du meme POS)
    valid_pos: set[str] = set()
    for entry in infos:
        cgram = entry.get("cgram", "")
        lemme = entry.get("lemme", "")
        if lemme and cgram:
            # Restreindre aux lemmes du POS actuel (ou du meme POS de base)
            pos_base = cgram.split(":")[0]
            pos_actuel_base = pos_actuel.split(":")[0] if pos_actuel else ""
            if pos_base == pos_actuel_base or not pos_actuel:
                lemmes.add(lemme.lower())
                valid_pos.add(cgram)

    if not lemmes or not hasattr(lexique, "formes_de"):
        return candidates

    for lemme in lemmes:
        variantes = lexique.formes_de(lemme)
        if not variantes:
            continue
        for var_entry in variantes:
            var_forme = var_entry.get("ortho", "").lower()
            if not var_forme or var_forme in seen_formes:
                continue

            cgram = var_entry.get("cgram", "")
            if not cgram:
                continue

            # Restriction POS : meme categorie de base
            if cgram.split(":")[0] != pos_actuel.split(":")[0]:
                continue

            # Restriction distance : flexion genre/nombre typique <= 2
            if _edit_distance_simple(low, var_forme) > 2:
                continue

            genre_raw = var_entry.get("genre", "") or ""
            nombre_raw = var_entry.get("nombre", "") or ""
            freq = float(var_entry.get("freq") or 0)

            # Restriction flexion : exclure les conjugaisons (mode/temps different)
            # On n'accepte que les variantes qui ont un genre non-vide
            # (= participe passe flexionnel, adjectif, nom) pour les verbes.
            orig_has_genre = any(
                e.get("genre") for e in infos
                if (e.get("cgram", "").split(":")[0] == cgram.split(":")[0])
            )
            if orig_has_genre and not genre_raw:
                continue

            genre = _GENRE_MAP.get(genre_raw, "_")
            nombre = _NOMBRE_MAP.get(nombre_raw, "_")
            var_personne_raw = var_entry.get("personne", "") or ""
            var_personne = {"1": "1", "2": "2", "3": "3"}.get(var_personne_raw, "_")
            pm_tag = f"{cgram}|{nombre}|{genre}|{var_personne}"

            if pm_tag in seen_pm:
                continue
            seen_pm.add(pm_tag)

            em = math.log(max(freq, 0.01) + _EPS)

            candidates.append(CandidatMorpho(
                forme=var_forme,
                pm_tag=pm_tag,
                pos=cgram,
                genre=genre,
                nombre=nombre,
                freq=max(freq, 0.01),
                emission=em,
            ))
            seen_formes.add(var_forme)

    return candidates


# ---------------------------------------------------------------------------
# Meilleure forme par PM tag
# ---------------------------------------------------------------------------

def _best_form_per_pm(
    candidates: list[CandidatMorpho],
) -> dict[str, CandidatMorpho]:
    """Pour chaque PM tag, garde le candidat avec la meilleure emission."""
    best: dict[str, CandidatMorpho] = {}
    for c in candidates:
        if c.pm_tag not in best or c.emission > best[c.pm_tag].emission:
            best[c.pm_tag] = c
    return best


# ---------------------------------------------------------------------------
# Transition PM avec backoff
# ---------------------------------------------------------------------------

def _transition_pm_logp(pos_ngram, pm1: str, pm2: str, pm3: str) -> float:
    """Log-probabilite de transition PM avec backoff bigram."""
    tri = pos_ngram.logp_pm_trigram(pm1, pm2, pm3)
    if tri > _BACKOFF_PM_THRESHOLD:
        return tri
    bi = pos_ngram.logp_pm_bigram(pm2, pm3)
    return bi - 1.0


# ---------------------------------------------------------------------------
# Viterbi trigramme PM
# ---------------------------------------------------------------------------

def _viterbi_pm_trigram(
    emissions_per_pm: list[dict[str, float]],
    best_forms: list[dict[str, CandidatMorpho]],
    pos_ngram,
    w_emission: float,
    w_transition: float,
) -> list[ResultatMorpho]:
    """Decodeur Viterbi trigramme sur les PM tags.

    Identique en structure au Viterbi POS de _analyse_viterbi.py
    mais avec des PM tags (~92 tags totaux, ~2-5 par position).
    """
    n = len(emissions_per_pm)
    if n == 0:
        return []

    states: list[list[str]] = [list(em.keys()) for em in emissions_per_pm]

    # --- Initialisation t=0 ---
    viterbi_scores: list[dict[tuple[str, str], float]] = []
    backptrs: list[dict[tuple[str, str], tuple[str, str]]] = []

    v0: dict[tuple[str, str], float] = {}
    for c in states[0]:
        trans = _transition_pm_logp(pos_ngram, _BOS, _BOS, c)
        score = w_transition * trans + w_emission * emissions_per_pm[0][c]
        state = (_BOS, c)
        if state not in v0 or score > v0[state]:
            v0[state] = score
    viterbi_scores.append(v0)
    backptrs.append({})

    # --- Recurrence ---
    for t in range(1, n):
        vt: dict[tuple[str, str], float] = {}
        bt: dict[tuple[str, str], tuple[str, str]] = {}

        for (pp, p), prev_score in viterbi_scores[t - 1].items():
            for c in states[t]:
                trans = _transition_pm_logp(pos_ngram, pp, p, c)
                score = (
                    prev_score
                    + w_transition * trans
                    + w_emission * emissions_per_pm[t][c]
                )
                new_state = (p, c)
                if new_state not in vt or score > vt[new_state]:
                    vt[new_state] = score
                    bt[new_state] = (pp, p)

        viterbi_scores.append(vt)
        backptrs.append(bt)

    # --- Meilleur etat final ---
    if not viterbi_scores[-1]:
        return [
            ResultatMorpho(
                forme=list(bf.values())[0].forme if bf else "",
                pos=list(bf.values())[0].pos if bf else "NOM",
                pm_tag=list(bf.keys())[0] if bf else "NOM|_|_",
                genre="_", nombre="_", changed=False,
            )
            for bf in best_forms
        ]

    best_final_state = max(viterbi_scores[-1], key=viterbi_scores[-1].get)

    # --- Backtracking ---
    path_states: list[tuple[str, str]] = [best_final_state]
    for t in range(n - 1, 0, -1):
        prev_state = backptrs[t].get(path_states[-1])
        if prev_state is None:
            if viterbi_scores[t - 1]:
                prev_state = max(
                    viterbi_scores[t - 1],
                    key=viterbi_scores[t - 1].get,
                )
            else:
                prev_state = (_BOS, _BOS)
        path_states.append(prev_state)
    path_states.reverse()

    pm_sequence = [s[1] for s in path_states]

    # --- Construire les resultats ---
    results: list[ResultatMorpho] = []
    for t in range(n):
        chosen_pm = pm_sequence[t]
        bf = best_forms[t]

        if chosen_pm in bf:
            cand = bf[chosen_pm]
            results.append(ResultatMorpho(
                forme=cand.forme,
                pos=cand.pos,
                pm_tag=chosen_pm,
                genre=cand.genre,
                nombre=cand.nombre,
                changed=False,  # sera mis a jour par l'appelant
            ))
        else:
            fallback = max(bf.values(), key=lambda c: c.emission) if bf else None
            results.append(ResultatMorpho(
                forme=fallback.forme if fallback else "",
                pos=fallback.pos if fallback else "NOM",
                pm_tag=chosen_pm,
                genre=fallback.genre if fallback else "_",
                nombre=fallback.nombre if fallback else "_",
                changed=False,
            ))

    return results


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def viterbi_morpho(
    mots: list[str],
    pos_list: list[str],
    lexique,
    pos_ngram,
    *,
    bonus_current: float = 2.0,
    w_emission: float = 1.0,
    w_transition: float = 1.0,
    use_variants: bool = False,
    protected: list[bool] | None = None,
) -> list[ResultatMorpho]:
    """Viterbi POS+Morpho sur les mots corriges.

    Args:
        mots: formes corrigees (apres grammaire)
        pos_list: POS assignes a chaque position
        lexique: lexique avec info()
        pos_ngram: PosNgram avec logp_pm_trigram/bigram
        bonus_current: bonus pour le PM tag correspondant au POS actuel
        w_emission: poids des emissions
        w_transition: poids des transitions PM
        use_variants: si True, inclut les variantes flexionnelles
        protected: masque bool par position — True = ne pas expanser
            (mot deja corrige par les regles de grammaire)

    Returns:
        list[ResultatMorpho] de meme longueur que mots
    """
    n = len(mots)
    if n == 0:
        return []

    # 1. Construire les candidats PM pour chaque position
    all_candidates: list[list[CandidatMorpho]] = []
    for i in range(n):
        pos = pos_list[i] if i < len(pos_list) else ""
        is_protected = protected[i] if protected and i < len(protected) else False
        if use_variants and not is_protected:
            cands = _build_morpho_variants(mots[i], pos, lexique, bonus_current)
        else:
            cands = _build_pm_candidates(mots[i], pos, lexique, bonus_current)
        all_candidates.append(cands)

    # 2. Meilleure forme par PM tag
    best_forms: list[dict[str, CandidatMorpho]] = [
        _best_form_per_pm(cands) for cands in all_candidates
    ]

    # 3. Emissions par PM tag
    emissions_per_pm: list[dict[str, float]] = []
    for bf in best_forms:
        emissions_per_pm.append({pm: c.emission for pm, c in bf.items()})

    # 4. Viterbi
    results = _viterbi_pm_trigram(
        emissions_per_pm, best_forms, pos_ngram,
        w_emission, w_transition,
    )

    # 5. Marquer les changements
    for i, r in enumerate(results):
        r.changed = (r.forme != mots[i].lower())

    return results
