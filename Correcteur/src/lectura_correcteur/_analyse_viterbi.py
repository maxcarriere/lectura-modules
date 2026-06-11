"""Decodeur Viterbi trigramme POS avec expansion de candidats.

Resout simultanement la desambiguation POS et la correction accent/homophone
en cherchant la meilleure sequence POS puis en recuperant les meilleures
formes associees.

Optimisation cle : l'espace d'etats est reduit aux ~18 tags POS (pas les
paires forme×POS). Pour chaque position et chaque POS, on pre-selectionne
la meilleure forme (argmax emission) puis le Viterbi opere sur les POS
uniquement. Complexite : O(N × T^3) avec T~18.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS


# ---------------------------------------------------------------------------
# Structures de donnees
# ---------------------------------------------------------------------------

@dataclass
class CandidatForme:
    """Un candidat forme+POS pour une position donnee."""
    forme: str       # ex: "prepare"
    pos: str         # ex: "VER"
    freq: float      # frequence lexique agregee
    emission: float  # log-score calcule
    source: str      # "original" | "accent" | "homophone"


@dataclass
class ResultatViterbi:
    """Resultat du Viterbi pour une position."""
    forme: str       # Forme choisie
    pos: str         # POS choisi
    score: float     # Score Viterbi
    confiance: float # Softmax sur les POS alternatifs
    source: str      # D'ou vient la forme
    changed: bool    # True si forme != original


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_BOS = "<BOS>"
_EPS = 1e-6  # lissage pour log(freq)
_DEFAULT_LOGP = -15.0  # coherent avec _pos_ngram.py
_BACKOFF_TRIGRAM_THRESHOLD = -14.0  # seuil pour fallback bigram

# Homophones grammaticaux : ne pas les proposer comme candidats homophones
# car ils sont geres par les regles de grammaire (plus precises).
_GRAMMAR_HOMOPHONES = frozenset({
    "et", "est",
    "son", "sont",
    "a", "\xe0",       # a / à
    "ou", "o\xf9",     # ou / où
    "on", "ont",
    "ce", "se",
    "la", "l\xe0",     # la / là
    "leur", "leurs",
    "\xe7a", "sa",     # ça / sa
    "ces", "ses",
    "peu", "peut", "peux",
    "ma", "m'a",
    "ta", "t'a",
    "dans", "d'en",
    "sans", "s'en",
    "si", "s'y",
    "ni", "n'y",
    "mais", "mes",
    "d\xe8s", "des",   # dès / des
})

# Mots-outils courts : pas de substitution homophone
_MOTS_OUTILS = frozenset({
    "il", "ils", "elle", "elles", "on", "nous", "vous",
    "le", "la", "les", "de", "des", "du", "un", "une",
    "je", "tu", "me", "te", "ne", "se", "ce",
    "en", "y", "au", "aux", "que", "qui", "dont",
})


# ---------------------------------------------------------------------------
# Expansion des candidats
# ---------------------------------------------------------------------------

def _expand_candidates(
    mot: str,
    idx: int,
    lexique,
    lm_homophones,
    mots: list[str],
    seuil_freq_expansion: float = 5.0,
) -> list[CandidatForme]:
    """Genere tous les candidats (forme, POS) pour la position idx.

    Sources :
    - Original : POS du lexique pour le mot tel quel (toujours)
    - Variantes accent : seulement si le mot est OOV ou basse frequence
    - Homophones : seulement si le mot est OOV ou basse frequence
    - OOV fallback : si aucun candidat, POS=NOM freq=0

    Logique conservative : les mots in-lexique courants ne sont pas
    expanses pour eviter les faux positifs (cour->courte, vert->vers).
    Les mots-outils et homophones grammaticaux ne sont jamais expanses.
    """
    candidates: list[CandidatForme] = []
    seen: set[tuple[str, str]] = set()  # (forme_low, pos)
    low = mot.lower()

    # --- Source 1 : mot original (toujours) ---
    _add_from_lexique(low, lexique, "original", candidates, seen)

    # Determiner si on doit expanser : seulement si OOV ou basse frequence
    is_oov = not lexique.existe(low) if hasattr(lexique, "existe") else not candidates
    freq_orig = (
        lexique.frequence(low) if hasattr(lexique, "frequence") else 0.0
    )
    should_expand = (
        is_oov
        or freq_orig < seuil_freq_expansion
    ) and low not in _MOTS_OUTILS and low not in _GRAMMAR_HOMOPHONES

    # --- Source 2 : variantes accent ---
    if should_expand:
        from lectura_correcteur.orthographe._suggestions import _variantes_accents
        for forme_acc, _freq in _variantes_accents(low, lexique):
            if forme_acc in _GRAMMAR_HOMOPHONES or forme_acc in _MOTS_OUTILS:
                continue
            _add_from_lexique(forme_acc, lexique, "accent", candidates, seen)

    # --- Source 3 : homophones ---
    if should_expand and lm_homophones is not None:
        homo_cands = lm_homophones.candidats(low)
        if homo_cands:
            same_lemma = getattr(lm_homophones, "_same_lemma_pairs", set())
            for ortho, _freq in homo_cands:
                ortho_low = ortho.lower()
                if ortho_low == low:
                    continue
                if (low, ortho_low) in same_lemma:
                    continue
                if ortho_low in _GRAMMAR_HOMOPHONES:
                    continue
                if ortho_low in _MOTS_OUTILS:
                    continue
                _add_from_lexique(
                    ortho_low, lexique, "homophone", candidates, seen,
                )

    # --- OOV fallback ---
    if not candidates:
        candidates.append(CandidatForme(
            forme=low, pos="NOM", freq=0.0, emission=0.0, source="original",
        ))

    return candidates


def _add_from_lexique(
    forme: str,
    lexique,
    source: str,
    candidates: list[CandidatForme],
    seen: set[tuple[str, str]],
) -> None:
    """Ajoute les entrees du lexique pour *forme* aux candidats."""
    infos = lexique.info(forme) if hasattr(lexique, "info") else []

    # Agreger les frequences par POS
    pos_freq: dict[str, float] = {}
    for entry in infos:
        cgram = entry.get("cgram")
        if cgram:
            freq = float(entry.get("freq") or 0)
            pos_freq[cgram] = pos_freq.get(cgram, 0) + max(freq, 0.01)

    # Override mots-outils
    override = _FUNCTION_WORD_POS.get(forme)
    if override:
        if override not in pos_freq:
            pos_freq[override] = 100.0
        else:
            max_f = max(pos_freq.values()) if pos_freq else 1.0
            pos_freq[override] = max(pos_freq[override], max_f * 5)

    for pos, freq in pos_freq.items():
        key = (forme, pos)
        if key not in seen:
            seen.add(key)
            candidates.append(CandidatForme(
                forme=forme, pos=pos, freq=freq,
                emission=0.0, source=source,
            ))


# ---------------------------------------------------------------------------
# Modele d'emission
# ---------------------------------------------------------------------------

def _compute_emission(
    candidat: CandidatForme,
    mot_original: str,
    mots: list[str],
    idx: int,
    lm_homophones,
    bonus_original: float,
    bonus_lm: float,
) -> float:
    """Calcule le log-score d'emission pour un candidat.

    emission = log(freq + eps)
             + bonus_original   si forme == mot_original
             + bonus_lm         si homophone et LM score > 0
    """
    score = math.log(candidat.freq + _EPS)

    # Biais conservateur : garder la forme d'entree
    if candidat.forme == mot_original.lower():
        score += bonus_original

    # Bonus LM homophones (contexte original)
    if candidat.source == "homophone" and lm_homophones is not None and bonus_lm > 0:
        ctx_g = mots[idx - 1] if idx > 0 else None
        ctx_d = mots[idx + 1] if idx + 1 < len(mots) else None
        lm_score = lm_homophones.scorer(candidat.forme, ctx_g, ctx_d)
        if lm_score > 0:
            score += bonus_lm * math.log(1 + lm_score)

    return score


# ---------------------------------------------------------------------------
# Pre-selection : meilleure forme par POS
# ---------------------------------------------------------------------------

def _best_form_per_pos(
    candidates: list[CandidatForme],
) -> dict[str, CandidatForme]:
    """Pour chaque POS, garde le candidat avec la meilleure emission."""
    best: dict[str, CandidatForme] = {}
    for c in candidates:
        if c.pos not in best or c.emission > best[c.pos].emission:
            best[c.pos] = c
    return best


# ---------------------------------------------------------------------------
# Modele de transition (avec backoff)
# ---------------------------------------------------------------------------

def _transition_logp(pos_ngram, p1: str, p2: str, p3: str) -> float:
    """Log-probabilite de transition P(p3 | p1, p2) avec backoff.

    Convention identique a PosNgram.score_sequence() :
    - Si le trigramme est trouve (logp > -14.0), on l'utilise.
    - Sinon, backoff vers le bigramme avec une penalite de -1.0.
    """
    tri = pos_ngram.logp_trigram(p1, p2, p3)
    if tri > _BACKOFF_TRIGRAM_THRESHOLD:
        return tri
    bi = pos_ngram.logp_bigram(p2, p3)
    return bi - 1.0


# ---------------------------------------------------------------------------
# Algorithme Viterbi trigramme
# ---------------------------------------------------------------------------

def _viterbi_trigram(
    emissions_per_pos: list[dict[str, float]],
    best_forms: list[dict[str, CandidatForme]],
    pos_ngram,
    w_emission: float,
    w_transition: float,
) -> list[ResultatViterbi]:
    """Decodeur Viterbi trigramme sur les POS.

    Etat = (prev_pos, curr_pos).
    A chaque position t >= 1 :
      Pour chaque (pp, p) dans viterbi[t-1] :
        Pour chaque c dans states[t] :
          score = viterbi[t-1][(pp, p)]
                + w_transition * transition(pp, p, c)
                + w_emission * emission[t][c]
          Mettre a jour viterbi[t][(p, c)] si meilleur

    Initialisation t=0 : transitions depuis (BOS, BOS).
    """
    n = len(emissions_per_pos)
    if n == 0:
        return []

    # Liste des POS disponibles a chaque position
    states: list[list[str]] = [list(em.keys()) for em in emissions_per_pos]

    # --- Initialisation t=0 ---
    # viterbi[t] : dict[(prev_pos, curr_pos)] -> score
    # backptr[t] : dict[(prev_pos, curr_pos)] -> prev_state (pp, p) au temps t-1
    viterbi_scores: list[dict[tuple[str, str], float]] = []
    backptrs: list[dict[tuple[str, str], tuple[str, str]]] = []

    v0: dict[tuple[str, str], float] = {}
    for c in states[0]:
        trans = _transition_logp(pos_ngram, _BOS, _BOS, c)
        score = w_transition * trans + w_emission * emissions_per_pos[0][c]
        state = (_BOS, c)
        if state not in v0 or score > v0[state]:
            v0[state] = score
    viterbi_scores.append(v0)
    backptrs.append({})  # pas de backpointer au temps 0

    # --- Recurrence t=1..n-1 ---
    for t in range(1, n):
        vt: dict[tuple[str, str], float] = {}
        bt: dict[tuple[str, str], tuple[str, str]] = {}

        for (pp, p), prev_score in viterbi_scores[t - 1].items():
            for c in states[t]:
                trans = _transition_logp(pos_ngram, pp, p, c)
                score = (
                    prev_score
                    + w_transition * trans
                    + w_emission * emissions_per_pos[t][c]
                )
                new_state = (p, c)
                if new_state not in vt or score > vt[new_state]:
                    vt[new_state] = score
                    bt[new_state] = (pp, p)

        viterbi_scores.append(vt)
        backptrs.append(bt)

    # --- Trouver le meilleur etat final ---
    if not viterbi_scores[-1]:
        # Degenere : aucun etat valide
        return [
            ResultatViterbi(
                forme=list(bf.values())[0].forme if bf else "",
                pos=list(bf.keys())[0] if bf else "NOM",
                score=0.0, confiance=0.0,
                source=list(bf.values())[0].source if bf else "original",
                changed=False,
            )
            for bf in best_forms
        ]

    best_final_state = max(viterbi_scores[-1], key=viterbi_scores[-1].get)
    best_final_score = viterbi_scores[-1][best_final_state]

    # --- Backtracking ---
    path_states: list[tuple[str, str]] = [best_final_state]
    for t in range(n - 1, 0, -1):
        prev_state = backptrs[t].get(path_states[-1])
        if prev_state is None:
            # Fallback : prendre n'importe quel etat precedent
            if viterbi_scores[t - 1]:
                prev_state = max(
                    viterbi_scores[t - 1],
                    key=viterbi_scores[t - 1].get,
                )
            else:
                prev_state = (_BOS, _BOS)
        path_states.append(prev_state)
    path_states.reverse()

    # Extraire la sequence POS (le 2e element de chaque paire)
    pos_sequence = [s[1] for s in path_states]

    # --- Construire les resultats avec confiance ---
    results: list[ResultatViterbi] = []
    for t in range(n):
        chosen_pos = pos_sequence[t]
        bf = best_forms[t]

        # Confiance : softmax sur les scores des POS alternatifs
        confiance = _compute_confidence(
            chosen_pos, emissions_per_pos[t], viterbi_scores, t, path_states,
        )

        if chosen_pos in bf:
            cand = bf[chosen_pos]
            results.append(ResultatViterbi(
                forme=cand.forme,
                pos=chosen_pos,
                score=best_final_score,
                confiance=confiance,
                source=cand.source,
                changed=(cand.source != "original"),
            ))
        else:
            # POS pas dans best_forms (ne devrait pas arriver)
            # Fallback : prendre le candidat avec la meilleure emission
            fallback = max(bf.values(), key=lambda c: c.emission) if bf else None
            results.append(ResultatViterbi(
                forme=fallback.forme if fallback else "",
                pos=chosen_pos,
                score=best_final_score,
                confiance=confiance,
                source=fallback.source if fallback else "original",
                changed=False,
            ))

    return results


def _compute_confidence(
    chosen_pos: str,
    emission_scores: dict[str, float],
    viterbi_scores: list[dict[tuple[str, str], float]],
    t: int,
    path_states: list[tuple[str, str]],
) -> float:
    """Confiance du POS choisi via softmax sur les emissions a la position t."""
    if len(emission_scores) <= 1:
        return 1.0

    values = list(emission_scores.values())
    max_val = max(values)
    # Softmax numeriquement stable
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    if total == 0:
        return 1.0 / len(values)

    chosen_val = emission_scores.get(chosen_pos, max_val)
    chosen_exp = math.exp(chosen_val - max_val)
    return chosen_exp / total


# ---------------------------------------------------------------------------
# Point d'entree principal
# ---------------------------------------------------------------------------

def analyse_viterbi(
    mots: list[str],
    lexique,
    pos_ngram,
    *,
    lm_homophones=None,
    bonus_original: float = 2.0,
    bonus_lm: float = 1.0,
    w_emission: float = 1.0,
    w_transition: float = 1.0,
) -> list[ResultatViterbi]:
    """Analyse Viterbi trigramme : desambiguation POS + correction forme.

    Pour chaque mot, expanse les candidats (original, accents, homophones),
    calcule les emissions, selectionne la meilleure forme par POS, puis
    execute le Viterbi trigramme sur les POS pour trouver la sequence
    optimale. Retourne la forme associee a chaque POS choisi.

    Args:
        mots: liste de tokens mots (minuscules non requis)
        lexique: objet LexiqueProtocol
        pos_ngram: PosNgram (trigrammes/bigrammes POS)
        lm_homophones: LMHomophones optionnel
        bonus_original: biais conservateur pour garder la forme d'entree
        bonus_lm: poids du bonus LM homophones
        w_emission: poids des emissions dans le Viterbi
        w_transition: poids des transitions POS n-gram

    Returns:
        list[ResultatViterbi] de meme longueur que mots
    """
    n = len(mots)
    if n == 0:
        return []

    # 1. Expansion des candidats pour chaque position
    all_candidates: list[list[CandidatForme]] = []
    for i, mot in enumerate(mots):
        cands = _expand_candidates(mot, i, lexique, lm_homophones, mots)
        all_candidates.append(cands)

    # 2. Calculer les emissions
    for i, cands in enumerate(all_candidates):
        for c in cands:
            c.emission = _compute_emission(
                c, mots[i], mots, i,
                lm_homophones, bonus_original, bonus_lm,
            )

    # 3. Pre-selection : meilleure forme par POS
    best_forms: list[dict[str, CandidatForme]] = [
        _best_form_per_pos(cands) for cands in all_candidates
    ]

    # 4. Construire les emissions par POS (pour le Viterbi)
    emissions_per_pos: list[dict[str, float]] = []
    for bf in best_forms:
        emissions_per_pos.append({pos: c.emission for pos, c in bf.items()})

    # 5. Viterbi trigramme
    results = _viterbi_trigram(
        emissions_per_pos, best_forms, pos_ngram,
        w_emission, w_transition,
    )

    return results
