"""Decodeur Viterbi trigramme generique.

Factorise l'algorithme Viterbi commun aux passes 2 (POS) et 3 (PM).
L'appelant fournit les etats, les emissions et la fonction de transition ;
ce module s'occupe du decodage forward + backtracking.
"""

from __future__ import annotations

import math
from typing import Callable

_BOS = "<BOS>"


def viterbi_trigram(
    states: list[list[str]],
    emissions: list[dict[str, float]],
    transition_fn: Callable[[str, str, str], float],
    w_emission: float = 1.0,
    w_transition: float = 1.0,
) -> list[tuple[str, float]]:
    """Decodeur Viterbi trigramme generique.

    Args:
        states: candidats par position (states[t] = list de tags possibles)
        emissions: emissions[t][tag] = log-score d'emission
        transition_fn: (prev_prev, prev, curr) -> log-probabilite
        w_emission: poids des emissions
        w_transition: poids des transitions

    Returns:
        list[(best_tag, confiance)] de meme longueur que states
    """
    n = len(states)
    if n == 0:
        return []

    # viterbi[t] : dict[(prev_tag, curr_tag)] -> score
    # backptr[t] : dict[(prev_tag, curr_tag)] -> (prev_prev, prev) au t-1
    viterbi_scores: list[dict[tuple[str, str], float]] = []
    backptrs: list[dict[tuple[str, str], tuple[str, str]]] = []

    # --- Initialisation t=0 ---
    v0: dict[tuple[str, str], float] = {}
    for c in states[0]:
        em = emissions[0].get(c, -30.0)
        trans = transition_fn(_BOS, _BOS, c)
        score = w_transition * trans + w_emission * em
        state = (_BOS, c)
        if state not in v0 or score > v0[state]:
            v0[state] = score
    viterbi_scores.append(v0)
    backptrs.append({})

    # --- Recurrence t=1..n-1 ---
    for t in range(1, n):
        vt: dict[tuple[str, str], float] = {}
        bt: dict[tuple[str, str], tuple[str, str]] = {}

        for (pp, p), prev_score in viterbi_scores[t - 1].items():
            for c in states[t]:
                em = emissions[t].get(c, -30.0)
                trans = transition_fn(pp, p, c)
                score = (
                    prev_score
                    + w_transition * trans
                    + w_emission * em
                )
                new_state = (p, c)
                if new_state not in vt or score > vt[new_state]:
                    vt[new_state] = score
                    bt[new_state] = (pp, p)

        viterbi_scores.append(vt)
        backptrs.append(bt)

    # --- Meilleur etat final ---
    if not viterbi_scores[-1]:
        # Degenere : retourner le premier etat de chaque position
        return [
            (states[t][0] if states[t] else "", 0.0)
            for t in range(n)
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

    # Extraire la sequence de tags (2e element de chaque paire)
    tag_sequence = [s[1] for s in path_states]

    # --- Confiances par softmax sur les emissions ---
    results: list[tuple[str, float]] = []
    for t in range(n):
        chosen = tag_sequence[t]
        conf = _softmax_confidence(chosen, emissions[t])
        results.append((chosen, conf))

    return results


def _softmax_confidence(chosen: str, emission_scores: dict[str, float]) -> float:
    """Confiance du tag choisi via softmax sur les emissions."""
    if len(emission_scores) <= 1:
        return 1.0

    values = list(emission_scores.values())
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    if total == 0:
        return 1.0 / len(values)

    chosen_val = emission_scores.get(chosen, max_val)
    chosen_exp = math.exp(chosen_val - max_val)
    return chosen_exp / total
