"""Decodeur Viterbi pour la desambiguation POS.

Combine les scores d'emission (confiance du tagger) avec les log-probabilites
de transition bigram pour choisir la meilleure sequence POS.

L'espace d'etats est tiny (top-K=3-5 par position), donc Viterbi est
O(n * K^2) ~ instantane.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def charger_matrice_transition(path: str | Path) -> dict[str, dict[str, float]]:
    """Charge la matrice de transition bigram depuis un fichier JSON.

    Returns:
        Dict[context -> Dict[label -> log_prob]].
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["log_probs"]


def viterbi_decode(
    pos_scores_seq: list[list[tuple[str, float]]],
    transition_log_probs: dict[str, dict[str, float]],
    *,
    w_emission: float = 0.7,
    w_transition: float = 0.3,
) -> list[tuple[str, float]]:
    """Decode Viterbi sur une sequence de scores POS.

    Args:
        pos_scores_seq: Pour chaque position, liste de (pos_label, prob).
            prob est une probabilite normalisee (somme ~1).
        transition_log_probs: Dict[context -> Dict[label -> log_prob]].
        w_emission: Poids de l'emission (confiance tagger).
        w_transition: Poids de la transition (matrice bigram).

    Returns:
        Liste de (pos_label, confiance) par position.
        confiance = softmax locale du score Viterbi.
    """
    n = len(pos_scores_seq)
    if n == 0:
        return []

    BOS = "<BOS>"

    # Filtrer les positions sans scores (garde le top-1 tel quel)
    # Pour chaque position, construire les etats candidats
    states: list[list[str]] = []
    emission_log: list[dict[str, float]] = []

    for pos_scores in pos_scores_seq:
        if not pos_scores:
            states.append([""])
            emission_log.append({"": 0.0})
        else:
            s = [label for label, _ in pos_scores]
            e = {}
            for label, prob in pos_scores:
                # Clamp prob pour eviter log(0)
                e[label] = math.log(max(prob, 1e-10))
            states.append(s)
            emission_log.append(e)

    def _trans_log(prev: str, curr: str) -> float:
        """Log-prob de transition prev->curr, avec fallback uniforme."""
        if prev in transition_log_probs:
            row = transition_log_probs[prev]
            if curr in row:
                return row[curr]
        # Fallback: uniforme sur tous les labels (~log(1/19))
        return -3.0

    # Viterbi forward pass
    # viterbi[t][state] = (score, prev_state)
    viterbi: list[dict[str, tuple[float, str]]] = []

    # t=0: transition depuis BOS
    v0: dict[str, tuple[float, str]] = {}
    for s in states[0]:
        emit = emission_log[0].get(s, -10.0)
        trans = _trans_log(BOS, s)
        score = w_emission * emit + w_transition * trans
        v0[s] = (score, BOS)
    viterbi.append(v0)

    # t=1..n-1
    for t in range(1, n):
        vt: dict[str, tuple[float, str]] = {}
        for curr_s in states[t]:
            best_score = -math.inf
            best_prev = ""
            emit = emission_log[t].get(curr_s, -10.0)
            for prev_s in states[t - 1]:
                prev_score = viterbi[t - 1][prev_s][0]
                trans = _trans_log(prev_s, curr_s)
                score = prev_score + w_emission * emit + w_transition * trans
                if score > best_score:
                    best_score = score
                    best_prev = prev_s
            vt[curr_s] = (best_score, best_prev)
        viterbi.append(vt)

    # Backtrack
    best_path: list[str] = [""] * n

    # Trouver le meilleur etat final
    last = viterbi[n - 1]
    best_final = max(last, key=lambda s: last[s][0])
    best_path[n - 1] = best_final

    for t in range(n - 2, -1, -1):
        best_path[t] = viterbi[t + 1][best_path[t + 1]][1]

    # Calculer la confiance par position (softmax des scores Viterbi)
    result: list[tuple[str, float]] = []
    for t in range(n):
        chosen = best_path[t]
        scores_t = viterbi[t]
        if len(scores_t) <= 1:
            result.append((chosen, 1.0))
        else:
            # Softmax locale
            vals = [scores_t[s][0] for s in scores_t]
            max_v = max(vals)
            exp_vals = {s: math.exp(scores_t[s][0] - max_v) for s in scores_t}
            total = sum(exp_vals.values())
            conf = exp_vals[chosen] / total if total > 0 else 1.0
            result.append((chosen, conf))

    return result
