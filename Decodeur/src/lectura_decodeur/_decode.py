"""Decodage CTC greedy — zero dependance.

Porte depuis train.py::_ctc_greedy_decode (version torch) vers numpy/listes.

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later
"""

from __future__ import annotations

import math


def ctc_greedy_decode(logits, blank_id: int = 0) -> list[int]:
    """Decodage CTC greedy sur un array (T, V) → liste d'IDs.

    Parameters
    ----------
    logits : array-like
        Logits de shape (T, V) — numpy ou liste de listes.
    blank_id : int
        ID du token blank/PAD (defaut : 0).

    Returns
    -------
    list[int]
        Sequence d'IDs decodee (sans blanks ni repetitions consecutives).
    """
    decoded: list[int] = []
    prev = -1
    for frame in logits:
        # argmax sur le vocabulaire
        best = 0
        best_val = frame[0]
        for i in range(1, len(frame)):
            if frame[i] > best_val:
                best = i
                best_val = frame[i]
        if best != blank_id and best != prev:
            decoded.append(best)
        prev = best
    return decoded


def ctc_greedy_decode_with_alternatives(
    logits,
    blank_id: int = 0,
    top_k: int = 5,
) -> list[dict]:
    """Decodage CTC greedy avec top-K alternatives par token emis.

    Pour chaque token emis (apres deduplication CTC), retourne le phone
    predit, sa confiance (softmax), et les top-K alternatives.

    Parameters
    ----------
    logits : array-like
        Logits de shape (T, V) — numpy ou liste de listes.
    blank_id : int
        ID du token blank/PAD.
    top_k : int
        Nombre d'alternatives a conserver par token.

    Returns
    -------
    list[dict]
        Tokens enrichis avec keys : phone_id, confidence, entropy,
        frame, alternatives (list[(phone_id, prob)]).
    """
    tokens: list[dict] = []
    prev_id = -1
    v = len(logits[0]) if len(logits) > 0 else 0

    for t, frame in enumerate(logits):
        # Softmax stable (numpy-free : listes pures)
        max_val = frame[0]
        for x in frame:
            if x > max_val:
                max_val = x
        exp_vals = [math.exp(x - max_val) for x in frame]
        exp_sum = sum(exp_vals)
        probs = [e / exp_sum for e in exp_vals]

        # Argmax
        best = 0
        best_val = probs[0]
        for i in range(1, v):
            if probs[i] > best_val:
                best = i
                best_val = probs[i]

        if best != blank_id and best != prev_id:
            # Entropie
            entropy = 0.0
            for p in probs:
                if p > 1e-10:
                    entropy -= p * math.log(p)

            # Top-K alternatives (exclure blank et argmax)
            # Tri partiel : collecter les indices par prob decroissante
            indexed = [(probs[i], i) for i in range(v)
                       if i != blank_id and i != best]
            indexed.sort(reverse=True)
            alternatives = [(idx, prob) for prob, idx in indexed[:top_k]]

            tokens.append({
                "phone_id": best,
                "confidence": best_val,
                "entropy": entropy,
                "frame": t,
                "frame_end": t,
                "alternatives": alternatives,
            })
        elif best != blank_id and best == prev_id and tokens:
            # Meme phone continue — etendre frame_end
            tokens[-1]["frame_end"] = t

        prev_id = best

    return tokens


def ids_vers_phones(ids: list[int], vocab_inv: dict[int, str]) -> str:
    """Convertit une liste d'IDs en chaine IPA lisible.

    Les phones sont separes par des espaces.
    Le token ``|`` (separateur de mots) est conserve tel quel.
    Les tokens speciaux ([PAD], [UNK]) sont ignores.

    Parameters
    ----------
    ids : list[int]
        Liste d'IDs de phones.
    vocab_inv : dict[int, str]
        Mapping ID → phone IPA.

    Returns
    -------
    str
        Chaine IPA, ex: ``"b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d"``.
    """
    phones: list[str] = []
    for idx in ids:
        token = vocab_inv.get(idx, "")
        if not token or token in ("[PAD]", "[UNK]"):
            continue
        phones.append(token)
    return " ".join(phones)
