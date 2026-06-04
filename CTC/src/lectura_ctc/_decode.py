"""Decodage CTC greedy — zero dependance.

Porte depuis train.py::_ctc_greedy_decode (version torch) vers numpy/listes.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later
"""

from __future__ import annotations


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
