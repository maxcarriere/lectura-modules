"""Labels P2G par caractère IPA : _CONT + labels_from_p2g_alignment + reconstruct_ortho.

En P2G (phonème → graphème), l'input est une séquence de caractères IPA Unicode.
Chaque caractère IPA reçoit un label graphème (le graphème aligné correspondant).
Les combining marks (ex: nasales ɑ̃ = 2 car Unicode) reçoivent le label _CONT.
"""

from __future__ import annotations

import unicodedata

_CONT = "_CONT"


def _clean_grapheme(gr: str) -> str:
    """Nettoie un graphème aligné : supprime ° (muettes) et ² (diacritiques)."""
    return gr.replace("°", "").replace("²", "")


def labels_from_p2g_alignment(
    phone_str: str,
    dec_ph: list[str],
    dec_gr: list[str],
) -> list[str]:
    """Convertit un alignement en labels graphèmes par caractère IPA Unicode.

    Pour chaque phonème de l'alignement, le graphème nettoyé est assigné au
    premier caractère Unicode du phonème. Les combining marks (nasales, etc.)
    reçoivent le label _CONT.

    Args:
        phone_str: La chaîne IPA complète du mot (ex: "bɔ̃ʒuʁ")
        dec_ph: Les phonèmes décomposés de l'alignement (ex: ["b", "ɔ̃", "ʒ", "u", "ʁ"])
        dec_gr: Les graphèmes décomposés correspondants (ex: ["b", "on", "j", "ou", "r"])

    Returns:
        Liste de labels, un par caractère Unicode de phone_str.
        Ex: ["b", "on", "_CONT", "j", "ou", "r"] pour "bɔ̃ʒuʁ"
    """
    # Build a mapping: for each phoneme in dec_ph, assign its grapheme
    # to the IPA characters, expanding combining marks as _CONT
    labels: list[str] = []

    # Walk through phone_str character by character
    # and match against dec_ph phonemes
    ph_idx = 0  # index into dec_ph
    char_idx = 0  # index into phone_str

    while char_idx < len(phone_str):
        if ph_idx >= len(dec_ph):
            # Remaining characters get _CONT
            labels.append(_CONT)
            char_idx += 1
            continue

        phoneme = dec_ph[ph_idx]
        grapheme = _clean_grapheme(dec_gr[ph_idx])

        # Check how many Unicode characters this phoneme spans
        ph_chars = list(phoneme)

        # Assign the grapheme to the first character (base character)
        labels.append(grapheme)
        char_idx += 1

        # Remaining characters of this phoneme (combining marks) get _CONT
        for i in range(1, len(ph_chars)):
            if char_idx < len(phone_str):
                labels.append(_CONT)
                char_idx += 1

        ph_idx += 1

    return labels


def reconstruct_ortho(labels: list[str]) -> str:
    """Reconstruit l'orthographe depuis les labels par caractère IPA.

    Concatène tous les labels non-_CONT et non-<PAD>.
    """
    return "".join(label for label in labels if label != _CONT and label != "<PAD>")
