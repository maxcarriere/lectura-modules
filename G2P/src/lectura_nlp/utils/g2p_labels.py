"""Labels G2P par caractère : _CONT + labels_from_alignment + reconstruct_ipa."""

from __future__ import annotations

_CONT = "_CONT"

_ORTHO_VOWELS = set("aeiouyàâéèêëïîôûùüœæ")


def labels_from_alignment(
    word: str,
    dec_ph: list[str],
    dec_spans: list[tuple[int, int]],
) -> list[str]:
    """Convertit un alignement (phonèmes + spans) en labels par caractère.

    Chaque caractère du mot reçoit :
    - le phonème IPA correspondant (premier caractère du span)
    - _CONT si c'est une continuation d'un graphème multi-caractères
    """
    n = len(word)
    labels = [_CONT] * n

    for phoneme, (start, end) in zip(dec_ph, dec_spans):
        if start < n:
            labels[start] = phoneme

    return labels


def reconstruct_ipa(labels: list[str]) -> str:
    """Reconstruit la chaîne IPA depuis les labels caractère."""
    return "".join(label for label in labels if label != _CONT)
