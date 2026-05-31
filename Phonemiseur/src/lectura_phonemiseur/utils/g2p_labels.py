"""Labels G2P par caractère : _CONT + labels_from_alignment + reconstruct_ipa."""

from __future__ import annotations

_CONT = "_CONT"
_SEP_APOS = "_SEP_APOS"
_SEP_HYPHEN = "_SEP_HYPHEN"
_SEP_SPACE = "_SEP_SPACE"
_SKIP_LABELS = {"_CONT", "<PAD>", _SEP_HYPHEN, _SEP_APOS, _SEP_SPACE}

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


def reconstruct_ipa(
    labels: list[str],
    *,
    sep_apos: bool = False,
    sep_hyphen: bool = False,
    sep_space: bool = False,
) -> str:
    """Reconstruit la chaine IPA depuis les labels caractere.

    Par defaut, tous les separateurs sont filtres (comportement historique).
    Les options permettent d'inclure les caracteres de separation dans la sortie :
      - sep_apos=True  : inclut ' aux positions d'apostrophe
      - sep_hyphen=True : inclut - aux positions de tiret
      - sep_space=True  : inclut un espace aux positions de frontiere de mot
    """
    parts: list[str] = []
    for label in labels:
        if label in (_CONT, "<PAD>"):
            continue
        if label == _SEP_APOS:
            if sep_apos:
                parts.append("'")
            continue
        if label == _SEP_HYPHEN:
            if sep_hyphen:
                parts.append("-")
            continue
        if label == _SEP_SPACE:
            if sep_space:
                parts.append(" ")
            continue
        parts.append(label)
    return "".join(parts)
