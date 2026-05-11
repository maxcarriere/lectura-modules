"""Tokenisation IPA pour le modèle unifié P2G.

Découpe une phrase IPA en mots IPA et prépare la séquence de
caractères pour le modèle char-level sur phrase entière.
"""

from __future__ import annotations


def tokeniser_ipa(ipa_sentence: str) -> list[str]:
    """Tokenise une phrase IPA en mots (split sur espaces).

    Args:
        ipa_sentence: Phrase IPA avec mots séparés par des espaces.

    Returns:
        Liste de mots IPA.
    """
    return [w for w in ipa_sentence.strip().split() if w]


def ipa_phrase_vers_chars(
    ipa_words: list[str],
) -> tuple[list[str], list[int], list[int], list[int]]:
    """Convertit une liste de mots IPA en séquence de caractères pour le modèle.

    Utilise <SEP> entre les mots, <BOS> au début, <EOS> à la fin.

    Args:
        ipa_words: Liste de mots IPA (ex: ["le", "ʃa"])

    Returns:
        (chars, word_ids, word_starts, word_ends) :
        - chars: liste de caractères IPA Unicode + marqueurs spéciaux
        - word_ids: word_ids[i] = indice du mot (-1 pour marqueurs)
        - word_starts: position du 1er caractère de chaque mot
        - word_ends: position du dernier caractère de chaque mot
    """
    chars: list[str] = ["<BOS>"]
    word_ids: list[int] = [-1]
    word_starts: list[int] = []
    word_ends: list[int] = []

    for w_idx, word in enumerate(ipa_words):
        if w_idx > 0:
            chars.append("<SEP>")
            word_ids.append(-1)

        word_start = len(chars)
        for ch in word:
            chars.append(ch)
            word_ids.append(w_idx)
        word_end = len(chars) - 1

        word_starts.append(word_start)
        word_ends.append(word_end)

    chars.append("<EOS>")
    word_ids.append(-1)

    return chars, word_ids, word_starts, word_ends
