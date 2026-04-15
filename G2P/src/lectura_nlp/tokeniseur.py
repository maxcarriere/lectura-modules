"""Tokenisation française pour le modèle unifié.

Découpe une phrase en tokens (mots + ponctuation), compatible avec l'entrée
du modèle char-level sur phrase entière.
"""

from __future__ import annotations

import re
import unicodedata

# Ponctuation séparée en tokens individuels
_PUNCT_RE = re.compile(
    r"""([,;:!?.\u2026\u00ab\u00bb"()\[\]{}\u2013\u2014/])"""
)

# Apostrophe française : "l'", "d'", "qu'", etc. → on sépare après l'apostrophe
_APOS_RE = re.compile(r"(\w+['\u2019])(\w)", re.UNICODE)


def tokeniser(text: str) -> list[str]:
    """Tokenise une phrase française en mots.

    - Sépare la ponctuation
    - Gère les apostrophes françaises (l', d', qu', n', s', j', c', m', t')
    - Gère les contractions (aujourd'hui reste intact)

    Returns:
        Liste de tokens (mots et ponctuation).
    """
    text = text.strip()
    if not text:
        return []

    # Normaliser les apostrophes typographiques
    text = text.replace("\u2019", "'")

    # Séparer la ponctuation
    text = _PUNCT_RE.sub(r" \1 ", text)

    # Gérer les apostrophes : séparer "l'" de "enfant" mais garder "aujourd'hui"
    from lectura_nlp._chargeur import kept_intact as _load_kept_intact
    _KEPT_INTACT = _load_kept_intact()

    tokens: list[str] = []
    for raw_tok in text.split():
        if not raw_tok:
            continue
        low = raw_tok.lower()
        if low in _KEPT_INTACT:
            tokens.append(raw_tok)
            continue

        # Séparer après apostrophe pour les clitiques
        m = _APOS_RE.match(raw_tok)
        if m:
            tokens.append(m.group(1))
            tokens.append(m.group(2) + raw_tok[m.end():])
        else:
            tokens.append(raw_tok)

    return tokens


def phrase_vers_chars(tokens: list[str]) -> tuple[list[str], list[int]]:
    """Convertit une liste de tokens en séquence de caractères pour le modèle.

    Utilise <SEP> entre les mots, <BOS> au début, <EOS> à la fin.

    Returns:
        (chars, word_ids) où word_ids[i] donne l'indice du mot
        auquel appartient chars[i] (-1 pour les marqueurs spéciaux).
    """
    chars: list[str] = ["<BOS>"]
    word_ids: list[int] = [-1]

    for w_idx, token in enumerate(tokens):
        if w_idx > 0:
            chars.append("<SEP>")
            word_ids.append(-1)
        for ch in token.lower():
            chars.append(ch)
            word_ids.append(w_idx)

    chars.append("<EOS>")
    word_ids.append(-1)

    return chars, word_ids
