"""Segmentation et correction lexicale pour le post-traitement STT.

Contient les fonctions de pre-traitement qui operent entre le parsing CTC
et la conversion P2G :
  - strip_liaisons: retire les consonnes de liaison absorbees par le CTC
  - split_elisions: separe les clitiques elides (l'ami → l + ami)
  - split_merged_words: decoupe les mots colles en exactement 2 mots connus

Note : split_merged_words utilise un split strict en 2 parties (v3.1+).
Les decompositions en 3+ parties degradaient le WER de +2.7% et ont ete
desactivees. La fonction _decompose_dp reste disponible pour _correction.py.

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lectura_stt._lexicon import PhoneLexicon

from lectura_stt._lexicon import (
    LIAISON_CONSONANTS,
    CLITIC_MAP,
    IPA_VOWELS,
    SCHWA,
    _CONFUSION_MAP,
    _ipa_grapheme_clusters,
    _normalize_ipa,
    _tier4_ctc_confusions,
)


def strip_liaisons(
    ipa_words: list[str],
    lexicon: "PhoneLexicon",
) -> list[str]:
    """Retire les consonnes de liaison en debut de mot.

    Quand le mot precedent a une liaison connue (ipa_with_liaison) et que
    le mot courant commence par cette consonne, on la retire si le reste
    est un mot connu.

    Ex: ['le', 'zami'] → ['le', 'ami']  (car 'lez' in ipa_with_liaison)
    """
    result = list(ipa_words)
    for i in range(1, len(result)):
        word = result[i]
        if word in lexicon.phone_set:
            continue
        clusters = _ipa_grapheme_clusters(word)
        if len(clusters) < 2:
            continue
        head = clusters[0]
        if head not in LIAISON_CONSONANTS:
            continue
        rest = "".join(clusters[1:])
        if rest not in lexicon.phone_set and rest not in lexicon.phone_set_reliable:
            continue
        prev = result[i - 1]
        if (prev + head) in lexicon.ipa_with_liaison:
            result[i] = rest
    return result


def split_elisions(
    ipa_words: list[str],
    lexicon: "PhoneLexicon",
) -> list[str]:
    """Detecte et splitte les clitiques elides dans une liste de mots IPA.

    Ex: ['labatwAR'] → ['l', 'abatwAR']  (si 'abatwAR' in lexique)

    Le P2G traitera ensuite chaque partie separement.
    """
    result = []
    for word in ipa_words:
        clusters = _ipa_grapheme_clusters(word)
        if (len(clusters) >= 3
                and clusters[0] in CLITIC_MAP
                and clusters[0] not in IPA_VOWELS):
            # Ne PAS splitter si le mot entier est connu
            if word in lexicon.phone_set:
                result.append(word)
                continue
            rest = "".join(clusters[1:])
            if rest in lexicon.phone_set or rest in lexicon.phone_set_reliable:
                result.append(clusters[0])
                result.append(rest)
                continue
        result.append(word)
    return result


# ── Split de mots colles par le CTC (decomposition DP) ──────

_SPLIT_MIN_FREQ = 5.0           # Frequence min par partie (mots normaux)
_SPLIT_CLITIC_MIN_FREQ = 1.0    # Frequence min pour clitiques
_SPLIT_MIN_SEG = 2              # Segments min par partie (sauf clitiques)
_SPLIT_MAX_PARTS = 4            # Max parties dans une decomposition
_SPLIT_MAX_CHUNK_SEG = 15       # Max segments par chunk
_SPLIT_TOP_K = 8                # Max chemins par position DP


def _match_chunk_for_split(
    chunk_str: str,
    lex: "PhoneLexicon",
    min_freq: float,
) -> list[tuple[str, float, str]]:
    """Essaie de matcher un chunk IPA a un mot connu.

    Returns: [(matched_phone, freq, method), ...]
    method: "exact" ou "perturb"
    """
    import math
    results: list[tuple[str, float, str]] = []

    # Match exact
    if chunk_str in lex.phone_set:
        freq = lex.best_freq(chunk_str)
        if freq >= min_freq:
            results.append((chunk_str, freq, "exact"))
            return results

    # Perturbations CTC (une seule substitution)
    segs = _ipa_grapheme_clusters(chunk_str)
    n_s = len(segs)
    seen: set[str] = set()

    for idx in range(n_s):
        base = segs[idx][0] if segs[idx] else segs[idx]
        if base in _CONFUSION_MAP:
            for repl in _CONFUSION_MAP[base]:
                new_seg = repl + segs[idx][1:]
                variant = "".join(segs[:idx] + [new_seg] + segs[idx + 1:])
                if variant not in seen and variant in lex.phone_set:
                    seen.add(variant)
                    freq = lex.best_freq(variant)
                    if freq >= min_freq:
                        results.append((variant, freq, "perturb"))

        # Suppression schwa
        if segs[idx] == SCHWA:
            variant = "".join(segs[:idx] + segs[idx + 1:])
            if variant and variant not in seen and variant in lex.phone_set:
                seen.add(variant)
                freq = lex.best_freq(variant)
                if freq >= min_freq:
                    results.append((variant, freq, "perturb"))

    # Insertion schwa
    for idx in range(n_s + 1):
        variant = "".join(segs[:idx] + [SCHWA] + segs[idx:])
        if variant not in seen and variant in lex.phone_set:
            seen.add(variant)
            freq = lex.best_freq(variant)
            if freq >= min_freq:
                results.append((variant, freq, "perturb"))

    return results


def _freq_score(freq: float) -> float:
    """Score de frequence normalise."""
    import math
    if freq <= 0:
        return 0.0
    return min(1.0, math.log10(freq + 1) / 5.0)


def _decompose_dp(
    segments: list[str],
    lex: "PhoneLexicon",
) -> list[tuple[list[str], float]]:
    """Decomposition DP d'une sequence IPA en mots connus.

    Explore toutes les facons de decouper ``segments`` en mots connus,
    en gerant matchs exacts/perturbes et consonnes de liaison aux frontieres.

    Returns: [(words, score), ...] trie par score decroissant.
    """
    n = len(segments)
    if n < 4:
        return []

    # dp[i] = list of (words, score, n_parts)
    dp: list[list[tuple[list[str], float, int]]] = [[] for _ in range(n + 1)]
    dp[0] = [([], 0.0, 0)]

    for i in range(n):
        if not dp[i]:
            continue
        dp[i].sort(key=lambda x: -x[1])
        dp[i] = dp[i][:_SPLIT_TOP_K]

        max_j = min(n + 1, i + _SPLIT_MAX_CHUNK_SEG + 1)

        for j in range(i + 1, max_j):
            n_seg = j - i
            chunk_str = "".join(segments[i:j])

            is_clitic = (n_seg == 1 and chunk_str in CLITIC_MAP)
            if n_seg < _SPLIT_MIN_SEG and not is_clitic:
                continue

            chunk_min = _SPLIT_CLITIC_MIN_FREQ if is_clitic else _SPLIT_MIN_FREQ
            matches = _match_chunk_for_split(chunk_str, lex, chunk_min)
            if not matches:
                continue

            matches.sort(key=lambda x: -x[1])
            matched_phone, freq, method = matches[0]

            score_inc = _freq_score(freq)
            if method == "perturb":
                score_inc -= 0.05

            for words, cum_score, n_parts in dp[i]:
                if n_parts >= _SPLIT_MAX_PARTS:
                    continue
                new_words = words + [matched_phone]
                new_score = cum_score + score_inc
                new_parts = n_parts + 1

                dp[j].append((new_words, new_score, new_parts))

                # Transition liaison : consommer une consonne de liaison en j
                if (j < n
                        and method == "exact"
                        and segments[j] in LIAISON_CONSONANTS):
                    next_is_vowel = (
                        j + 1 < n
                        and len(segments[j + 1]) > 0
                        and segments[j + 1][0] in IPA_VOWELS
                    )
                    if next_is_vowel:
                        with_liaison = chunk_str + segments[j]
                        if with_liaison in lex.ipa_with_liaison:
                            dp[j + 1].append((new_words, new_score, new_parts))

    # Collecter les decompositions completes (au moins 2 parties)
    results: list[tuple[list[str], float]] = []
    for words, score, n_parts in dp[n]:
        if n_parts >= 2:
            adjusted = score - 0.05 * (n_parts - 2)
            results.append((words, adjusted))

    results.sort(key=lambda x: -x[1])
    return results[:5]


_SPLIT_2P_MIN_FREQ = 20.0       # Frequence min par partie (split 2-parts)


def split_merged_words(
    ipa_words: list[str],
    lexicon: "PhoneLexicon",
) -> list[str]:
    """Pre-traitement : decoupe les mots inconnus en exactement 2 mots connus.

    Seuls les splits en 2 parties exactes (les deux frequentes) sont tentes.
    Les decompositions en 3+ parties degradent fortement le WER et sont
    desactivees. Le P2G gere mieux les mots OOV entiers que des fragments
    incorrects.

    Doit etre appele AVANT la conversion P2G.
    """
    import math

    result = []
    for word in ipa_words:
        word = _normalize_ipa(word)
        if word in lexicon.phone_set:
            result.append(word)
            continue

        if _tier4_ctc_confusions(word, lexicon):
            result.append(word)
            continue

        segments = _ipa_grapheme_clusters(word)
        n = len(segments)
        if n < 4:
            result.append(word)
            continue

        best_split: tuple[str, str] | None = None
        best_score = 0.0

        for j in range(2, n - 1):
            left = _normalize_ipa("".join(segments[:j]))
            right = _normalize_ipa("".join(segments[j:]))

            if left not in lexicon.phone_set or right not in lexicon.phone_set:
                continue

            freq_l = lexicon.best_freq(left)
            freq_r = lexicon.best_freq(right)
            if freq_l >= _SPLIT_2P_MIN_FREQ and freq_r >= _SPLIT_2P_MIN_FREQ:
                score = math.log10(freq_l + 1) + math.log10(freq_r + 1)
                if score > best_score:
                    best_score = score
                    best_split = (left, right)

        if best_split:
            result.extend(best_split)
        else:
            result.append(word)

    return result
