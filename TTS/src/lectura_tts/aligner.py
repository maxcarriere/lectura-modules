"""Alignement DTW des durées TTS sur les syllabes du syllabeur.

Stratégie :
1. Segmenter les phonèmes TTS aux frontières de mots (espaces)
2. Pour chaque groupe, aligner les phonèmes TTS via DTW simplifié
3. Agréger les durées phonèmes au sein de chaque syllabe

Fallback : répartition proportionnelle si divergence majeure.
"""

from __future__ import annotations

import unicodedata
from typing import Protocol

from lectura_tts.ipa import iter_phonemes
from lectura_tts.models import GroupeTiming, PhonemeTiming, SyllabeTiming, TTSResult


class _SyllabeProto(Protocol):
    phone: str
    ortho: str


class _GroupeProto(Protocol):
    @property
    def syllabes(self) -> list: ...


def align_tts_to_syllables(
    tts_result: TTSResult,
    groups: list,
) -> list[GroupeTiming]:
    """Aligne les timings TTS aux groupes du syllabeur.

    Retourne un GroupeTiming par groupe avec syllabes, avec start_ms/end_ms.
    Les groupes doivent avoir un attribut .syllabes (list avec .phone et .ortho).
    """
    tts_phonemes = tts_result.phoneme_timings
    if not tts_phonemes:
        return _fallback_proportional(groups, 0.0, 0.0)

    merged = _merge_combining_marks(tts_phonemes)
    tts_words = _segment_by_words(merged)

    groups_with_syllables = [
        (i, g) for i, g in enumerate(groups) if hasattr(g, "syllabes") and g.syllabes
    ]

    result: list[GroupeTiming] = []
    tts_word_idx = 0

    for group_idx, group in groups_with_syllables:
        if tts_word_idx < len(tts_words):
            word_phonemes = tts_words[tts_word_idx]
            tts_word_idx += 1
        else:
            prev_end = (result[-1].syllabe_timings[-1].end_ms
                        if result and result[-1].syllabe_timings else 0.0)
            gt = _fallback_group(group_idx, group, prev_end)
            result.append(gt)
            continue

        gt = _align_group_to_word(group_idx, group, word_phonemes)
        result.append(gt)

    return result


def _merge_combining_marks(phonemes: list[PhonemeTiming]) -> list[PhonemeTiming]:
    """Fusionne les combining marks et stress marks avec leur phonème adjacent."""
    if not phonemes:
        return []

    merged: list[PhonemeTiming] = []

    for pt in phonemes:
        ch = pt.ipa
        if ch and unicodedata.category(ch[0]).startswith("M") and merged:
            prev = merged[-1]
            merged[-1] = PhonemeTiming(
                ipa=prev.ipa + ch, start_ms=prev.start_ms, end_ms=pt.end_ms,
            )
            continue
        if ch in ("ˈ", "ˌ"):
            if merged:
                prev = merged[-1]
                merged[-1] = PhonemeTiming(
                    ipa=prev.ipa, start_ms=prev.start_ms, end_ms=pt.end_ms,
                )
            continue
        merged.append(PhonemeTiming(ipa=ch, start_ms=pt.start_ms, end_ms=pt.end_ms))

    return merged


def _segment_by_words(phonemes: list[PhonemeTiming]) -> list[list[PhonemeTiming]]:
    """Segmente les phonèmes TTS en mots (séparés par les espaces)."""
    words: list[list[PhonemeTiming]] = []
    current: list[PhonemeTiming] = []

    for pt in phonemes:
        if pt.ipa == " ":
            if current:
                words.append(current)
                current = []
            continue
        current.append(pt)

    if current:
        words.append(current)
    return words


def _align_group_to_word(
    group_idx: int,
    group,
    tts_phonemes: list[PhonemeTiming],
) -> GroupeTiming:
    syll_phoneme_counts = []
    for syll in group.syllabes:
        count = len(iter_phonemes(syll.phone))
        syll_phoneme_counts.append(max(1, count))

    total_tts_phonemes = len(tts_phonemes)
    if total_tts_phonemes == 0:
        return _fallback_group(group_idx, group, 0.0)

    alignment = _dtw_align(group, tts_phonemes)
    if alignment is not None:
        return _build_timing_from_alignment(group_idx, group, tts_phonemes, alignment)

    return _proportional_split(group_idx, group, tts_phonemes, syll_phoneme_counts)


def _dtw_align(group, tts_phonemes: list[PhonemeTiming]) -> list[tuple[int, int]] | None:
    syll_phones: list[str] = []
    for syll in group.syllabes:
        syll_phones.extend(iter_phonemes(syll.phone))

    tts_phones = [pt.ipa for pt in tts_phonemes]

    n = len(syll_phones)
    m = len(tts_phones)
    if n == 0 or m == 0:
        return None

    cost = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = _phoneme_distance(syll_phones[i - 1], tts_phones[j - 1])
            cost[i][j] = d + min(cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1])

    avg_cost = cost[n][m] / max(n, m)
    if avg_cost > 0.8:
        return None

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (cost[i - 1][j - 1], i - 1, j - 1),
            (cost[i - 1][j], i - 1, j),
            (cost[i][j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])

    path.reverse()
    return path


def _phoneme_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0

    base_a = a[0] if a else ""
    base_b = b[0] if b else ""
    if base_a == base_b:
        return 0.2

    _EQUIVALENCES = {
        frozenset({"ɹ", "ʁ"}), frozenset({"ɾ", "ʁ"}), frozenset({"r", "ʁ"}),
        frozenset({"g", "ɡ"}), frozenset({"ɛ", "e"}), frozenset({"ɔ", "o"}),
        frozenset({"œ", "ø"}), frozenset({"a", "ɑ"}),
    }

    pair = frozenset({base_a, base_b})
    if pair in _EQUIVALENCES:
        return 0.3

    return 1.0


def _build_timing_from_alignment(
    group_idx: int, group, tts_phonemes: list[PhonemeTiming],
    alignment: list[tuple[int, int]],
) -> GroupeTiming:
    syll_to_tts: dict[int, int] = {}
    for syll_idx, tts_idx in alignment:
        if syll_idx not in syll_to_tts:
            syll_to_tts[syll_idx] = tts_idx

    syll_phoneme_offset = 0
    syllabe_timings: list[SyllabeTiming] = []

    for syll in group.syllabes:
        n_phones = max(1, len(iter_phonemes(syll.phone)))

        tts_indices = []
        for k in range(syll_phoneme_offset, syll_phoneme_offset + n_phones):
            if k in syll_to_tts:
                tts_indices.append(syll_to_tts[k])

        if tts_indices:
            first = min(tts_indices)
            last = max(tts_indices)
            start_ms = tts_phonemes[first].start_ms
            end_ms = tts_phonemes[last].end_ms
        elif syllabe_timings:
            start_ms = syllabe_timings[-1].end_ms
            end_ms = start_ms
        else:
            start_ms = tts_phonemes[0].start_ms if tts_phonemes else 0.0
            end_ms = start_ms

        syllabe_timings.append(SyllabeTiming(
            phone=syll.phone, ortho=syll.ortho, start_ms=start_ms, end_ms=end_ms,
        ))
        syll_phoneme_offset += n_phones

    return GroupeTiming(group_index=group_idx, syllabe_timings=syllabe_timings)


def _proportional_split(
    group_idx: int, group, tts_phonemes: list[PhonemeTiming],
    syll_phoneme_counts: list[int],
) -> GroupeTiming:
    total_count = sum(syll_phoneme_counts)
    if total_count == 0:
        return GroupeTiming(group_index=group_idx)

    total_start = tts_phonemes[0].start_ms
    total_end = tts_phonemes[-1].end_ms
    total_dur = total_end - total_start

    syllabe_timings: list[SyllabeTiming] = []
    cursor = total_start

    for syll, count in zip(group.syllabes, syll_phoneme_counts):
        dur = total_dur * count / total_count
        syllabe_timings.append(SyllabeTiming(
            phone=syll.phone, ortho=syll.ortho, start_ms=cursor, end_ms=cursor + dur,
        ))
        cursor += dur

    return GroupeTiming(group_index=group_idx, syllabe_timings=syllabe_timings)


def _fallback_group(group_idx: int, group, start_ms: float) -> GroupeTiming:
    return GroupeTiming(
        group_index=group_idx,
        syllabe_timings=[
            SyllabeTiming(phone=syll.phone, ortho=syll.ortho,
                          start_ms=start_ms, end_ms=start_ms)
            for syll in group.syllabes
        ],
    )


def _fallback_proportional(groups: list, start_ms: float, end_ms: float) -> list[GroupeTiming]:
    result: list[GroupeTiming] = []
    for i, group in enumerate(groups):
        if hasattr(group, "syllabes") and group.syllabes:
            result.append(_fallback_group(i, group, start_ms))
    return result
