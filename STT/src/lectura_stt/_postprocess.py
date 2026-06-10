"""Correction post-P2G pour le pipeline STT optimal.

Fusion de mots sur-segmentes et elisions clitiques apres la conversion P2G.

Fonctions principales :
  - merge_and_rescore : fusion de mots sur-segmentes
  - try_elision_merges : fusion clitiques elides

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lectura_stt._lexicon import PhoneLexicon

from lectura_stt._lexicon import (
    IPA_VOWELS,
    _ipa_grapheme_clusters,
    _normalize_ipa,
)


# ── Scoring helpers ──────────────────────────────────────────


def _tier_score(tier: int) -> float:
    """Score de confiance base sur le tier de perturbation."""
    if tier == 0:
        return 1.0
    elif tier <= 3:
        return 0.8
    elif tier <= 5:
        return 0.6
    elif tier == 6:
        return 0.4
    else:
        return 0.2


def _freq_score(freq: float) -> float:
    """Score de frequence normalise."""
    if freq <= 0:
        return 0.0
    return min(1.0, math.log10(freq + 1) / 5.0)


# ── Shift compound joins ────────────────────────────────────


def _shift_compound_joins(
    old_words: list[str],
    new_words: list[str],
    joins: set[int],
) -> set[int]:
    """Recalcul des indices compound_joins apres modification de la liste de mots.

    Quand merge_and_rescore ou try_elision_merges modifie la liste de mots,
    les indices de compound_joins doivent etre mis a jour pour pointer vers
    les bonnes positions dans la nouvelle liste.

    Strategie : on mappe chaque ancien indice vers le nouvel indice en
    parcourant les deux listes en parallele.
    """
    if not joins:
        return set()

    # Construire un mapping old_idx → new_idx
    # Les merges remplacent N mots par 1, donc on suit la position
    new_joins: set[int] = set()
    old_idx = 0
    new_idx = 0

    while old_idx < len(old_words) and new_idx < len(new_words):
        if old_idx in joins:
            new_joins.add(new_idx)
        # Avancer : si le mot new est le meme, avancer les deux
        # Si le mot new est different (merge), avancer old jusqu'a
        # ce qu'on ait consomme tous les mots merges
        if old_idx < len(old_words) and new_idx < len(new_words):
            if old_words[old_idx] == new_words[new_idx]:
                old_idx += 1
                new_idx += 1
            else:
                # Merge : plusieurs old_words → 1 new_word
                # Trouver combien d'anciens mots ont ete fusionnes
                new_idx += 1
                old_idx += 1
                # Consommer les mots supplementaires absorbes
                while old_idx < len(old_words):
                    if new_idx < len(new_words) and old_words[old_idx] == new_words[new_idx]:
                        break
                    old_idx += 1

    # Filtrer les joins invalides (au-dela de la nouvelle longueur)
    return {j for j in new_joins if j < len(new_words) - 1}


# ── MergeCandidate ──────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MergeCandidate:
    """Un candidat de fusion de mots consecutifs."""
    start: int          # index du premier mot
    end: int            # index exclusif du dernier mot
    merged_ipa: str     # IPA concatene
    resolved_ipa: str   # IPA apres perturbation
    ortho: str          # orthographe du mot fusionne
    freq: float         # frequence lexicale
    tier: int           # tier de resolution (0=exact, 4=perturbe)


# ── Generation de candidats de fusion ────────────────────────


def merge_candidates(
    ipa_words: list[str],
    ortho_words: list[str],
    pos_confidences: list[float],
    lexicon: "PhoneLexicon",
    *,
    max_merge: int = 4,
    confidence_threshold: float = 0.6,
    min_freq: float = 1.0,
) -> list[MergeCandidate]:
    """Genere les candidats de fusion pour des mots potentiellement sur-segmentes."""
    n = len(ipa_words)
    candidates: list[MergeCandidate] = []

    for i in range(n - 1):
        segments_i = _ipa_grapheme_clusters(ipa_words[i])
        freq_i = lexicon.best_freq(ipa_words[i])
        is_suspect = (
            pos_confidences[i] < confidence_threshold
            or len(segments_i) <= 2
            or freq_i < 20.0
        )
        if not is_suspect:
            segments_next = _ipa_grapheme_clusters(ipa_words[i + 1])
            freq_next = lexicon.best_freq(ipa_words[i + 1])
            next_suspect = (
                pos_confidences[i + 1] < confidence_threshold
                or len(segments_next) <= 2
                or freq_next < 20.0
            )
            if not next_suspect:
                continue

        for k in range(2, min(max_merge + 1, n - i + 1)):
            end = i + k
            merged_ipa = "".join(ipa_words[i:end])
            merged_ipa_n = _normalize_ipa(merged_ipa)

            effective_min_freq = min_freq if k <= 2 else min(min_freq, 0.1)

            # Chercher exact
            if lexicon.exists(merged_ipa_n):
                freq = lexicon.best_freq(merged_ipa_n)
                if freq >= effective_min_freq:
                    ortho = lexicon.best_ortho(merged_ipa_n)
                    if ortho:
                        candidates.append(MergeCandidate(
                            start=i, end=end,
                            merged_ipa=merged_ipa,
                            resolved_ipa=merged_ipa_n,
                            ortho=ortho, freq=freq, tier=0,
                        ))
                        continue

            # Chercher avec perturbations CTC
            entries, _rmap = lexicon.all_entries_with_perturbations(merged_ipa_n)
            for entry in entries:
                freq = entry.get("freq", 0) or 0
                if freq >= effective_min_freq:
                    ortho = entry.get("ortho", "")
                    source = entry.get("_source", "exact")
                    tier = 0 if source == "exact" else 4
                    if ortho:
                        candidates.append(MergeCandidate(
                            start=i, end=end,
                            merged_ipa=merged_ipa,
                            resolved_ipa=entry.get("phone", merged_ipa_n),
                            ortho=ortho, freq=freq, tier=tier,
                        ))
                    break

    candidates.sort(key=lambda c: (c.start, -c.freq))
    return candidates


# ── Scoring merge vs split ───────────────────────────────────


def score_merge_vs_split(
    candidate: MergeCandidate,
    ipa_words: list[str],
    ortho_words: list[str],
    pos_confidences: list[float],
    lexicon: "PhoneLexicon",
) -> tuple[float, float]:
    """Compare le score du merge vs garder les fragments separes."""
    n_merged = candidate.end - candidate.start

    fragment_raw_freqs: list[float] = []
    fragment_content_freqs: list[float] = []
    for j in range(candidate.start, candidate.end):
        freq_j = lexicon.best_freq(ipa_words[j])
        fragment_raw_freqs.append(freq_j)
        segs_j = _ipa_grapheme_clusters(ipa_words[j])
        if len(segs_j) > 2:
            fragment_content_freqs.append(freq_j)

    relevant_freqs = fragment_content_freqs if fragment_content_freqs else fragment_raw_freqs
    min_freq_fragment = min(relevant_freqs) if relevant_freqs else 0.0

    # Score merge
    s_freq_merge = _freq_score(candidate.freq)
    s_tier = _tier_score(candidate.tier)

    freq_ratio = candidate.freq / max(min_freq_fragment, 0.01)
    ratio_bonus = min(1.0, math.log10(max(freq_ratio, 1.0)) / 3.0)

    merged_segs = sum(
        len(_ipa_grapheme_clusters(ipa_words[j]))
        for j in range(candidate.start, candidate.end)
    )
    seg_bonus = min(1.0, merged_segs / 8.0)

    score_merge = (
        0.30 * s_freq_merge
        + 0.15 * s_tier
        + 0.30 * ratio_bonus
        + 0.15 * seg_bonus
        + 0.10 * (1.0 - min(pos_confidences[candidate.start:candidate.end]))
    )

    # Score split
    s_min_freq_split = _freq_score(min_freq_fragment)
    avg_conf = sum(pos_confidences[candidate.start:candidate.end]) / n_merged

    score_split = (
        0.60 * s_min_freq_split
        + 0.20 * avg_conf
        + 0.20 * (1.0 if candidate.tier > 0 else 0.0)
    )

    return score_merge, score_split


# ── Orchestration de la fusion ───────────────────────────────


def merge_and_rescore(
    ipa_words: list[str],
    ortho_words: list[str],
    pos_confidences: list[float],
    lexicon: "PhoneLexicon",
    p2g: object | None = None,
    *,
    merge_margin: float = 0.05,
    max_merge: int = 4,
    confidence_threshold: float = 0.6,
) -> tuple[list[str], list[str], list[dict]]:
    """Orchestre la fusion de mots sur-segmentes.

    Returns:
        (new_ipa_words, new_ortho_words, merge_actions)
    """
    candidates = merge_candidates(
        ipa_words, ortho_words, pos_confidences, lexicon,
        max_merge=max_merge, confidence_threshold=confidence_threshold,
    )

    if not candidates:
        return list(ipa_words), list(ortho_words), []

    scored: list[tuple[MergeCandidate, float]] = []
    for cand in candidates:
        s_merge, s_split = score_merge_vs_split(
            cand, ipa_words, ortho_words, pos_confidences, lexicon,
        )
        delta = s_merge - s_split
        n_frags = cand.end - cand.start
        if n_frags <= 2:
            frag_freqs = [lexicon.best_freq(ipa_words[j]) for j in range(cand.start, cand.end)]
            frag_segs = [len(_ipa_grapheme_clusters(ipa_words[j])) for j in range(cand.start, cand.end)]
            content_freqs = [f for f, s in zip(frag_freqs, frag_segs) if s > 2]
            min_content_freq = min(content_freqs) if content_freqs else min(frag_freqs)
            effective_margin = 0.02 if min_content_freq < 1.0 else merge_margin
        else:
            effective_margin = 0.0
        if delta > effective_margin:
            scored.append((cand, delta))

    if not scored:
        return list(ipa_words), list(ortho_words), []

    # Greedy : trier par delta decroissant, appliquer sans chevauchement
    scored.sort(key=lambda x: -x[1])
    used: set[int] = set()
    accepted: list[MergeCandidate] = []
    for cand, _delta in scored:
        positions = set(range(cand.start, cand.end))
        if positions & used:
            continue
        used |= positions
        accepted.append(cand)

    if not accepted:
        return list(ipa_words), list(ortho_words), []

    accepted.sort(key=lambda c: c.start)

    # Construire les nouvelles listes
    new_ipa: list[str] = []
    new_ortho: list[str] = []
    merge_actions: list[dict] = []
    i = 0
    accept_idx = 0

    while i < len(ipa_words):
        if accept_idx < len(accepted) and i == accepted[accept_idx].start:
            cand = accepted[accept_idx]
            new_ipa.append(cand.resolved_ipa)
            new_ortho.append(cand.ortho)
            merge_actions.append({
                "action": "MERGE",
                "start": cand.start,
                "end": cand.end,
                "fragments_ipa": ipa_words[cand.start:cand.end],
                "fragments_ortho": ortho_words[cand.start:cand.end],
                "merged_ipa": cand.resolved_ipa,
                "merged_ortho": cand.ortho,
                "freq": cand.freq,
                "tier": cand.tier,
            })
            i = cand.end
            accept_idx += 1
        else:
            new_ipa.append(ipa_words[i])
            new_ortho.append(ortho_words[i])
            i += 1

    # Re-run P2G sur les mots fusionnes
    if p2g is not None and merge_actions:
        try:
            result_v2 = p2g.analyser_v2(new_ipa)
            if result_v2 and "ortho" in result_v2:
                new_ortho = list(result_v2["ortho"])
        except Exception:
            pass

    return new_ipa, new_ortho, merge_actions


# ── Elisions clitiques ──────────────────────────────────────

# Phones clitiques mono-phone pour la fusion d'elisions
_ELISION_CLITICS = {"l", "d", "s", "n", "ʒ", "k", "m", "t"}

# Prefixes elidables multi-phones (jusqu'a, lorsqu'il, puisqu'on...)
_MULTI_ELISION_PREFIXES = {
    "ʒysk",     # jusqu'
    "lɔʁsk",    # lorsqu'
    "pɥisk",    # puisqu'
    "kɛlk",     # quelqu'
    "kwak",     # quoiqu'  (rare)
}

# Voyelles IPA de base (premier codepoint) — pour detecter debut vocalique
_VOWEL_STARTS = set("aeiouyøœəɛɔɑ")


def try_elision_merges(
    ipa_words: list[str],
    ortho_words: list[str],
    lexicon: "PhoneLexicon",
) -> tuple[list[str], list[str], list[dict]]:
    """Fusionne les clitiques mono-phone elides avec le mot suivant.

    Returns:
        (new_ipa_words, new_ortho_words, elision_actions)
    """
    n = len(ipa_words)
    if n < 2:
        return list(ipa_words), list(ortho_words), []

    new_ipa: list[str] = []
    new_ortho: list[str] = []
    actions: list[dict] = []
    skip_next = False

    for i in range(n):
        if skip_next:
            skip_next = False
            continue

        if i + 1 < n:
            word_i = ipa_words[i]
            word_next = ipa_words[i + 1]
            segments_i = _ipa_grapheme_clusters(word_i)

            is_clitic = (
                len(segments_i) == 1
                and segments_i[0] in _ELISION_CLITICS
            )
            is_multi_elision = _normalize_ipa(word_i) in _MULTI_ELISION_PREFIXES
            next_starts_vowel = (
                len(word_next) > 0
                and word_next[0] in _VOWEL_STARTS
            )

            if (is_clitic or is_multi_elision) and next_starts_vowel:
                merged = word_i + word_next
                merged_n = _normalize_ipa(merged)
                freq_merged = lexicon.best_freq(merged_n)
                freq_next = lexicon.best_freq(_normalize_ipa(word_next))

                # Multi-phone : merge si le mot fusionne existe (jusqu+a → jusqu'a)
                # Mono-phone : merge si freq ratio > 2x (l+ami → l'ami)
                should_merge = (
                    (is_multi_elision and freq_merged > 0)
                    or (freq_merged > 0 and (freq_next <= 0
                                             or freq_merged > 2.0 * freq_next))
                )
                if should_merge:
                    ortho_merged = lexicon.best_ortho(merged_n)
                    if ortho_merged:
                        new_ipa.append(merged_n)
                        new_ortho.append(ortho_merged)
                        actions.append({
                            "action": "ELISION_MERGE",
                            "pos": i,
                            "clitic": word_i,
                            "next": word_next,
                            "merged_ipa": merged_n,
                            "merged_ortho": ortho_merged,
                            "freq_merged": freq_merged,
                            "freq_next": freq_next,
                        })
                        skip_next = True
                        continue

        new_ipa.append(ipa_words[i])
        new_ortho.append(ortho_words[i])

    return new_ipa, new_ortho, actions
