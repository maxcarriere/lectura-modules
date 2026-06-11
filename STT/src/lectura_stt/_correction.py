"""Correction phonetique CTC par confiance et lexique.

Exploite la confiance CTC par phone pour identifier les mots douteux
(conf < seuil ET hors lexique phonetique) et tenter de les corriger
via des variantes (substitution softmax, deletion, insertion, merge, re-split).

Resultats experimentaux (500 phrases, seuil=0.98) :
  WER 19.94% -> 19.71% (-0.23%), precision 91.7% (11 TP / 1 FP)

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lectura_stt._lexicon import PhoneLexicon

from lectura_stt._lexicon import (
    _CONFUSION_MAP,
    _ipa_grapheme_clusters,
    SCHWA,
)
from lectura_stt._segmentation import _decompose_dp


# ==========================================================================
#  Groupement tokens CTC en mots avec stats
# ==========================================================================

def map_tokens_to_words(
    tokens: list[dict],
    vocab_inv: dict[int, str],
) -> list[dict]:
    """Groupe les tokens CTC en mots avec confiance ET alternatives par phone.

    Parameters
    ----------
    tokens : list[dict]
        Sortie de ``ctc_greedy_decode_with_alternatives``.
    vocab_inv : dict[int, str]
        Mapping ID -> phone IPA.

    Returns
    -------
    list[dict]
        Mots avec keys : ipa, phones, confidence, min_confidence,
        entropy, max_entropy, token_count, phone_confidences,
        phone_alternatives.
    """
    SEPS = {"|", ",", ".", "?", "!", "\u2026", "[-]", "[']"}
    LIAISON = {"[z]", "[t]", "[n]", "[\u0281]", "[p]"}

    words: list[dict] = []
    current_phones: list[str] = []
    current_confs: list[float] = []
    current_ents: list[float] = []
    current_alts: list[list[tuple[str, float]]] = []

    def flush() -> None:
        if current_phones and current_confs:
            words.append({
                "ipa": "".join(current_phones),
                "phones": list(current_phones),
                "confidence": sum(current_confs) / len(current_confs),
                "min_confidence": min(current_confs),
                "entropy": sum(current_ents) / len(current_ents),
                "max_entropy": max(current_ents),
                "token_count": len(current_phones),
                "phone_confidences": list(current_confs),
                "phone_alternatives": list(current_alts),
            })

    for tok in tokens:
        phone = vocab_inv.get(tok["phone_id"], "?")
        if phone in SEPS or phone in LIAISON:
            flush()
            current_phones = []
            current_confs = []
            current_ents = []
            current_alts = []
        else:
            current_phones.append(phone)
            current_confs.append(tok["confidence"])
            current_ents.append(tok["entropy"])
            alts = [(vocab_inv.get(pid, "?"), prob)
                    for pid, prob in tok.get("alternatives", [])]
            current_alts.append(alts)

    flush()
    return words


# ==========================================================================
#  Force du signal d'erreur
# ==========================================================================

# Mots fonctionnels tres frequents (1-2 phones) : une deletion
# qui produit un de ces mots est presque toujours un faux positif.
_SHORT_FUNCTION_WORDS = frozenset({
    "a", "e", "i", "o", "y", "u",
    "la", "le", "ly", "l\u00f8", "lo",
    "d\u0259", "du", "dy",
    "se", "sa", "s\u0259", "si", "so", "sy",
    "vu", "va", "vi",
    "m\u0259", "ma", "m\u0254\u0303",
    "t\u0259", "ta", "t\u0254\u0303",
    "n\u0259", "na", "nu", "ni",
    "\u0292\u0259", "\u0292e",
    "il", "\u025bl",
    "\u0251\u0303", "\u0254\u0303",
    "ki", "k\u0259",
    "pa",
})

# Seuils de force du signal par strategie
SIGNAL_THRESHOLDS: dict[str, float] = {
    "softmax_sub":     0.55,
    "softmax_double":  0.72,
    "insertion_schwa": 0.55,
    "deletion":        0.72,
    "merge":           0.55,
    "resplit":         0.80,
}


def _signal_strength(
    word_ipa: str,
    stats: dict,
    phone_lexicon: "PhoneLexicon",
) -> float:
    """Force du signal d'erreur pour un mot (0 = correct, 1 = errone)."""
    conf = stats["confidence"]
    min_conf = stats.get("min_confidence", conf)
    entropy = stats.get("entropy", 0.0)
    in_lex = word_ipa in phone_lexicon.phone_set

    s_conf = max(0.0, 1.0 - conf)
    s_min_conf = max(0.0, 1.0 - min_conf)
    s_entropy = min(1.0, entropy / 3.0)
    s_lex = 0.0 if in_lex else 1.0

    return 0.25 * s_conf + 0.20 * s_min_conf + 0.10 * s_entropy + 0.45 * s_lex


def _is_doubtful(
    word_ipa: str,
    confidence: float,
    phone_lexicon: "PhoneLexicon",
    conf_threshold: float,
) -> bool:
    """Un mot est douteux si conf < seuil ET hors lexique."""
    return confidence < conf_threshold and word_ipa not in phone_lexicon.phone_set


# ==========================================================================
#  Generation de candidats
# ==========================================================================

def _generate_softmax_variants(
    word_stats: dict,
    phone_lexicon: "PhoneLexicon",
    min_freq: float = 1.0,
    max_candidates: int = 50,
) -> list[dict]:
    """Genere des variantes via alternatives softmax CTC.

    Inclut substitutions simples/doubles, deletions et insertions schwa.
    """
    phones = word_stats["phones"]
    phone_confs = word_stats.get("phone_confidences", [1.0] * len(phones))
    phone_alts = word_stats.get("phone_alternatives", [[] for _ in phones])
    word_ipa = word_stats["ipa"]
    n = len(phones)
    if n == 0:
        return []

    pos_by_conf = sorted(range(n),
                         key=lambda i: phone_confs[i] if i < len(phone_confs) else 1.0)

    candidates: list[dict] = []
    seen: set[str] = set()

    # --- Substitution simple (1 phone) ---
    for pos in pos_by_conf[:min(n, 5)]:
        alts = list(phone_alts[pos]) if pos < len(phone_alts) else []
        base_phone = phones[pos]
        for conf_phone in _CONFUSION_MAP.get(base_phone, set()):
            if conf_phone not in {a for a, _ in alts}:
                alts.append((conf_phone, 0.01))

        for alt_phone, alt_prob in alts:
            if alt_prob < 0.03:
                continue
            new_phones = list(phones)
            new_phones[pos] = alt_phone
            variant = "".join(new_phones)
            if variant == word_ipa or variant in seen:
                continue
            seen.add(variant)
            if variant in phone_lexicon.phone_set:
                freq = phone_lexicon.best_freq(variant)
                if freq >= min_freq:
                    candidates.append({
                        "variant_ipa": variant,
                        "freq": freq,
                        "desc": f"softmax {base_phone}->{alt_phone} @{pos} (p={alt_prob:.2f})",
                        "strategy": "softmax_sub",
                        "n_changes": 1,
                        "alt_prob": alt_prob,
                        "is_word_level": True,
                    })

    # --- Substitution double (2 phones) ---
    if n >= 4:
        top_positions = pos_by_conf[:min(n, 4)]
        for pi in range(len(top_positions)):
            for pj in range(pi + 1, len(top_positions)):
                p1, p2 = top_positions[pi], top_positions[pj]
                alts1 = list(phone_alts[p1]) if p1 < len(phone_alts) else []
                alts2 = list(phone_alts[p2]) if p2 < len(phone_alts) else []
                base1, base2 = phones[p1], phones[p2]
                alts1_ext = alts1 + [(c, 0.01) for c in _CONFUSION_MAP.get(base1, set())
                                     if c not in {a for a, _ in alts1}]
                alts2_ext = alts2 + [(c, 0.01) for c in _CONFUSION_MAP.get(base2, set())
                                     if c not in {a for a, _ in alts2}]
                for a1, prob1 in alts1_ext[:4]:
                    for a2, prob2 in alts2_ext[:4]:
                        if max(prob1, prob2) < 0.05:
                            continue
                        new_phones = list(phones)
                        new_phones[p1] = a1
                        new_phones[p2] = a2
                        variant = "".join(new_phones)
                        if variant == word_ipa or variant in seen:
                            continue
                        seen.add(variant)
                        if variant in phone_lexicon.phone_set:
                            freq = phone_lexicon.best_freq(variant)
                            if freq >= min_freq:
                                candidates.append({
                                    "variant_ipa": variant,
                                    "freq": freq,
                                    "desc": f"double {base1}->{a1}@{p1} + {base2}->{a2}@{p2}",
                                    "strategy": "softmax_double",
                                    "n_changes": 2,
                                    "alt_prob": min(prob1, prob2),
                                    "is_word_level": True,
                                })
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

    # --- Deletion de phone ---
    if n > 1:
        for pos in pos_by_conf[:min(n, 3)]:
            new_phones = phones[:pos] + phones[pos + 1:]
            variant = "".join(new_phones)
            if not variant or variant == word_ipa or variant in seen:
                continue
            seen.add(variant)
            if len(new_phones) <= 2 and variant in _SHORT_FUNCTION_WORDS:
                continue
            if variant in phone_lexicon.phone_set:
                freq = phone_lexicon.best_freq(variant)
                if freq >= min_freq:
                    candidates.append({
                        "variant_ipa": variant,
                        "freq": freq,
                        "desc": f"del {phones[pos]} @{pos}",
                        "strategy": "deletion",
                        "n_changes": 1,
                        "alt_prob": 0.0,
                        "is_word_level": True,
                    })

    # --- Insertion schwa ---
    for pos in range(n + 1):
        new_phones = list(phones[:pos]) + [SCHWA] + list(phones[pos:])
        variant = "".join(new_phones)
        if variant in seen:
            continue
        seen.add(variant)
        if variant in phone_lexicon.phone_set:
            freq = phone_lexicon.best_freq(variant)
            if freq >= min_freq:
                candidates.append({
                    "variant_ipa": variant,
                    "freq": freq,
                    "desc": f"ins schwa @{pos}",
                    "strategy": "insertion_schwa",
                    "n_changes": 1,
                    "alt_prob": 0.0,
                    "is_word_level": True,
                })

    return candidates[:max_candidates]


def _generate_merge_variants(
    idx: int,
    ipa_words: list[str],
    word_stats_list: list[dict],
    phone_lexicon: "PhoneLexicon",
    min_freq: float = 1.0,
) -> list[dict]:
    """Genere des variantes de merge avec les mots voisins et re-split."""
    candidates: list[dict] = []
    n = len(ipa_words)
    word_ipa = ipa_words[idx]

    # Merge avec precedent
    if idx > 0:
        merged = ipa_words[idx - 1] + ipa_words[idx]
        if merged in phone_lexicon.phone_set:
            freq_merged = phone_lexicon.best_freq(merged)
            freq_prev = phone_lexicon.best_freq(ipa_words[idx - 1])
            freq_curr = phone_lexicon.best_freq(ipa_words[idx])
            freq_parts = max(freq_prev, 0.01) * max(freq_curr, 0.01)
            if freq_merged >= min_freq and freq_merged > freq_parts * 2:
                candidates.append({
                    "strategy": "merge_prev",
                    "start": idx - 1, "end": idx + 1,
                    "new_words": [merged],
                    "freq": freq_merged,
                    "desc": f"merge '{ipa_words[idx-1]}'+'{ipa_words[idx]}'",
                    "is_word_level": False,
                })

    # Merge avec suivant
    if idx < n - 1:
        merged = ipa_words[idx] + ipa_words[idx + 1]
        if merged in phone_lexicon.phone_set:
            freq_merged = phone_lexicon.best_freq(merged)
            freq_curr = phone_lexicon.best_freq(ipa_words[idx])
            freq_next = phone_lexicon.best_freq(ipa_words[idx + 1])
            freq_parts = max(freq_curr, 0.01) * max(freq_next, 0.01)
            if freq_merged >= min_freq and freq_merged > freq_parts * 2:
                candidates.append({
                    "strategy": "merge_next",
                    "start": idx, "end": idx + 2,
                    "new_words": [merged],
                    "freq": freq_merged,
                    "desc": f"merge '{ipa_words[idx]}'+'{ipa_words[idx+1]}'",
                    "is_word_level": False,
                })

    # Re-split
    segments = _ipa_grapheme_clusters(word_ipa)
    if len(segments) >= 5:
        decompositions = _decompose_dp(segments, phone_lexicon)
        for words, score in decompositions[:3]:
            if len(words) != 2:
                continue
            if not all(len(_ipa_grapheme_clusters(w)) >= 2 for w in words):
                continue
            resplit_min = max(min_freq, 10.0)
            min_freq_parts = min(phone_lexicon.best_freq(w) for w in words)
            if min_freq_parts >= resplit_min:
                candidates.append({
                    "strategy": "resplit",
                    "new_words": words,
                    "freq": min_freq_parts,
                    "desc": f"resplit -> {' + '.join(words)}",
                    "is_word_level": False,
                })

    return candidates


# ==========================================================================
#  Scoring
# ==========================================================================

def _score_lm_context(
    candidate_ipa: str,
    phone_lexicon: "PhoneLexicon",
    prev_ortho: str | None,
    next_ortho: str | None,
    trigram_conn: object | None,
) -> float:
    """Score un candidat IPA dans son contexte via le LM trigramme."""
    if trigram_conn is None:
        return 0.0

    entries = phone_lexicon.all_entries(candidate_ipa)
    if not entries:
        ortho = phone_lexicon.best_ortho(candidate_ipa)
        if ortho:
            entries = [{"ortho": ortho}]
        else:
            return 0.0

    cur = trigram_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bigrams_right'")
    has_bigrams = cur.fetchone() is not None

    best_score = 0.0
    prev_w = prev_ortho.lower() if prev_ortho else None
    next_w = next_ortho.lower() if next_ortho else None

    for entry in entries:
        ortho_l = entry["ortho"].lower()
        score = 0.0

        if prev_w and next_w:
            cur.execute(
                "SELECT count FROM trigrams WHERE w_prev=? AND w_target=? AND w_next=?",
                (prev_w, ortho_l, next_w))
            row = cur.fetchone()
            if row:
                score += row[0] * 3

        if has_bigrams:
            if prev_w:
                cur.execute(
                    "SELECT total FROM bigrams_left WHERE w_prev=? AND w_target=?",
                    (prev_w, ortho_l))
                row = cur.fetchone()
                if row:
                    score += row[0]
            if next_w:
                cur.execute(
                    "SELECT total FROM bigrams_right WHERE w_target=? AND w_next=?",
                    (ortho_l, next_w))
                row = cur.fetchone()
                if row:
                    score += row[0]
        else:
            if prev_w:
                cur.execute(
                    "SELECT SUM(count) FROM trigrams WHERE w_prev=? AND w_target=?",
                    (prev_w, ortho_l))
                row = cur.fetchone()
                if row and row[0]:
                    score += row[0]
            if next_w:
                cur.execute(
                    "SELECT SUM(count) FROM trigrams WHERE w_target=? AND w_next=?",
                    (ortho_l, next_w))
                row = cur.fetchone()
                if row and row[0]:
                    score += row[0]

        best_score = max(best_score, score)

    return best_score


def _candidate_score(
    candidate: dict,
    word_ipa: str,
    phone_lexicon: "PhoneLexicon",
) -> float:
    """Score composite d'un candidat (freq + softmax + LM + strategie)."""
    freq = candidate["freq"]
    strategy = candidate["strategy"]

    s_freq = math.log10(max(freq, 1.0))

    alt_prob = candidate.get("alt_prob", 0.0)
    s_softmax = math.log10(max(alt_prob, 1e-6)) + 6
    s_softmax = max(0.0, s_softmax) * 0.3

    lm_score = candidate.get("lm_score", 0.0)
    lm_orig = candidate.get("lm_score_orig", 0.0)
    if lm_orig > 0 and lm_score > lm_orig:
        s_lm = min(math.log10(max(lm_score - lm_orig, 1.0)), 3.0) * 0.5
    elif lm_orig > 0 and lm_score < lm_orig:
        s_lm = -2.0
    else:
        s_lm = 0.0

    strategy_bonus = {
        "softmax_sub": 1.0,
        "softmax_double": -0.5,
        "insertion_schwa": 0.3,
        "deletion": -0.5,
        "merge_prev": 0.3,
        "merge_next": 0.3,
        "resplit": -1.0,
    }
    s_strat = strategy_bonus.get(strategy, 0.0)

    if candidate.get("is_word_level") and "variant_ipa" in candidate:
        variant = candidate["variant_ipa"]
        orig_segs = len(_ipa_grapheme_clusters(word_ipa))
        var_segs = len(_ipa_grapheme_clusters(variant))
        s_prox = min(orig_segs, var_segs) / max(orig_segs, var_segs, 1)
    else:
        s_prox = 0.5

    return s_freq + s_softmax + s_lm + s_strat + s_prox


# ==========================================================================
#  Alignement stats -> mots
# ==========================================================================

def _align_stats_to_words(
    ipa_words: list[str],
    word_stats: list[dict],
) -> dict[int, dict]:
    """Aligne les word_stats CTC avec les ipa_words post-preprocessing."""
    result: dict[int, dict] = {}
    used: set[int] = set()

    # Pass 1 : match exact
    for i, word in enumerate(ipa_words):
        for j, ws in enumerate(word_stats):
            if j in used:
                continue
            if ws["ipa"] == word:
                result[i] = ws
                used.add(j)
                break

    # Pass 2 : match par sous-chaine
    for i, word in enumerate(ipa_words):
        if i in result:
            continue
        for j, ws in enumerate(word_stats):
            if j in used:
                continue
            if word in ws["ipa"] or ws["ipa"] in word:
                result[i] = ws
                used.add(j)
                break

    # Pass 3 : match par position
    unmatched_w = [i for i in range(len(ipa_words)) if i not in result]
    unmatched_s = [j for j in range(len(word_stats)) if j not in used]
    for i, j in zip(unmatched_w, unmatched_s):
        result[i] = word_stats[j]

    return result


# ==========================================================================
#  Point d'entree principal
# ==========================================================================

def correct_doubtful_words(
    ipa_words: list[str],
    word_stats: list[dict],
    phone_lexicon: "PhoneLexicon",
    *,
    conf_threshold: float = 0.98,
    min_freq: float = 1.0,
    trigram_conn: object | None = None,
    ortho_context: list[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """Corrige les mots douteux dans la sequence IPA.

    Parameters
    ----------
    ipa_words : list[str]
        Mots IPA post-preprocessing.
    word_stats : list[dict]
        Sortie de ``map_tokens_to_words``.
    phone_lexicon : PhoneLexicon
        Lexique phonemique.
    conf_threshold : float
        Seuil de confiance pour flagger un mot (defaut 0.98).
    min_freq : float
        Frequence minimale pour accepter une correction.
    trigram_conn
        Connexion sqlite3 du LM trigramme (ou None).
    ortho_context : list[str] | None
        Mots ortho baseline pour contexte LM.

    Returns
    -------
    tuple[list[str], list[dict]]
        (mots IPA corriges, log des corrections).
    """
    word_conf_map = _align_stats_to_words(ipa_words, word_stats)

    result = list(ipa_words)
    corrections: list[dict] = []
    skip_indices: set[int] = set()

    for i in range(len(ipa_words)):
        if i in skip_indices:
            continue

        word = ipa_words[i]
        stats = word_conf_map.get(i)
        if stats is None:
            continue

        confidence = stats["confidence"]
        if not _is_doubtful(word, confidence, phone_lexicon, conf_threshold):
            continue

        signal = _signal_strength(word, stats, phone_lexicon)

        all_candidates: list[dict] = []

        if signal >= SIGNAL_THRESHOLDS["softmax_sub"]:
            variants = _generate_softmax_variants(stats, phone_lexicon, min_freq)
            for v in variants:
                strat = v["strategy"]
                if strat == "softmax_double" and signal < SIGNAL_THRESHOLDS["softmax_double"]:
                    continue
                if strat == "deletion" and signal < SIGNAL_THRESHOLDS["deletion"]:
                    continue
                if strat == "insertion_schwa" and signal < SIGNAL_THRESHOLDS["insertion_schwa"]:
                    continue
                all_candidates.append(v)

        if signal >= SIGNAL_THRESHOLDS["merge"]:
            merges = _generate_merge_variants(
                i, ipa_words, word_stats, phone_lexicon, min_freq,
            )
            for m in merges:
                if m["strategy"] == "resplit":
                    if signal < SIGNAL_THRESHOLDS["resplit"]:
                        continue
                    if any(c.get("is_word_level") for c in all_candidates):
                        continue
                all_candidates.append(m)

        if not all_candidates:
            continue

        # Scoring LM
        if trigram_conn is not None and ortho_context is not None:
            prev_o = ortho_context[i - 1] if i > 0 and i - 1 < len(ortho_context) else None
            next_o = ortho_context[i + 1] if i + 1 < len(ortho_context) else None
            lm_orig = _score_lm_context(
                word, phone_lexicon, prev_o, next_o, trigram_conn,
            )
            for c in all_candidates:
                if c.get("is_word_level") and "variant_ipa" in c:
                    c["lm_score"] = _score_lm_context(
                        c["variant_ipa"], phone_lexicon, prev_o, next_o, trigram_conn,
                    )
                    c["lm_score_orig"] = lm_orig

        best = max(all_candidates, key=lambda c: _candidate_score(c, word, phone_lexicon))

        if best.get("is_word_level"):
            old_word = result[i]
            result[i] = best["variant_ipa"]
            corrections.append({
                "pos": i,
                "old_ipa": old_word,
                "new_ipa": best["variant_ipa"],
                "old_ortho": phone_lexicon.best_ortho(old_word),
                "new_ortho": phone_lexicon.best_ortho(best["variant_ipa"]),
                "strategy": best["strategy"],
                "desc": best["desc"],
                "freq": best["freq"],
                "confidence": confidence,
                "signal": signal,
            })
        else:
            if best["strategy"].startswith("merge"):
                start = best["start"]
                end = best["end"]
                new_words = best["new_words"]
                old_words = result[start:end]
                result[start:end] = new_words

                for j in range(start, min(start + len(new_words), len(result))):
                    if j != i:
                        skip_indices.add(j)
                size_diff = len(new_words) - (end - start)
                if size_diff != 0:
                    skip_indices = {(s + size_diff if s >= end else s) for s in skip_indices}

                corrections.append({
                    "pos": start,
                    "old_ipa": old_words,
                    "new_ipa": new_words,
                    "old_ortho": [phone_lexicon.best_ortho(w) for w in old_words],
                    "new_ortho": [phone_lexicon.best_ortho(w) for w in new_words],
                    "strategy": best["strategy"],
                    "desc": best["desc"],
                    "freq": best["freq"],
                    "confidence": confidence,
                    "signal": signal,
                })
            elif best["strategy"] == "resplit":
                new_words = best["new_words"]
                old_word = result[i]
                result[i:i + 1] = new_words
                size_diff = len(new_words) - 1
                if size_diff != 0:
                    skip_indices = {(s + size_diff if s > i else s) for s in skip_indices}
                for j in range(i + 1, i + len(new_words)):
                    skip_indices.add(j)

                corrections.append({
                    "pos": i,
                    "old_ipa": old_word,
                    "new_ipa": new_words,
                    "old_ortho": phone_lexicon.best_ortho(old_word),
                    "new_ortho": [phone_lexicon.best_ortho(w) for w in new_words],
                    "strategy": "resplit",
                    "desc": best["desc"],
                    "freq": best["freq"],
                    "confidence": confidence,
                    "signal": signal,
                })

    return result, corrections
