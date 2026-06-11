#!/usr/bin/env python3
"""Benchmark POS/MORPHO du P2G sur le corpus UD French-GSD (1712 phrases).

Utilise les annotations gold UD comme reference et les phones du corpus
Kit-G2P-P2G. Deux scenarios :
  1. Phones gold (du corpus) → P2G sans ortho_words → eval POS/MORPHO
  2. G2P(forme) → P2G sans ortho_words → eval POS/MORPHO
     (simule le roundtrip correcteur)

Usage:
    python scripts/bench_p2g_morpho_ud.py
    python scripts/bench_p2g_morpho_ud.py --max-phrases 200  # rapide
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

logging.disable(logging.WARNING)

CORPUS_PATH = "/data/work/projets/lectura/workspace/Corpus/Kit-G2P-P2G/corpus/phrases/sentences_test.json"

FEATURES = ["pos", "Number", "Gender", "Person", "VerbForm", "Mood"]

# POS equivalences (P2G peut predire des variantes)
_POS_EQUIV: dict[str, set[str]] = {
    "ART:def": {"ART:def", "DET"},
    "ART:ind": {"ART:ind", "DET"},
    "ART:dem": {"ART:dem", "DET"},
    "ART:pos": {"ART:pos", "DET", "ADJ:pos"},
    "PRO:per": {"PRO:per", "PRO", "PRON", "PRO:ind"},
    "PRO:dem": {"PRO:dem", "PRO"},
    "PRO:rel": {"PRO:rel", "PRO"},
    "PRO:ind": {"PRO:ind", "PRO", "PRO:per"},
    "AUX": {"AUX", "VER"},
    "VER": {"VER", "AUX"},
    "CON": {"CON", "CCONJ", "KON"},
    "PRE": {"PRE", "PRP", "ADP"},
    "ADJ:pos": {"ADJ:pos", "ART:pos", "DET"},
    "NOM:pro": {"NOM:pro", "NOM", "PROPN"},
}


def _match(feat: str, predicted: str, expected: str) -> bool:
    if predicted == expected:
        return True
    if feat == "pos":
        return predicted in _POS_EQUIV.get(expected, {expected})
    return False


@dataclass
class FeatureStats:
    total: int = 0
    correct: int = 0
    errors: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def load_corpus(path: str, max_phrases: int = 0) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if max_phrases > 0:
        data = data[:max_phrases]
    return data


def run_benchmark(
    corpus: list[dict],
    p2g_adapter,
    scenario: str,
    g2p_tagger=None,
) -> dict[str, FeatureStats]:
    """Evalue le P2G sur le corpus UD.

    scenario: "gold_phones" ou "g2p_roundtrip"
    """
    stats_by_feat: dict[str, FeatureStats] = {f: FeatureStats() for f in FEATURES}
    stats_by_pos: dict[str, dict[str, FeatureStats]] = defaultdict(
        lambda: {f: FeatureStats() for f in FEATURES}
    )
    n_skipped = 0

    for entry in corpus:
        tokens = entry.get("tokens", [])
        if not tokens:
            continue

        # Extraire phones et formes
        forms = [t["form"] for t in tokens]
        ref_pos = [t["pos_tag"] for t in tokens]
        ref_morpho = [t.get("morpho", {}) for t in tokens]

        if scenario == "gold_phones":
            ipa_words = [t.get("phone", "") for t in tokens]
        else:
            # G2P roundtrip : phonemiser les formes
            tags = g2p_tagger.tag_words_rich(forms)
            ipa_words = []
            for i, t in enumerate(tags):
                phone = t.get("g2p", "")
                if not phone and hasattr(g2p_tagger, "prononcer"):
                    phone = g2p_tagger.prononcer(forms[i]) or ""
                ipa_words.append(phone or forms[i])

        # Filtrer les tokens sans phone
        valid = [bool(ip.strip()) for ip in ipa_words]
        if not any(valid):
            n_skipped += 1
            continue

        # P2G sans ortho_words
        try:
            result = p2g_adapter.transcrire_complet(ipa_words, ortho_words=None, k=1)
        except Exception:
            n_skipped += 1
            continue

        p2g_pos = result.get("pos", [])
        p2g_morpho = result.get("morpho", {})

        # Evaluer chaque token
        for i, tok in enumerate(tokens):
            if not valid[i]:
                continue

            gold_pos = ref_pos[i]
            gold_m = ref_morpho[i]

            # POS
            predicted_pos = p2g_pos[i] if i < len(p2g_pos) else ""
            stats_by_feat["pos"].total += 1
            stats_by_pos[gold_pos]["pos"].total += 1
            if _match("pos", predicted_pos, gold_pos):
                stats_by_feat["pos"].correct += 1
                stats_by_pos[gold_pos]["pos"].correct += 1
            else:
                stats_by_feat["pos"].errors.append({
                    "form": forms[i], "ipa": ipa_words[i],
                    "predicted": predicted_pos, "expected": gold_pos,
                })
                stats_by_pos[gold_pos]["pos"].errors.append({
                    "form": forms[i], "predicted": predicted_pos,
                })

            # Morpho features
            for feat in FEATURES[1:]:  # skip "pos"
                gold_val = gold_m.get(feat, "_")
                if gold_val == "_":
                    continue  # pas annote → on skip

                feat_list = p2g_morpho.get(feat, [])
                predicted_val = feat_list[i] if i < len(feat_list) else "_"

                stats_by_feat[feat].total += 1
                stats_by_pos[gold_pos][feat].total += 1
                if _match(feat, predicted_val, gold_val):
                    stats_by_feat[feat].correct += 1
                    stats_by_pos[gold_pos][feat].correct += 1
                else:
                    stats_by_feat[feat].errors.append({
                        "form": forms[i], "ipa": ipa_words[i],
                        "feat": feat, "predicted": predicted_val,
                        "expected": gold_val, "gold_pos": gold_pos,
                    })
                    stats_by_pos[gold_pos][feat].errors.append({
                        "form": forms[i], "predicted": predicted_val,
                        "expected": gold_val,
                    })

    return stats_by_feat, stats_by_pos, n_skipped


def afficher(label, stats_by_feat, stats_by_pos, n_skipped, elapsed):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    total_all = sum(s.total for s in stats_by_feat.values())
    correct_all = sum(s.correct for s in stats_by_feat.values())
    print(f"\n  GLOBAL: {correct_all}/{total_all} = {correct_all/total_all:.1%}  "
          f"({n_skipped} phrases skipped, {elapsed:.1f}s)")

    print(f"\n  {'Feature':12s}  {'OK':>6s} / {'N':>6s}  {'Accuracy':>8s}  {'Erreurs':>7s}")
    print(f"  {'-'*12}  {'-'*6}   {'-'*6}  {'-'*8}  {'-'*7}")
    for f in FEATURES:
        s = stats_by_feat[f]
        if s.total > 0:
            print(f"  {f:12s}  {s.correct:>6d} / {s.total:>6d}  {s.accuracy:>8.1%}  {len(s.errors):>7d}")

    # Par POS (top erreurs)
    print(f"\n  Accuracy par POS (features morpho, top erreurs):")
    pos_summary = []
    for pos, feat_stats in stats_by_pos.items():
        total = sum(feat_stats[f].total for f in FEATURES[1:])
        correct = sum(feat_stats[f].correct for f in FEATURES[1:])
        if total > 0:
            pos_summary.append((pos, correct, total, correct / total))
    pos_summary.sort(key=lambda x: x[2], reverse=True)
    print(f"  {'POS':12s}  {'OK':>6s} / {'N':>6s}  {'Accuracy':>8s}")
    print(f"  {'-'*12}  {'-'*6}   {'-'*6}  {'-'*8}")
    for pos, correct, total, acc in pos_summary[:15]:
        marker = " !" if acc < 0.85 else ""
        print(f"  {pos:12s}  {correct:>6d} / {total:>6d}  {acc:>8.1%}{marker}")

    # Top confusions POS
    print(f"\n  Top confusions POS:")
    from collections import Counter
    confusions = Counter()
    for e in stats_by_feat["pos"].errors:
        confusions[(e["expected"], e["predicted"])] += 1
    for (exp, pred), count in confusions.most_common(15):
        print(f"    {exp:12s} → {pred:12s}  x{count}")

    # Exemples d'erreurs morpho par feature
    for feat in FEATURES[1:]:
        errs = stats_by_feat[feat].errors
        if not errs:
            continue
        # Grouper par (expected, predicted)
        groups = Counter()
        for e in errs:
            groups[(e["expected"], e["predicted"], e.get("gold_pos", "?"))] += 1
        top = groups.most_common(5)
        print(f"\n  Top erreurs {feat}:")
        for (exp, pred, gpos), cnt in top:
            ex = next(e["form"] for e in errs
                      if e["expected"] == exp and e["predicted"] == pred)
            print(f"    {exp:8s} → {pred:8s} ({gpos:8s}) x{cnt:4d}  ex: {ex}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-phrases", type=int, default=0)
    args = parser.parse_args()

    corpus = load_corpus(CORPUS_PATH, args.max_phrases)
    n = len(corpus)
    n_tokens = sum(len(e.get("tokens", [])) for e in corpus)
    print(f"Corpus: {n} phrases, {n_tokens} tokens")

    from lectura_correcteur._adapter_p2g import creer_adapter_p2g
    from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
    from lectura_correcteur._tagger_hybride import TaggerHybride
    from lectura_correcteur._utils import LexiqueNormalise
    from lectura_lexique import Lexique

    LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"
    lexique = Lexique(LEXIQUE_DB)
    lex_n = LexiqueNormalise(lexique)
    g2p = creer_adapter_g2p_unifie()
    g2p_tagger = TaggerHybride(g2p, lex_n, lm_homophones=None)
    p2g = creer_adapter_p2g()

    if p2g is None:
        print("ERREUR: P2G indisponible")
        return

    # Scenario 1 : phones gold
    t0 = time.time()
    s1_feat, s1_pos, s1_skip = run_benchmark(corpus, p2g, "gold_phones")
    t1 = time.time() - t0
    afficher(f"Scenario 1 — Phones gold → P2G ({n} phrases)", s1_feat, s1_pos, s1_skip, t1)

    # Scenario 2 : G2P roundtrip
    t0 = time.time()
    s2_feat, s2_pos, s2_skip = run_benchmark(corpus, p2g, "g2p_roundtrip", g2p_tagger)
    t2 = time.time() - t0
    afficher(f"Scenario 2 — G2P roundtrip → P2G ({n} phrases)", s2_feat, s2_pos, s2_skip, t2)

    # Comparatif
    print(f"\n\n{'='*80}")
    print(f"  COMPARATIF")
    print(f"{'='*80}")
    print(f"  {'Feature':12s}  {'Gold phones':>12s}  {'G2P roundtrip':>13s}  {'Delta':>8s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*13}  {'-'*8}")
    for f in FEATURES:
        a1 = s1_feat[f].accuracy if s1_feat[f].total > 0 else 0
        a2 = s2_feat[f].accuracy if s2_feat[f].total > 0 else 0
        if s1_feat[f].total > 0:
            print(f"  {f:12s}  {a1:>12.1%}  {a2:>13.1%}  {a2-a1:>+8.1%}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
