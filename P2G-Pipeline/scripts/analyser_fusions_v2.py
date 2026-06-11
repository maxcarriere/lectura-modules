#!/usr/bin/env python3
"""Analyse des fusions IPA v2 — filtre par compose interne.

Teste la strategie: fusionner des IPA consecutifs quand le resultat
matche un compose du lexique (tiret/apostrophe INTERNE dans l'ortho).

Usage:
    python scripts/analyser_fusions_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Graphemiseur" / "src"))

from lectura_graphemiseur._phone_lexicon import PhoneLexicon


def est_compose_interne(ortho: str) -> bool:
    """Verifie si l'ortho a un tiret ou apostrophe INTERNE (pas en pos 0/-1)."""
    if len(ortho) < 3:
        return False
    inner = ortho[1:-1]
    return "-" in inner or "'" in inner


def charger_dev_set() -> list[dict]:
    dev_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "Corpus" / "Kit-G2P-P2G" / "corpus" / "phrases" / "sentences_dev.json"
    )
    with open(dev_path, encoding="utf-8") as f:
        return json.load(f)


def charger_phone_lexicon() -> PhoneLexicon:
    db_path = (
        Path(__file__).resolve().parent.parent.parent
        / "Graphemiseur" / "src" / "lectura_graphemiseur" / "modeles" / "phone_lexicon.db"
    )
    return PhoneLexicon(db_path)


def analyser(sentences: list[dict], lex: PhoneLexicon, max_k: int = 5):
    """Analyse les fusions composees internes sur le dev set."""

    # Pre-calculer: quels phones fusionnes donnent un compose interne?
    # Utiliser phone_to_best pour un check O(1)

    vp_list = []
    fp_list = []
    n_phrases = 0
    n_mots = 0

    for sent in sentences:
        tokens = sent["tokens"]
        n = len(tokens)
        n_phrases += 1
        n_mots += n

        phones = [t.get("phone", "") or "" for t in tokens]
        gold_words = [t.get("form", "") for t in tokens]

        for k in range(min(max_k, n), 1, -1):
            for start in range(n - k + 1):
                span_phones = phones[start:start + k]
                if not all(span_phones):
                    continue

                # Strip apostrophe prefix from each phone
                clean_phones = []
                for p in span_phones:
                    if "'" in p:
                        p = p[p.index("'") + 1:]
                    clean_phones.append(p)

                fused = "".join(clean_phones)
                if not fused or not lex.exists(fused):
                    continue

                best_ortho = lex.best_ortho(fused)
                if not best_ortho:
                    continue

                # Filtre: compose interne
                if not est_compose_interne(best_ortho):
                    # Check also all entries (maybe best_ortho is not compound
                    # but another entry is)
                    entries = lex.all_entries(fused)
                    compound_entries = [
                        e for e in entries
                        if est_compose_interne(e.get("ortho", ""))
                    ]
                    if not compound_entries:
                        continue
                    # Use the best compound entry
                    best_entry = max(compound_entries, key=lambda e: e.get("freq", 0) or 0)
                    best_ortho = best_entry["ortho"]
                    best_freq = best_entry.get("freq", 0) or 0
                else:
                    best_freq = lex.best_freq(fused)

                # Check if fusion is correct vs gold
                gold_span = gold_words[start:start + k]
                gold_joined_clean = "".join(gold_span).lower().replace("-", "").replace("'", "").replace(" ", "")
                ortho_clean = best_ortho.lower().replace("-", "").replace("'", "").replace(" ", "")

                is_correct = (gold_joined_clean == ortho_clean)

                entry = (gold_span, best_ortho, span_phones, best_freq, k)
                if is_correct:
                    vp_list.append(entry)
                else:
                    fp_list.append(entry)

    total = len(vp_list) + len(fp_list)
    prec = len(vp_list) / total * 100 if total > 0 else 0

    print(f"\n{'='*80}")
    print(f"Strategie: compose interne (tiret/apostrophe)")
    print(f"{'='*80}")
    print(f"  Phrases: {n_phrases}, Mots: {n_mots}")
    print(f"  VP: {len(vp_list)}, FP: {len(fp_list)}, Total: {total}")
    print(f"  Precision: {prec:.1f}%")

    # Details VP
    print(f"\n  Vrais positifs ({len(vp_list)}):")
    from collections import Counter
    vp_by_ortho = Counter()
    for gold, ortho, phones, freq, k in vp_list:
        vp_by_ortho[ortho] += 1
    for ortho, count in vp_by_ortho.most_common(40):
        print(f"    {ortho:35s} : {count:3d} occ")

    # Details FP
    print(f"\n  Faux positifs ({len(fp_list)}):")
    fp_by_ortho = Counter()
    for gold, ortho, phones, freq, k in fp_list:
        fp_by_ortho[ortho] += 1
    for ortho, count in fp_by_ortho.most_common(40):
        print(f"    {ortho:35s} : {count:3d} occ")

    # Sample FP details
    print(f"\n  Echantillon FP details (max 50):")
    for gold, ortho, phones, freq, k in fp_list[:50]:
        print(f"    {' '.join(gold):35s} -> {ortho:25s} (phones: {'+'.join(phones)}, freq={freq:.1f})")

    # Analyse par k (nombre de mots fusionnes)
    print(f"\n  Distribution par k:")
    for kk in range(2, max_k + 1):
        vp_k = sum(1 for _, _, _, _, k in vp_list if k == kk)
        fp_k = sum(1 for _, _, _, _, k in fp_list if k == kk)
        total_k = vp_k + fp_k
        prec_k = vp_k / total_k * 100 if total_k > 0 else 0
        print(f"    k={kk}: VP={vp_k:4d}, FP={fp_k:4d}, prec={prec_k:.1f}%")

    # Test: que se passe-t-il si on exige freq >= X?
    print(f"\n  Precision par seuil de frequence:")
    for seuil in [0.0, 0.1, 1.0, 5.0, 10.0, 50.0]:
        vp_s = sum(1 for _, _, _, f, _ in vp_list if f >= seuil)
        fp_s = sum(1 for _, _, _, f, _ in fp_list if f >= seuil)
        total_s = vp_s + fp_s
        prec_s = vp_s / total_s * 100 if total_s > 0 else 0
        print(f"    freq >= {seuil:5.1f}: VP={vp_s:4d}, FP={fp_s:4d}, prec={prec_s:.1f}%")


def main():
    print("Chargement du dev set...")
    sentences = charger_dev_set()
    print(f"  {len(sentences)} phrases chargees")

    print("Chargement du phone_lexicon...")
    lex = charger_phone_lexicon()
    print(f"  {len(lex.phone_set)} phones charges")

    print("Analyse...")
    t0 = time.time()
    analyser(sentences, lex, max_k=5)
    dt = time.time() - t0
    print(f"\nTermine en {dt:.1f}s")


if __name__ == "__main__":
    main()
