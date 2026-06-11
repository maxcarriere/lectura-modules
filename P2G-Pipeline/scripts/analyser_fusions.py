#!/usr/bin/env python3
"""Analyse des fusions IPA sur le dev set P2G.

Teste differentes strategies de filtrage pour determiner quand
fusionner des mots IPA consecutifs et utiliser le resultat du lexique.

Usage:
    python scripts/analyser_fusions.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ajouter le graphemiseur au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Graphemiseur" / "src"))

from lectura_graphemiseur._phone_lexicon import PhoneLexicon


def charger_dev_set() -> list[dict]:
    """Charge le dev set."""
    dev_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "Corpus" / "Kit-G2P-P2G" / "corpus" / "phrases" / "sentences_dev.json"
    )
    with open(dev_path, encoding="utf-8") as f:
        return json.load(f)


def charger_phone_lexicon() -> PhoneLexicon:
    """Charge le phone_lexicon."""
    db_path = (
        Path(__file__).resolve().parent.parent.parent
        / "Graphemiseur" / "src" / "lectura_graphemiseur" / "modeles" / "phone_lexicon.db"
    )
    return PhoneLexicon(db_path)


def est_oov(phone: str, lex: PhoneLexicon) -> bool:
    """Un phone est OOV s'il n'a aucune entree dans le lexique commun."""
    if not phone:
        return True
    entries = lex.all_entries(phone)
    if not entries:
        return True
    # OOV si seulement des NOM PROPRE
    has_commun = any(
        (e.get("cgram") or "").strip() != "NOM PROPRE"
        for e in entries
    )
    return not has_commun


def est_rare(phone: str, lex: PhoneLexicon, seuil: float = 1.0) -> bool:
    """Un phone est rare si sa meilleure freq est sous le seuil."""
    freq = lex.best_freq(phone)
    return freq < seuil


def analyser_fusions(
    sentences: list[dict],
    lex: PhoneLexicon,
    max_k: int = 5,
) -> None:
    """Analyse toutes les fusions possibles et evalue les strategies."""

    # Strategies de filtrage
    strategies = {
        "naive": lambda phones, fused_phone, fused_entries, lex, gold_words, start: True,
        "tous_oov": lambda phones, fused_phone, fused_entries, lex, gold_words, start: all(
            est_oov(p, lex) for p in phones
        ),
        "au_moins_1_oov": lambda phones, fused_phone, fused_entries, lex, gold_words, start: any(
            est_oov(p, lex) for p in phones
        ),
        "fused_has_special": lambda phones, fused_phone, fused_entries, lex, gold_words, start: any(
            ("'" in (e.get("ortho") or "") or "-" in (e.get("ortho") or ""))
            for e in fused_entries
        ),
        "tous_oov+special": lambda phones, fused_phone, fused_entries, lex, gold_words, start: (
            all(est_oov(p, lex) for p in phones)
            and any(
                ("'" in (e.get("ortho") or "") or "-" in (e.get("ortho") or ""))
                for e in fused_entries
            )
        ),
        "1oov+special": lambda phones, fused_phone, fused_entries, lex, gold_words, start: (
            any(est_oov(p, lex) for p in phones)
            and any(
                ("'" in (e.get("ortho") or "") or "-" in (e.get("ortho") or ""))
                for e in fused_entries
            )
        ),
        "freq_ratio": lambda phones, fused_phone, fused_entries, lex, gold_words, start: (
            max((e.get("freq") or 0) for e in fused_entries) > 10 * max(
                (lex.best_freq(p) for p in phones), default=0
            )
        ),
        "tous_oov+freq5": lambda phones, fused_phone, fused_entries, lex, gold_words, start: (
            all(est_oov(p, lex) for p in phones)
            and max((e.get("freq") or 0) for e in fused_entries) >= 5.0
        ),
        "1oov+freq10": lambda phones, fused_phone, fused_entries, lex, gold_words, start: (
            any(est_oov(p, lex) for p in phones)
            and max((e.get("freq") or 0) for e in fused_entries) >= 10.0
        ),
        "tous_rare+special": lambda phones, fused_phone, fused_entries, lex, gold_words, start: (
            all(est_rare(p, lex, 1.0) for p in phones)
            and any(
                ("'" in (e.get("ortho") or "") or "-" in (e.get("ortho") or ""))
                for e in fused_entries
            )
        ),
    }

    # Compteurs par strategie
    results = {name: {"vp": 0, "fp": 0, "matches": [], "errors": []}
               for name in strategies}

    n_phrases = 0
    n_mots = 0

    for sent in sentences:
        tokens = sent["tokens"]
        n = len(tokens)
        n_phrases += 1
        n_mots += n

        phones = [t.get("phone", "") or "" for t in tokens]
        gold_words = [t.get("form", "") for t in tokens]

        # Tester les fusions de longueur k (2 a max_k)
        for k in range(2, min(max_k + 1, n + 1)):
            for start in range(n - k + 1):
                span_phones = phones[start:start + k]
                if not all(span_phones):
                    continue

                fused_phone = "".join(span_phones)
                fused_entries = lex.all_entries(fused_phone)
                if not fused_entries:
                    continue

                # Determiner le gold : est-ce que la fusion est correcte?
                # La fusion est correcte si le gold correspond au best_ortho
                # du phone fusionne
                gold_span = gold_words[start:start + k]
                gold_joined = "".join(gold_span).lower()
                # Variantes : avec tiret, apostrophe
                gold_joined_sp = " ".join(gold_span).lower()
                gold_joined_tiret = "-".join(gold_span).lower()
                gold_joined_apo = "'".join(gold_span).lower()

                # Verifier si une entree du lexique fusionne correspond
                fused_orthos = set()
                for e in fused_entries:
                    ortho = (e.get("ortho") or "").lower()
                    fused_orthos.add(ortho)
                    # Normaliser : retirer tirets et apostrophes pour comparer
                    ortho_clean = ortho.replace("-", "").replace("'", "").replace(" ", "")
                    fused_orthos.add(ortho_clean)

                is_correct = (
                    gold_joined in fused_orthos
                    or gold_joined_sp in fused_orthos
                    or gold_joined_tiret in fused_orthos
                    or gold_joined_apo in fused_orthos
                    or gold_joined.replace("-", "").replace("'", "") in fused_orthos
                )

                best_fused = max(fused_entries, key=lambda e: e.get("freq") or 0)
                best_ortho = best_fused.get("ortho", "")
                best_freq = best_fused.get("freq", 0)

                for name, filtre in strategies.items():
                    try:
                        if filtre(span_phones, fused_phone, fused_entries, lex, gold_words, start):
                            if is_correct:
                                results[name]["vp"] += 1
                                results[name]["matches"].append(
                                    (gold_span, best_ortho, span_phones, best_freq)
                                )
                            else:
                                results[name]["fp"] += 1
                                results[name]["errors"].append(
                                    (gold_span, best_ortho, span_phones, best_freq)
                                )
                    except Exception:
                        pass

    # Afficher les resultats
    print(f"\n{'='*80}")
    print(f"Analyse des fusions IPA sur le dev set")
    print(f"{'='*80}")
    print(f"  Phrases: {n_phrases}")
    print(f"  Mots:    {n_mots}")
    print(f"  Max k:   {max_k}")
    print(f"{'='*80}\n")

    print(f"{'Strategie':<25} {'VP':>5} {'FP':>5} {'Total':>6} {'Precision':>10}")
    print("-" * 60)

    for name in sorted(results, key=lambda n: -(results[n]["vp"])):
        r = results[name]
        total = r["vp"] + r["fp"]
        prec = r["vp"] / total * 100 if total > 0 else 0
        print(f"{name:<25} {r['vp']:>5} {r['fp']:>5} {total:>6} {prec:>9.1f}%")

    # Details pour les strategies avec special chars
    for name in ["fused_has_special", "1oov+special", "au_moins_1_oov"]:
        r = results[name]
        total = r["vp"] + r["fp"]
        if total == 0:
            continue
        prec = r["vp"] / total * 100

        print(f"\n{'='*80}")
        print(f"--- {name} (VP={r['vp']}, FP={r['fp']}, prec={prec:.1f}%) ---")
        print(f"{'='*80}")

        print(f"\n  Vrais positifs (max 50):")
        for gold, ortho, phones, freq in r["matches"][:50]:
            print(f"    {' '.join(gold):30s} -> {ortho:25s} (phones: {'+'.join(phones)}, freq={freq:.1f})")

        print(f"\n  Faux positifs (max 50):")
        for gold, ortho, phones, freq in r["errors"][:50]:
            print(f"    {' '.join(gold):30s} -> {ortho:25s} (phones: {'+'.join(phones)}, freq={freq:.1f})")

    # Analyse des FP de fused_has_special par type d'erreur
    r = results["fused_has_special"]
    if r["fp"] > 0:
        print(f"\n{'='*80}")
        print("Analyse des FP de fused_has_special")
        print(f"{'='*80}\n")

        # Compter par ortho fusionne
        from collections import Counter
        fp_by_ortho = Counter()
        for gold, ortho, phones, freq in r["errors"]:
            fp_by_ortho[ortho] += 1
        print("  Top 30 faux positifs par ortho fusionne:")
        for ortho, count in fp_by_ortho.most_common(30):
            print(f"    {ortho:30s} : {count} occurrences")

        vp_by_ortho = Counter()
        for gold, ortho, phones, freq in r["matches"]:
            vp_by_ortho[ortho] += 1
        print("\n  Top 30 vrais positifs par ortho fusionne:")
        for ortho, count in vp_by_ortho.most_common(30):
            print(f"    {ortho:30s} : {count} occurrences")


def main():
    print("Chargement du dev set...")
    sentences = charger_dev_set()
    print(f"  {len(sentences)} phrases chargees")

    print("Chargement du phone_lexicon...")
    lex = charger_phone_lexicon()
    print(f"  {len(lex.phone_set)} phones charges")

    print("Analyse des fusions...")
    t0 = time.time()
    analyser_fusions(sentences, lex, max_k=5)
    dt = time.time() - t0
    print(f"\nAnalyse terminee en {dt:.1f}s")


if __name__ == "__main__":
    main()
