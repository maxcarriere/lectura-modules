#!/usr/bin/env python3
"""Benchmark d'etiquetage grammatical pur (POS + Morpho).

Evalue la precision de l'etiquetage POS et morphologique sur le corpus
UD French-GSD (dev set), en comparant au gold standard UD.

Strategies testees :
  A. G2P seul         — tagger neural, mot par mot avec contexte BiLSTM
  B. Lexique seul     — hypothese la plus frequente dans le lexique
  C. Viterbi seul     — Viterbi trigramme PM sur hypotheses lexique
  D. G2P + Viterbi    — G2P comme prior d'emission dans le Viterbi
  E. Lexique + G2P    — G2P pour desambiguer parmi les hypotheses lexique

Gold standard : annotations UD French-GSD (pos_tag, morpho).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/data/work/projets/lectura/workspace/Modules/Lexique/src")

KIT_DIR = "/home/moi/Documents/work/projets/lectura/workspace/Corpus/Kit-G2P-P2G/corpus/phrases"
LEXIQUE_DB = "/data/work/projets/lectura/workspace/Lexique/lexique_lectura_v4.db"


def charger_dev_set(max_n: int = 0) -> list[list[dict]]:
    """Charge les phrases du dev set UD French-GSD."""
    path = os.path.join(KIT_DIR, "sentences_dev.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    phrases = [s["tokens"] for s in data if s.get("tokens")]
    if max_n:
        phrases = phrases[:max_n]
    return phrases


# ── Mapping POS UD → POS base ─────────────────────────────────────────────

def _pos_base(pos: str) -> str:
    """Extrait le POS de base (sans sous-type). Ex: ART:def → ART."""
    return pos.split(":")[0]


# ── Strategie A : G2P seul ─────────────────────────────────────────────────

def etiqueter_g2p(phrases: list[list[dict]], tagger) -> list[list[dict]]:
    """Etiquette chaque phrase avec le G2P neural."""
    results = []
    for tokens in phrases:
        mots = [t["form"] for t in tokens]
        try:
            tags = tagger.tag_words_rich(mots)
        except Exception:
            tags = [{}] * len(mots)

        phrase_result = []
        for i, t in enumerate(tokens):
            g = tags[i] if i < len(tags) else {}
            # Mapper les valeurs G2P (courtes) vers UD (longues)
            nombre_map = {"s": "Sing", "p": "Plur"}
            genre_map = {"m": "Masc", "f": "Fem"}
            phrase_result.append({
                "pos": g.get("pos", "NOM"),
                "number": nombre_map.get(g.get("nombre", ""), "_"),
                "gender": genre_map.get(g.get("genre", ""), "_"),
                "person": g.get("personne", "_") or "_",
            })
        results.append(phrase_result)
    return results


# ── Strategie B : Lexique seul (hypothese la plus frequente) ───────────────

def etiqueter_lexique(phrases: list[list[dict]], lexique) -> list[list[dict]]:
    """Etiquette chaque mot avec l'hypothese lexique la plus frequente."""
    from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS

    _GENRE_MAP = {"m": "Masc", "f": "Fem", "": "_"}
    _NOMBRE_MAP = {"s": "Sing", "p": "Plur", "": "_"}

    results = []
    for tokens in phrases:
        phrase_result = []
        for t in tokens:
            form = t["form"].lower()

            # Mots-outils : POS force
            override = _FUNCTION_WORD_POS.get(form)

            infos = lexique.info(form) if hasattr(lexique, "info") else []
            if not infos:
                phrase_result.append({
                    "pos": override or "NOM",
                    "number": "_", "gender": "_", "person": "_",
                })
                continue

            # Filtrer par override POS si disponible
            if override:
                override_base = override.split(":")[0]
                filtered = [e for e in infos if (e.get("cgram", "").split(":")[0]) == override_base]
                if filtered:
                    infos = filtered

            # Prendre l'entree la plus frequente
            best = max(infos, key=lambda e: float(e.get("freq") or 0))
            cgram = best.get("cgram", "NOM")
            if override and ":" not in cgram:
                if cgram == override.split(":")[0]:
                    cgram = override

            genre = _GENRE_MAP.get(best.get("genre", ""), "_")
            nombre = _NOMBRE_MAP.get(best.get("nombre", ""), "_")
            personne = best.get("personne", "") or "_"

            phrase_result.append({
                "pos": cgram,
                "number": nombre,
                "gender": genre,
                "person": personne,
            })
        results.append(phrase_result)
    return results


# ── Strategie C : Viterbi seul (lexique + n-gram PM) ──────────────────────

def etiqueter_viterbi(
    phrases: list[list[dict]],
    lexique,
    pos_ngram,
    lm_homophones=None,
) -> list[list[dict]]:
    """Etiquette avec le Viterbi PM (sans expand, sans G2P)."""
    from lectura_correcteur._analyse_grammaticale import analyser_phrase

    results = []
    for tokens in phrases:
        mots = [t["form"] for t in tokens]
        analyses = analyser_phrase(
            mots, lexique, pos_ngram,
            lm_homophones=lm_homophones,
            expand_morpho=False,
            expand_homophones=False,
        )
        phrase_result = []
        for a in analyses:
            phrase_result.append({
                "pos": a.pos,
                "number": a.nombre,
                "gender": a.genre,
                "person": a.personne if a.personne != "_" else "_",
            })
        results.append(phrase_result)
    return results


# ── Strategie D : G2P + Viterbi ───────────────────────────────────────────

def etiqueter_g2p_viterbi(
    phrases: list[list[dict]],
    lexique,
    pos_ngram,
    tagger,
    lm_homophones=None,
    w_g2p: float = 1.0,
) -> list[list[dict]]:
    """Etiquette avec le Viterbi PM + G2P comme prior d'emission."""
    from lectura_correcteur._analyse_grammaticale import analyser_phrase

    results = []
    for tokens in phrases:
        mots = [t["form"] for t in tokens]
        analyses = analyser_phrase(
            mots, lexique, pos_ngram,
            lm_homophones=lm_homophones,
            tagger=tagger,
            w_g2p=w_g2p,
            expand_morpho=False,
            expand_homophones=False,
        )
        phrase_result = []
        for a in analyses:
            phrase_result.append({
                "pos": a.pos,
                "number": a.nombre,
                "gender": a.genre,
                "person": a.personne if a.personne != "_" else "_",
            })
        results.append(phrase_result)
    return results


# ── Strategie E : Lexique + G2P (G2P desambigue parmi hypotheses lexique) ─

def etiqueter_lexique_g2p(
    phrases: list[list[dict]],
    lexique,
    tagger,
) -> list[list[dict]]:
    """G2P choisit parmi les hypotheses du lexique."""
    from lectura_correcteur._tagger_lexique import _FUNCTION_WORD_POS

    _GENRE_MAP = {"m": "Masc", "f": "Fem", "": "_"}
    _NOMBRE_MAP = {"s": "Sing", "p": "Plur", "": "_"}

    results = []
    for tokens in phrases:
        mots = [t["form"] for t in tokens]
        try:
            g2p_tags = tagger.tag_words_rich(mots)
        except Exception:
            g2p_tags = [{}] * len(mots)

        phrase_result = []
        for i, t in enumerate(tokens):
            form = t["form"].lower()
            g = g2p_tags[i] if i < len(g2p_tags) else {}
            g2p_pos = g.get("pos", "NOM")

            override = _FUNCTION_WORD_POS.get(form)
            infos = lexique.info(form) if hasattr(lexique, "info") else []

            if not infos:
                nombre_map = {"s": "Sing", "p": "Plur"}
                genre_map = {"m": "Masc", "f": "Fem"}
                phrase_result.append({
                    "pos": override or g2p_pos,
                    "number": nombre_map.get(g.get("nombre", ""), "_"),
                    "gender": genre_map.get(g.get("genre", ""), "_"),
                    "person": g.get("personne", "_") or "_",
                })
                continue

            # Filtrer par POS G2P (base)
            g2p_base = g2p_pos.split(":")[0]
            matching = [e for e in infos if e.get("cgram", "").split(":")[0] == g2p_base]
            if not matching:
                matching = infos  # fallback : tous

            # Override mot-outil
            if override:
                override_base = override.split(":")[0]
                ov_matching = [e for e in matching if e.get("cgram", "").split(":")[0] == override_base]
                if ov_matching:
                    matching = ov_matching

            # Prendre la plus frequente parmi les matchs
            best = max(matching, key=lambda e: float(e.get("freq") or 0))
            cgram = best.get("cgram", "NOM")
            if override and ":" not in cgram:
                if cgram == override.split(":")[0]:
                    cgram = override

            genre = _GENRE_MAP.get(best.get("genre", ""), "_")
            nombre = _NOMBRE_MAP.get(best.get("nombre", ""), "_")
            personne = best.get("personne", "") or "_"

            phrase_result.append({
                "pos": cgram,
                "number": nombre,
                "gender": genre,
                "person": personne,
            })
        results.append(phrase_result)
    return results


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluer(
    phrases: list[list[dict]],
    predictions: list[list[dict]],
    label: str,
) -> dict[str, float]:
    """Compare predictions au gold UD. Retourne les metriques."""
    total = 0
    pos_ok = 0
    pos_base_ok = 0
    number_ok = 0
    gender_ok = 0
    person_ok = 0
    morpho_ok = 0  # number + gender
    full_ok = 0    # pos_base + number + gender

    # Erreurs POS detaillees
    pos_errors: Counter = Counter()

    for phrase, preds in zip(phrases, predictions):
        for i, (tok, pred) in enumerate(zip(phrase, preds)):
            total += 1

            gold_pos = tok["pos_tag"]
            gold_morpho = tok.get("morpho", {})
            gold_number = gold_morpho.get("Number", "_")
            gold_gender = gold_morpho.get("Gender", "_")
            gold_person = gold_morpho.get("Person", "_")

            pred_pos = pred["pos"]
            pred_number = pred["number"]
            pred_gender = pred["gender"]
            pred_person = pred["person"]

            # POS exact (avec sous-type)
            if pred_pos == gold_pos:
                pos_ok += 1

            # POS base (sans sous-type)
            if _pos_base(pred_pos) == _pos_base(gold_pos):
                pos_base_ok += 1
            else:
                pos_errors[(_pos_base(gold_pos), _pos_base(pred_pos))] += 1

            # Number (ignorer si gold = "_")
            num_match = (gold_number == "_") or (pred_number == gold_number)
            if num_match:
                number_ok += 1

            # Gender (ignorer si gold = "_")
            gen_match = (gold_gender == "_") or (pred_gender == gold_gender)
            if gen_match:
                gender_ok += 1

            # Person (ignorer si gold = "_")
            per_match = (gold_person == "_") or (pred_person == gold_person)
            if per_match:
                person_ok += 1

            # Morpho = number + gender corrects
            if num_match and gen_match:
                morpho_ok += 1

            # Full = pos_base + morpho
            if _pos_base(pred_pos) == _pos_base(gold_pos) and num_match and gen_match:
                full_ok += 1

    metrics = {
        "total": total,
        "pos_exact": pos_ok / total * 100 if total else 0,
        "pos_base": pos_base_ok / total * 100 if total else 0,
        "number": number_ok / total * 100 if total else 0,
        "gender": gender_ok / total * 100 if total else 0,
        "person": person_ok / total * 100 if total else 0,
        "morpho": morpho_ok / total * 100 if total else 0,
        "full": full_ok / total * 100 if total else 0,
    }

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  Mots evalues      : {total}")
    print(f"  POS exact         : {pos_ok}/{total} ({metrics['pos_exact']:.2f}%)")
    print(f"  POS base          : {pos_base_ok}/{total} ({metrics['pos_base']:.2f}%)")
    print(f"  Number            : {number_ok}/{total} ({metrics['number']:.2f}%)")
    print(f"  Gender            : {gender_ok}/{total} ({metrics['gender']:.2f}%)")
    print(f"  Person            : {person_ok}/{total} ({metrics['person']:.2f}%)")
    print(f"  Morpho (N+G)      : {morpho_ok}/{total} ({metrics['morpho']:.2f}%)")
    print(f"  Full (POS+N+G)    : {full_ok}/{total} ({metrics['full']:.2f}%)")

    if pos_errors:
        print(f"\n  Top 15 confusions POS :")
        for (gold, pred), n in pos_errors.most_common(15):
            print(f"    {gold:12s} -> {pred:12s} : {n}")

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark etiquetage POS+Morpho")
    parser.add_argument("--max-phrases", type=int, default=0,
                        help="Max phrases (0 = toutes)")
    parser.add_argument("--strategies", type=str, default="A,B,C,D,E",
                        help="Strategies a tester (A,B,C,D,E)")
    args = parser.parse_args()

    strategies = [s.strip().upper() for s in args.strategies.split(",")]

    print("Chargement dev set UD French-GSD...")
    phrases = charger_dev_set(max_n=args.max_phrases)
    total_tokens = sum(len(p) for p in phrases)
    print(f"  {len(phrases)} phrases, {total_tokens} tokens")

    # Charger les ressources selon les strategies demandees
    need_lexique = any(s in strategies for s in ("B", "C", "D", "E"))
    need_g2p = any(s in strategies for s in ("A", "D", "E"))
    need_viterbi = any(s in strategies for s in ("C", "D"))

    lexique = None
    lex = None
    pos_ngram = None
    lm_homophones = None
    tagger = None

    if need_lexique:
        from lectura_lexique import Lexique
        from lectura_correcteur._utils import LexiqueNormalise
        print("Chargement lexique...")
        lexique = Lexique(LEXIQUE_DB)
        lex = LexiqueNormalise(lexique)

    if need_viterbi:
        from lectura_correcteur._pos_ngram import PosNgram
        pos_ngram_db = os.path.join(
            _PROJECT_ROOT, "src", "lectura_correcteur", "data", "pos_ngram.db",
        )
        print("Chargement POS n-gram...")
        pos_ngram = PosNgram(pos_ngram_db)

        try:
            from lectura_correcteur._lm_homophones import LMHomophones
            lm_db = os.path.join(
                _PROJECT_ROOT, "src", "lectura_correcteur", "data", "homophones_trigrams.db",
            )
            if os.path.exists(lm_db):
                lm_homophones = LMHomophones(lm_db)
        except Exception:
            pass

    if need_g2p:
        print("Chargement G2P Unifie V2...")
        try:
            from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
            tagger = creer_adapter_g2p_unifie()
            if tagger:
                print("  G2P charge")
            else:
                print("  ATTENTION: G2P indisponible")
                strategies = [s for s in strategies if s not in ("A", "D", "E")]
        except Exception as e:
            print(f"  ERREUR G2P: {e}")
            strategies = [s for s in strategies if s not in ("A", "D", "E")]

    print(f"\nStrategies a evaluer : {', '.join(strategies)}")

    all_metrics = {}

    # A. G2P seul
    if "A" in strategies and tagger:
        t0 = time.time()
        preds = etiqueter_g2p(phrases, tagger)
        elapsed = time.time() - t0
        m = evaluer(phrases, preds, f"A. G2P seul ({elapsed:.1f}s)")
        all_metrics["A"] = m

    # B. Lexique seul
    if "B" in strategies and lex:
        t0 = time.time()
        preds = etiqueter_lexique(phrases, lex)
        elapsed = time.time() - t0
        m = evaluer(phrases, preds, f"B. Lexique seul ({elapsed:.1f}s)")
        all_metrics["B"] = m

    # C. Viterbi seul (lexique + n-gram)
    if "C" in strategies and lex and pos_ngram:
        t0 = time.time()
        preds = etiqueter_viterbi(phrases, lex, pos_ngram, lm_homophones)
        elapsed = time.time() - t0
        m = evaluer(phrases, preds, f"C. Viterbi (lexique + n-gram PM) ({elapsed:.1f}s)")
        all_metrics["C"] = m

    # D. G2P + Viterbi
    if "D" in strategies and lex and pos_ngram and tagger:
        t0 = time.time()
        preds = etiqueter_g2p_viterbi(
            phrases, lex, pos_ngram, tagger, lm_homophones, w_g2p=1.0,
        )
        elapsed = time.time() - t0
        m = evaluer(phrases, preds, f"D. G2P + Viterbi (w_g2p=1.0) ({elapsed:.1f}s)")
        all_metrics["D"] = m

    # E. Lexique + G2P
    if "E" in strategies and lex and tagger:
        t0 = time.time()
        preds = etiqueter_lexique_g2p(phrases, lex, tagger)
        elapsed = time.time() - t0
        m = evaluer(phrases, preds, f"E. Lexique + G2P ({elapsed:.1f}s)")
        all_metrics["E"] = m

    # ── Tableau comparatif ──────────────────────────────────────────────
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARATIF")
        print(f"{'='*70}")
        header = f"  {'Strategie':<25s} {'POS%':>7s} {'POSb%':>7s} {'Nbr%':>7s} {'Gen%':>7s} {'Per%':>7s} {'N+G%':>7s} {'Full%':>7s}"
        print(header)
        print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        labels = {
            "A": "G2P seul",
            "B": "Lexique seul",
            "C": "Viterbi",
            "D": "G2P + Viterbi",
            "E": "Lexique + G2P",
        }
        for key in ("A", "B", "C", "D", "E"):
            if key not in all_metrics:
                continue
            m = all_metrics[key]
            print(
                f"  {labels[key]:<25s}"
                f" {m['pos_exact']:7.2f}"
                f" {m['pos_base']:7.2f}"
                f" {m['number']:7.2f}"
                f" {m['gender']:7.2f}"
                f" {m['person']:7.2f}"
                f" {m['morpho']:7.2f}"
                f" {m['full']:7.2f}"
            )

    print("\nTermine.")


if __name__ == "__main__":
    main()
