#!/usr/bin/env python3
"""Evaluation du BiLSTM edit tagger.

Metriques :
- Tag accuracy (par token)
- Phrase accuracy
- F1 par type d'erreur (HOMO, ACC, CONJ, PP, PHON, ACCENT, TYPO)
- Precision sur paires identite (faux positifs)
- Comparaison pipeline regles vs pipeline editeur sur le benchmark
- Vitesse par backend (ONNX / NumPy / Pure)

Usage :
    python scripts/evaluer_editeur.py
    python scripts/evaluer_editeur.py --corpus data/corpus/corpus_edit.jsonl
    python scripts/evaluer_editeur.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))
sys.path.insert(0, str(_ROOT.parent / "Lexique" / "src"))

from lectura_correcteur._tags import KEEP, N_TAGS, TAG2IDX, TAGS, appliquer_tag
from lectura_lexique import Lexique

DEFAULT_CORPUS = _ROOT / "data" / "corpus" / "corpus_edit.jsonl"
DEFAULT_LEXIQUE = _ROOT.parent / "Lexique" / "lexique_lectura.db"
DEFAULT_DATA = _ROOT / "src" / "lectura_correcteur" / "data"
DEFAULT_VOCAB = DEFAULT_DATA / "editeur_vocab.json"
DEFAULT_WEIGHTS = DEFAULT_DATA / "editeur_weights.json.gz"
DEFAULT_ONNX = DEFAULT_DATA / "editeur_int8.onnx"


# ---------------------------------------------------------------------------
# Evaluation sur corpus edit
# ---------------------------------------------------------------------------

def evaluer_tags(
    corpus_path: Path,
    editeur,
    max_lines: int = 0,
) -> dict:
    """Evalue le tagger sur le corpus edit-tag."""
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    n_tokens = 0
    n_tokens_ok = 0
    n_phrases = 0
    n_phrases_ok = 0
    n_identity = 0
    n_identity_ok = 0

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            data = json.loads(line)
            tokens = data["tokens"]
            morpho = data["morpho"]
            gold_tags = data["tags"]

            pred_tags = editeur.predire_tags(tokens, morpho)

            is_identity = all(t == KEEP for t in gold_tags)
            if is_identity:
                n_identity += 1

            phrase_ok = True
            for j in range(len(tokens)):
                g = gold_tags[j] if j < len(gold_tags) else KEEP
                p = pred_tags[j] if j < len(pred_tags) else KEEP

                n_tokens += 1
                if p == g:
                    n_tokens_ok += 1
                    tp[g] += 1
                else:
                    phrase_ok = False
                    fp[p] += 1
                    fn[g] += 1

            n_phrases += 1
            if phrase_ok:
                n_phrases_ok += 1
                if is_identity:
                    n_identity_ok += 1

    # F1 par tag
    f1_par_tag: dict[str, dict[str, float]] = {}
    for tag in TAGS:
        t = tp[tag]
        f = fp[tag]
        n = fn[tag]
        prec = t / (t + f) if (t + f) > 0 else 0
        rec = t / (t + n) if (t + n) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if t + f + n > 0:
            f1_par_tag[tag] = {"precision": prec, "recall": rec, "f1": f1,
                               "support": t + n}

    # Macro F1 non-KEEP
    non_keep_f1 = [v["f1"] for k, v in f1_par_tag.items() if k != KEEP]
    macro_f1 = sum(non_keep_f1) / len(non_keep_f1) if non_keep_f1 else 0

    return {
        "tag_accuracy": n_tokens_ok / n_tokens if n_tokens else 0,
        "phrase_accuracy": n_phrases_ok / n_phrases if n_phrases else 0,
        "macro_f1_non_keep": macro_f1,
        "f1_par_tag": f1_par_tag,
        "identity_precision": n_identity_ok / n_identity if n_identity else 0,
        "n_tokens": n_tokens,
        "n_phrases": n_phrases,
        "n_identity": n_identity,
    }


# ---------------------------------------------------------------------------
# Benchmark de vitesse
# ---------------------------------------------------------------------------

def benchmark_vitesse(
    editeur,
    n_phrases: int = 100,
) -> float:
    """Mesure la vitesse moyenne en ms/phrase."""
    # Generer des phrases test
    phrases = [
        (
            ["les", "enfant", "mange", "des", "pomme", "rouge"],
            [{"pos": "ART:def"}, {"pos": "NOM"}, {"pos": "VER"},
             {"pos": "ART:ind"}, {"pos": "NOM"}, {"pos": "ADJ"}],
        )
    ] * n_phrases

    t0 = time.perf_counter()
    for tokens, morpho in phrases:
        editeur.predire_tags(tokens, morpho)
    dt = time.perf_counter() - t0

    return (dt / n_phrases) * 1000  # ms/phrase


# ---------------------------------------------------------------------------
# Benchmark comparatif (pipeline regles vs editeur)
# ---------------------------------------------------------------------------

def benchmark_comparatif(lexique_path: Path) -> None:
    """Compare le pipeline regles vs editeur sur le benchmark existant."""
    # Importer le benchmark
    try:
        from corpus_evaluation import CORPUS
    except ImportError:
        print("corpus_evaluation.py non trouve dans tests/")
        return

    from lectura_correcteur import Correcteur
    from lectura_correcteur._config import CorrecteurConfig

    lex = Lexique(str(lexique_path))

    # Pipeline regles
    config_regles = CorrecteurConfig()
    correcteur_regles = Correcteur(lex, config=config_regles)

    # Pipeline editeur (si disponible)
    config_editeur = CorrecteurConfig(activer_editeur=True)
    correcteur_editeur = Correcteur(lex, config=config_editeur)
    has_editeur = correcteur_editeur._editeur is not None

    print(f"\nBenchmark comparatif ({len(CORPUS)} cas)")
    print(f"{'':3s} {'Fautif':40s} {'Attendu':40s} {'Regles':40s}", end="")
    if has_editeur:
        print(f" {'Editeur':40s}", end="")
    print()
    print("-" * (123 + (41 if has_editeur else 0)))

    n_ok_regles = 0
    n_ok_editeur = 0

    for i, cas in enumerate(CORPUS):
        fautif = cas.erronee
        attendu = cas.attendue

        result_regles = correcteur_regles.corriger(fautif)
        ok_r = result_regles.phrase_corrigee.lower() == attendu.lower()
        n_ok_regles += int(ok_r)
        status_r = "OK" if ok_r else "XX"

        line = f"{i+1:3d} {fautif:40s} {attendu:40s} [{status_r}] {result_regles.phrase_corrigee:35s}"

        if has_editeur:
            result_editeur = correcteur_editeur.corriger(fautif)
            ok_e = result_editeur.phrase_corrigee.lower() == attendu.lower()
            n_ok_editeur += int(ok_e)
            status_e = "OK" if ok_e else "XX"
            line += f" [{status_e}] {result_editeur.phrase_corrigee:35s}"

        print(line)

    total = len(CORPUS)
    print(f"\nRegles : {n_ok_regles}/{total} = {100*n_ok_regles/total:.1f}%")
    if has_editeur:
        print(f"Editeur: {n_ok_editeur}/{total} = {100*n_ok_editeur/total:.1f}%")

    lex.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluation du BiLSTM edit tagger")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--lexique", type=Path, default=DEFAULT_LEXIQUE)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark comparatif regles vs editeur")
    parser.add_argument("--speed", action="store_true",
                        help="Benchmark de vitesse par backend")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_comparatif(args.lexique)
        return

    # Evaluer sur le corpus edit
    if not args.corpus.exists():
        print(f"Corpus non trouve : {args.corpus}")
        print("Executez d'abord : python scripts/corpus/convertir_corpus_edit.py")
        return

    if not args.vocab.exists():
        print(f"Vocabulaire non trouve : {args.vocab}")
        print("Executez d'abord : python scripts/entrainement/entrainer_editeur.py")
        return

    if not args.weights.exists():
        print(f"Poids non trouves : {args.weights}")
        return

    # Charger le backend NumPy
    from lectura_correcteur._editeur_numpy import EditeurNumpy
    editeur = EditeurNumpy(args.weights, args.vocab)

    print("Evaluation sur le corpus edit-tag...")
    metrics = evaluer_tags(args.corpus, editeur, max_lines=args.max_lines)

    print(f"\n--- Resultats ---")
    print(f"Tag accuracy    : {metrics['tag_accuracy']:.3f}")
    print(f"Phrase accuracy : {metrics['phrase_accuracy']:.3f}")
    print(f"Macro F1 (non-KEEP) : {metrics['macro_f1_non_keep']:.3f}")
    print(f"Identity precision  : {metrics['identity_precision']:.3f}")
    print(f"Tokens: {metrics['n_tokens']:,}, Phrases: {metrics['n_phrases']:,}, "
          f"Identity: {metrics['n_identity']:,}")

    print(f"\nF1 par tag :")
    for tag, vals in sorted(metrics["f1_par_tag"].items(),
                             key=lambda x: -x[1]["f1"]):
        if vals["support"] > 0:
            print(f"  {tag:15s} P={vals['precision']:.3f} R={vals['recall']:.3f} "
                  f"F1={vals['f1']:.3f} (n={vals['support']})")

    # Benchmark de vitesse
    if args.speed:
        print("\n--- Vitesse ---")
        ms = benchmark_vitesse(editeur, n_phrases=200)
        print(f"  NumPy  : {ms:.1f} ms/phrase")

        # ONNX si disponible
        onnx_path = _ROOT / "src" / "lectura_correcteur" / "data" / "editeur_int8.onnx"
        if onnx_path.exists():
            try:
                from lectura_correcteur._editeur_onnx import EditeurOnnx
                editeur_onnx = EditeurOnnx(onnx_path, args.vocab)
                ms_onnx = benchmark_vitesse(editeur_onnx, n_phrases=200)
                print(f"  ONNX   : {ms_onnx:.1f} ms/phrase")
            except ImportError:
                print("  ONNX   : onnxruntime non installe")

        # Pure Python
        try:
            from lectura_correcteur._editeur_pure import EditeurPure
            editeur_pure = EditeurPure(args.weights, args.vocab)
            ms_pure = benchmark_vitesse(editeur_pure, n_phrases=20)
            print(f"  Pure   : {ms_pure:.1f} ms/phrase")
        except Exception as e:
            print(f"  Pure   : erreur ({e})")


if __name__ == "__main__":
    main()
