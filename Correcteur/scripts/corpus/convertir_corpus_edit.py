"""Convertit le corpus 200K en format edit-tag pour le BiLSTM tagger.

Entree : corpus_200k.jsonl (paires fautif/correct avec erreurs annotees)
Sortie  : corpus_edit.jsonl

Chaque ligne de sortie :
  {
    "tokens": ["les","chien","mange"],
    "morpho": [{"pos":"ART:def","nombre":"Plur",...}, ...],
    "tags": ["KEEP", "PLUR", "CONJ_3P"],
    "correct_tokens": ["les","chiens","mangent"]
  }

Usage :
    python scripts/corpus/convertir_corpus_edit.py [--input PATH] [--output PATH]
        [--identity N] [--validate] [--max-lines N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Ajouter src/ au path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT.parent / "Lexique" / "src"))

from lectura_correcteur._morpho import MorphoTagger
from lectura_correcteur._tags import KEEP, TAGS, appliquer_tag, detecter_tag
from lectura_lexique import Lexique

DEFAULT_INPUT = _ROOT / "data" / "corpus" / "corpus_200k.jsonl"
DEFAULT_OUTPUT = _ROOT / "data" / "corpus" / "corpus_edit.jsonl"
DEFAULT_LEXIQUE = _ROOT.parent / "Lexique" / "lexique_lectura.db"


def convertir_ligne(
    entry: dict,
    tagger: MorphoTagger,
    lexique: Lexique,
) -> dict | None:
    """Convertit une ligne du corpus en format edit-tag.

    Returns:
        Dictionnaire avec tokens, morpho, tags, correct_tokens.
        None si la conversion echoue.
    """
    fautif = entry["correct"]  # Note : dans le corpus, "correct" est la phrase source
    correct = entry["correct"]
    erreurs = entry.get("erreurs", [])

    # Tokeniser fautif et correct
    tokens_fautifs = entry["fautif"].split()
    tokens_corrects = correct.split()

    # Verifier que l'alignement est trivial (meme nombre de tokens)
    if len(tokens_fautifs) != len(tokens_corrects):
        return None

    # Tagger morpho sur le texte fautif (c'est ce que le modele verra)
    morpho_results = tagger.tag_words(tokens_fautifs)

    # Construire les morpho sous forme compacte
    morpho_list = []
    for mr in morpho_results:
        morpho_compact = {"pos": mr.get("pos", "")}
        for feat in ("genre", "nombre", "temps", "mode", "personne"):
            val = mr.get(feat)
            if val is not None:
                morpho_compact[feat] = val
        morpho_list.append(morpho_compact)

    # Construire les tags
    # Mapper les positions d'erreur pour un acces rapide
    erreur_par_pos: dict[int, dict] = {}
    for err in erreurs:
        erreur_par_pos[err["position"]] = err

    tags = []
    for i, (tok_f, tok_c) in enumerate(zip(tokens_fautifs, tokens_corrects)):
        if tok_f.lower() == tok_c.lower():
            tags.append(KEEP)
        else:
            err = erreur_par_pos.get(i, {})
            type_err = err.get("type", "")
            tag = detecter_tag(tok_f, tok_c, type_err, lexique)
            tags.append(tag)

    return {
        "tokens": tokens_fautifs,
        "morpho": morpho_list,
        "tags": tags,
        "correct_tokens": tokens_corrects,
    }


def generer_identite(
    entry: dict,
    tagger: MorphoTagger,
) -> dict | None:
    """Genere une paire identite (phrase correcte, tous tags = KEEP)."""
    tokens = entry["correct"].split()
    if len(tokens) < 3 or len(tokens) > 50:
        return None

    morpho_results = tagger.tag_words(tokens)

    morpho_list = []
    for mr in morpho_results:
        morpho_compact = {"pos": mr.get("pos", "")}
        for feat in ("genre", "nombre", "temps", "mode", "personne"):
            val = mr.get(feat)
            if val is not None:
                morpho_compact[feat] = val
        morpho_list.append(morpho_compact)

    return {
        "tokens": tokens,
        "morpho": morpho_list,
        "tags": [KEEP] * len(tokens),
        "correct_tokens": tokens,
    }


def valider_ligne(line_data: dict, lexique: Lexique) -> tuple[bool, str]:
    """Valide qu'appliquer_tag reproduit les tokens corrects."""
    tokens = line_data["tokens"]
    tags = line_data["tags"]
    correct = line_data["correct_tokens"]

    for i, (tok, tag, corr) in enumerate(zip(tokens, tags, correct)):
        if tag == KEEP:
            if tok.lower() != corr.lower():
                return False, f"pos={i}: KEEP mais {tok!r} != {corr!r}"
        else:
            result = appliquer_tag(tok.lower(), tag, lexique)
            if result.lower() != corr.lower():
                return False, f"pos={i}: appliquer_tag({tok!r}, {tag}) = {result!r}, attendu {corr!r}"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Convertir corpus en format edit-tag")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lexique", type=Path, default=DEFAULT_LEXIQUE)
    parser.add_argument("--identity", type=int, default=70000,
                        help="Nombre de paires identite a ajouter")
    parser.add_argument("--validate", action="store_true",
                        help="Valider chaque ligne (lent)")
    parser.add_argument("--max-lines", type=int, default=0,
                        help="Limiter le nombre de lignes source (0 = toutes)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Chargement du lexique : {args.lexique}")
    lexique = Lexique(str(args.lexique))
    tagger = MorphoTagger()

    # Compteurs
    tag_counts: Counter = Counter()
    n_converted = 0
    n_skipped = 0
    n_identity = 0
    n_valid = 0
    n_invalid = 0
    uncovered: list[str] = []

    # Lire toutes les lignes pour pouvoir echantillonner les identites
    print(f"Lecture du corpus : {args.input}")
    entries = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if args.max_lines and i >= args.max_lines:
                break
            entries.append(json.loads(line))

    print(f"  {len(entries)} lignes chargees")

    # Convertir les paires avec erreurs
    print("Conversion des paires avec erreurs...")
    output_lines: list[dict] = []

    for i, entry in enumerate(entries):
        if i % 10000 == 0 and i > 0:
            print(f"  {i}/{len(entries)}...")

        result = convertir_ligne(entry, tagger, lexique)
        if result is None:
            n_skipped += 1
            continue

        # Compter les tags
        for tag in result["tags"]:
            tag_counts[tag] += 1

        # Verifier les tags non-KEEP pour les erreurs attendues
        erreurs = entry.get("erreurs", [])
        for err in erreurs:
            pos = err["position"]
            type_err = err["type"]
            if type_err not in ("PHON", "ACCENT", "TYPO"):
                if pos < len(result["tags"]) and result["tags"][pos] == KEEP:
                    uncovered.append(
                        f"type={type_err}, orig={err['perturbe']!r}, "
                        f"corr={err['original']!r}"
                    )

        # Validation optionnelle
        if args.validate:
            ok, msg = valider_ligne(result, lexique)
            if ok:
                n_valid += 1
            else:
                n_invalid += 1
                if n_invalid <= 20:
                    print(f"  INVALIDE ligne {i}: {msg}")

        output_lines.append(result)
        n_converted += 1

    # Generer les paires identite
    print(f"Generation de {args.identity} paires identite...")
    # Echantillonner des phrases correctes
    indices = list(range(len(entries)))
    random.shuffle(indices)

    for idx in indices:
        if n_identity >= args.identity:
            break
        result = generer_identite(entries[idx], tagger)
        if result is None:
            continue

        for tag in result["tags"]:
            tag_counts[tag] += 1

        output_lines.append(result)
        n_identity += 1

    # Melanger
    random.shuffle(output_lines)

    # Ecrire
    print(f"Ecriture : {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for line_data in output_lines:
            f.write(json.dumps(line_data, ensure_ascii=False) + "\n")

    # Stats
    total_tags = sum(tag_counts.values())
    print(f"\n--- Statistiques ---")
    print(f"Paires converties  : {n_converted}")
    print(f"Paires identite    : {n_identity}")
    print(f"Paires sautees     : {n_skipped}")
    print(f"Total ecrit        : {len(output_lines)}")
    if args.validate:
        print(f"Valides            : {n_valid}")
        print(f"Invalides          : {n_invalid}")
    print(f"\nDistribution des tags ({total_tags} total):")
    for tag in TAGS:
        count = tag_counts.get(tag, 0)
        pct = 100 * count / total_tags if total_tags else 0
        if count > 0:
            print(f"  {tag:15s} : {count:8d}  ({pct:5.1f}%)")

    n_edit = sum(c for t, c in tag_counts.items() if t != KEEP)
    n_edit_expected = sum(1 for case in uncovered for _ in [0])  # just len
    n_expected = n_edit + len(uncovered)
    if n_expected > 0:
        coverage = n_edit / n_expected
        print(f"\nCouverture edit (ACC/CONJ/PP/HOMO) : {n_edit}/{n_expected} = {coverage:.1%}")

    if uncovered:
        print(f"\nCas non couverts ({len(uncovered)}):")
        for case in uncovered[:30]:
            print(f"  {case}")

    lexique.close()
    print("\nTermine.")


if __name__ == "__main__":
    main()
