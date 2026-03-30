#!/usr/bin/env python3
"""Évaluation du Tokeniseur Lectura sur un jeu de test intégré.

Évalue trois aspects :
  1. Tokenisation : nombre et contenu des tokens vs gold
  2. Normalisation : texte normalisé vs gold
  3. Round-trip : tokens reconstituent-ils le texte normalisé ?

Usage :
    python evaluer.py
    python evaluer.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

from lectura_tokeniseur import (
    LecturaTokeniseur,
    Token,
    TokenType,
)


# ══════════════════════════════════════════════════════════════════════════════
# Jeu de test intégré
# ══════════════════════════════════════════════════════════════════════════════

# Format : (texte_brut, texte_normalisé_attendu, tokens_attendus)
# tokens_attendus : liste de (text, type_value)

_TEST_NORMALISATION: list[tuple[str, str]] = [
    # Espaces
    ("L'enfant   mange   du   chocolat.", "L'enfant mange du chocolat."),
    ("  bonjour  ", "bonjour"),
    # Apostrophes
    ("l' enfant", "l'enfant"),
    ("l  '  ami", "l'ami"),
    # Ellipses
    ("bonjour...monde", "bonjour … monde"),
    ("(...)", "(…)"),
    # Guillemets
    ('"bonjour"', "« bonjour »"),
    # Nombres
    ("1 000 000", "1'000'000"),
    ("3,14", "3.14"),
    ("12 345", "12345"),
    # Tirets
    ("peut-être", "peut-être"),
    ("bonjour - monde", "bonjour - monde"),
    # Ponctuation forte
    ("bonjour!monde", "bonjour ! monde"),
    ("vraiment?non!", "vraiment ? non !"),
    # Parenthèses
    ("( bonjour )", "(bonjour)"),
    ("[ test ]", "[test]"),
    # Ponctuation faible
    ("bonjour ,monde", "bonjour, monde"),
    ("a .b", "a. b"),
]

_TEST_TOKENISATION: list[tuple[str, list[tuple[str, str]]]] = [
    # Phrase simple
    ("L'enfant mange du chocolat.", [
        ("L", "mot"), ("'", "separateur"), ("enfant", "mot"),
        (" ", "separateur"), ("mange", "mot"), (" ", "separateur"),
        ("du", "mot"), (" ", "separateur"), ("chocolat", "mot"),
        (".", "ponctuation"),
    ]),
    # Nombres
    ("Il a 42 ans.", [
        ("Il", "mot"), (" ", "separateur"), ("a", "mot"),
        (" ", "separateur"), ("42", "nombre"), (" ", "separateur"),
        ("ans", "mot"), (".", "ponctuation"),
    ]),
    # Sigles
    ("Le FBI enquête.", [
        ("Le", "mot"), (" ", "separateur"), ("FBI", "sigle"),
        (" ", "separateur"), ("enquête", "mot"), (".", "ponctuation"),
    ]),
    # Mots composés → composé
    ("peut-être", [
        ("peut-être", "mot"),
    ]),
    # Ponctuation variée
    ("Ah, vraiment ?", [
        ("Ah", "mot"), (",", "ponctuation"), (" ", "separateur"),
        ("vraiment", "mot"), (" ", "separateur"), ("?", "ponctuation"),
    ]),
    # Apostrophe dans le mot → composé
    ("aujourd'hui", [
        ("aujourd'hui", "mot"),
    ]),
    # Phrase avec élision
    ("J'ai l'impression qu'il s'en va.", [
        ("J", "mot"), ("'", "separateur"), ("ai", "mot"),
        (" ", "separateur"),
        ("l", "mot"), ("'", "separateur"), ("impression", "mot"),
        (" ", "separateur"),
        ("qu", "mot"), ("'", "separateur"), ("il", "mot"),
        (" ", "separateur"),
        ("s", "mot"), ("'", "separateur"), ("en", "mot"),
        (" ", "separateur"),
        ("va", "mot"), (".", "ponctuation"),
    ]),
    # Nombre décimal (après normalisation 3,14 → 3.14)
    ("3.14", [
        ("3.14", "nombre"),
    ]),
    # Phrase vide
    ("", []),
    # Accents et caractères spéciaux
    ("À côté de l'hôtel", [
        ("À", "mot"), (" ", "separateur"), ("côté", "mot"),
        (" ", "separateur"), ("de", "mot"), (" ", "separateur"),
        ("l", "mot"), ("'", "separateur"), ("hôtel", "mot"),
    ]),
    # Guillemets (normalisés)
    ("« bonjour »", [
        ("«", "ponctuation"), (" ", "separateur"), ("bonjour", "mot"),
        (" ", "separateur"), ("»", "ponctuation"),
    ]),
    # Élisions (inchangé)
    ("l'enfant", [
        ("l", "mot"), ("'", "separateur"), ("enfant", "mot"),
    ]),
    # Composé avec apostrophe interne
    ("chef-d'œuvre", [
        ("chef-d'œuvre", "mot"),
    ]),
    # Composé à tiret triple
    ("arc-en-ciel", [
        ("arc-en-ciel", "mot"),
    ]),
]

# Cas spéciaux pour la couverture
_TEST_COUVERTURE: list[tuple[str, str, list[str]]] = [
    # (texte_brut, description_cas, mots_attendus)
    ("C'est-à-dire que non.", "locution composée", ["C'est-à-dire", "que", "non"]),
    ("Mme Dupont est arrivée.", "abréviation courante", ["Mme", "Dupont", "est", "arrivée"]),
    ("Il mange, boit et dort.", "énumération virgules", ["Il", "mange", "boit", "et", "dort"]),
    ("Le chat dort.", "phrase basique", ["Le", "chat", "dort"]),
    ("Bonjour le monde !", "exclamation simple", ["Bonjour", "le", "monde"]),
    ("Jean-Pierre est là.", "prénom composé", ["Jean-Pierre", "est", "là"]),
    ("L'arc-en-ciel est beau.", "composé après élision", ["L", "arc-en-ciel", "est", "beau"]),
    ("Le chef-d'œuvre est là.", "composé tiret+apostrophe", ["Le", "chef-d'œuvre", "est", "là"]),
    ("aujourd'hui il fait beau.", "apostrophe composé", ["aujourd'hui", "il", "fait", "beau"]),
]


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_normalisation(
    tok: LecturaTokeniseur, verbose: bool = False
) -> dict:
    """Évalue la normalisation sur le jeu de test."""
    total = len(_TEST_NORMALISATION)
    correct = 0
    errors: list[tuple[str, str, str]] = []

    for text_in, text_expected in _TEST_NORMALISATION:
        text_out = tok.normalize(text_in)
        if text_out == text_expected:
            correct += 1
        else:
            errors.append((text_in, text_expected, text_out))

    return {"total": total, "correct": correct, "errors": errors}


def evaluate_tokenisation(
    tok: LecturaTokeniseur, verbose: bool = False
) -> dict:
    """Évalue la tokenisation sur le jeu de test."""
    total = 0
    correct = 0
    phrase_correct = 0
    phrase_total = len(_TEST_TOKENISATION)
    errors: list[tuple[str, str, str]] = []

    for text_in, expected_tokens in _TEST_TOKENISATION:
        tokens = tok.tokenize(text_in)
        produced = [(t.text, t.type.value) for t in tokens]

        phrase_ok = True
        for i, (exp_text, exp_type) in enumerate(expected_tokens):
            total += 1
            if i < len(produced) and produced[i] == (exp_text, exp_type):
                correct += 1
            else:
                phrase_ok = False
                got = produced[i] if i < len(produced) else ("∅", "∅")
                errors.append((
                    f"[{text_in!r}] token {i}",
                    f"({exp_text!r}, {exp_type})",
                    f"({got[0]!r}, {got[1]})",
                ))

        # Vérifier aussi qu'il n'y a pas de tokens supplémentaires
        if len(produced) != len(expected_tokens):
            phrase_ok = False
            if len(produced) > len(expected_tokens):
                for j in range(len(expected_tokens), len(produced)):
                    errors.append((
                        f"[{text_in!r}] token {j} (extra)",
                        "∅",
                        f"({produced[j][0]!r}, {produced[j][1]})",
                    ))

        if phrase_ok:
            phrase_correct += 1

    return {
        "token_total": total,
        "token_correct": correct,
        "phrase_total": phrase_total,
        "phrase_correct": phrase_correct,
        "errors": errors,
    }


def evaluate_roundtrip(tok: LecturaTokeniseur) -> dict:
    """Évalue le round-trip : texte → tokens → reconstruction = texte normalisé."""
    texts = [t for t, _ in _TEST_NORMALISATION] + [t for t, _ in _TEST_TOKENISATION]
    total = 0
    correct = 0
    errors: list[tuple[str, str, str]] = []

    for text_in in texts:
        if not text_in:
            continue
        total += 1
        normalized = tok.normalize(text_in)
        tokens = tok.tokenize(text_in)
        reconstructed = "".join(t.text for t in tokens)
        if reconstructed == normalized:
            correct += 1
        else:
            errors.append((text_in, normalized, reconstructed))

    return {"total": total, "correct": correct, "errors": errors}


def evaluate_couverture(tok: LecturaTokeniseur) -> dict:
    """Évalue la couverture des cas spéciaux."""
    total = len(_TEST_COUVERTURE)
    correct = 0
    errors: list[tuple[str, str, str]] = []

    for text_in, description, expected_words in _TEST_COUVERTURE:
        result = tok.analyze(text_in)
        produced_words = [t.text for t in result.tokens if t.type == TokenType.MOT]
        if produced_words == expected_words:
            correct += 1
        else:
            errors.append((description, str(expected_words), str(produced_words)))

    return {"total": total, "correct": correct, "errors": errors}


# ══════════════════════════════════════════════════════════════════════════════
# Affichage
# ══════════════════════════════════════════════════════════════════════════════


def print_section(name: str, results: dict, verbose: bool = False) -> None:
    """Affiche les résultats d'une section."""
    if "token_total" in results:
        # Tokenisation
        tok_acc = results["token_correct"] / results["token_total"] if results["token_total"] else 0
        phr_acc = results["phrase_correct"] / results["phrase_total"] if results["phrase_total"] else 0
        print(f"\n  {name}")
        print(f"  {'-' * 50}")
        print(f"  Tokens corrects  : {tok_acc:.1%} ({results['token_correct']}/{results['token_total']})")
        print(f"  Phrases exactes  : {phr_acc:.1%} ({results['phrase_correct']}/{results['phrase_total']})")
    else:
        total = results["total"]
        correct = results["correct"]
        acc = correct / total if total else 0
        print(f"\n  {name}")
        print(f"  {'-' * 50}")
        print(f"  Correct : {acc:.1%} ({correct}/{total})")

    if verbose and results.get("errors"):
        print(f"\n  Erreurs :")
        for err in results["errors"][:15]:
            if len(err) == 3:
                print(f"    {err[0]}")
                print(f"      attendu : {err[1]}")
                print(f"      obtenu  : {err[2]}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation du Tokeniseur Lectura")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Afficher le détail des erreurs")
    args = parser.parse_args()

    tok = LecturaTokeniseur()

    print("=" * 60)
    print("  Évaluation — Lectura Tokeniseur v1.0")
    print("=" * 60)

    r_norm = evaluate_normalisation(tok, args.verbose)
    print_section("Normalisation", r_norm, args.verbose)

    r_tok = evaluate_tokenisation(tok, args.verbose)
    print_section("Tokenisation", r_tok, args.verbose)

    r_rt = evaluate_roundtrip(tok)
    print_section("Round-trip (tokens → texte)", r_rt, args.verbose)

    r_couv = evaluate_couverture(tok)
    print_section("Couverture cas spéciaux", r_couv, args.verbose)

    # Résumé
    total_tests = (r_norm["total"] + r_tok["token_total"]
                   + r_rt["total"] + r_couv["total"])
    total_ok = (r_norm["correct"] + r_tok["token_correct"]
                + r_rt["correct"] + r_couv["correct"])
    total_acc = total_ok / total_tests if total_tests else 0

    print(f"\n{'=' * 60}")
    print(f"  RÉSUMÉ")
    print(f"{'=' * 60}")
    print(f"  Score global : {total_acc:.1%} ({total_ok}/{total_tests})")
    print(f"  Normalisation : {r_norm['correct']}/{r_norm['total']}")
    print(f"  Tokenisation  : {r_tok['token_correct']}/{r_tok['token_total']}")
    print(f"  Round-trip    : {r_rt['correct']}/{r_rt['total']}")
    print(f"  Couverture    : {r_couv['correct']}/{r_couv['total']}")

    n_errors = (len(r_norm["errors"]) + len(r_tok["errors"])
                + len(r_rt["errors"]) + len(r_couv["errors"]))
    if n_errors > 0 and not args.verbose:
        print(f"\n  {n_errors} erreur(s) — relancer avec --verbose pour le détail")

    print()


if __name__ == "__main__":
    main()
