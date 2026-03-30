#!/usr/bin/env python3
"""Évaluation du Syllabeur Lectura sur un jeu de test intégré.

Évalue la syllabation IPA (sans phonémiseur) sur des mots français
dont la syllabation de référence est connue.

Usage :
    python evaluer.py
    python evaluer.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

from lectura_syllabeur import LecturaSyllabeur


# ══════════════════════════════════════════════════════════════════════════════
# Jeu de test intégré
# ══════════════════════════════════════════════════════════════════════════════

# Format : (mot, IPA, syllabes_IPA_attendues)
# La phonémisation est fournie directement (pas besoin d'eSpeak-NG).
# Sources : dictionnaires de référence (Lexique, GLAFF, manuels de phonétique).

_TEST_DATA: list[tuple[str, str, list[str]]] = [
    # ── Mots simples (1-2 syllabes) ──
    ("chat", "ʃa", ["ʃa"]),
    ("bon", "bɔ̃", ["bɔ̃"]),
    ("mer", "mɛʁ", ["mɛʁ"]),
    ("eau", "o", ["o"]),
    ("oui", "wi", ["wi"]),
    ("pain", "pɛ̃", ["pɛ̃"]),
    ("lait", "lɛ", ["lɛ"]),
    ("jour", "ʒuʁ", ["ʒuʁ"]),
    ("nuit", "nɥi", ["nɥi"]),
    ("feu", "fø", ["fø"]),

    # ── Mots 2 syllabes ──
    ("maison", "mɛzɔ̃", ["mɛ", "zɔ̃"]),
    ("bonjour", "bɔ̃ʒuʁ", ["bɔ̃", "ʒuʁ"]),
    ("papa", "papa", ["pa", "pa"]),
    ("maman", "mamɑ̃", ["ma", "mɑ̃"]),
    ("enfant", "ɑ̃fɑ̃", ["ɑ̃", "fɑ̃"]),
    ("soleil", "sɔlɛj", ["sɔ", "lɛj"]),
    ("jardin", "ʒaʁdɛ̃", ["ʒaʁ", "dɛ̃"]),
    ("fenêtre", "fənɛtʁ", ["fə", "nɛtʁ"]),
    ("lever", "ləve", ["lə", "ve"]),
    ("petit", "pəti", ["pə", "ti"]),

    # ── Mots 3 syllabes ──
    ("chocolat", "ʃɔkɔla", ["ʃɔ", "kɔ", "la"]),
    ("animal", "animal", ["a", "ni", "mal"]),
    ("avenue", "avəny", ["a", "və", "ny"]),
    ("cinéma", "sinema", ["si", "ne", "ma"]),
    ("abricot", "abʁiko", ["a", "bʁi", "ko"]),
    ("consulat", "kɔ̃syla", ["kɔ̃", "sy", "la"]),
    ("médecin", "medsɛ̃", ["me", "dsɛ̃"]),
    ("éléphant", "elefɑ̃", ["e", "le", "fɑ̃"]),
    ("parapluie", "paʁaplɥi", ["pa", "ʁa", "plɥi"]),
    ("papillon", "papijɔ̃", ["pa", "pi", "jɔ̃"]),

    # ── Mots 4+ syllabes ──
    ("université", "ynivɛʁsite", ["y", "ni", "vɛʁ", "si", "te"]),
    ("extraordinaire", "ɛkstʁaɔʁdinɛʁ", ["ɛk", "stʁa", "ɔʁ", "di", "nɛʁ"]),
    ("température", "tɑ̃peʁatyʁ", ["tɑ̃", "pe", "ʁa", "tyʁ"]),
    ("communication", "kɔmynikasjɔ̃", ["kɔ", "my", "ni", "ka", "sjɔ̃"]),
    ("informatique", "ɛ̃fɔʁmatik", ["ɛ̃", "fɔʁ", "ma", "tik"]),
    ("restaurant", "ʁɛstoʁɑ̃", ["ʁɛs", "to", "ʁɑ̃"]),
    ("automobile", "otomɔbil", ["o", "to", "mɔ", "bil"]),

    # ── Clusters consonantiques (attaques complexes) ──
    ("train", "tʁɛ̃", ["tʁɛ̃"]),
    ("classe", "klas", ["klas"]),
    ("plat", "pla", ["pla"]),
    ("bras", "bʁa", ["bʁa"]),
    ("sport", "spɔʁ", ["spɔʁ"]),
    ("strict", "stʁikt", ["stʁikt"]),
    ("spectacle", "spɛktakl", ["spɛk", "takl"]),
    ("plastique", "plastik", ["plas", "tik"]),
    ("problème", "pʁɔblɛm", ["pʁɔ", "blɛm"]),
    ("structure", "stʁyktyʁ", ["stʁyk", "tyʁ"]),

    # ── Semi-voyelles et diphtongues ──
    ("pied", "pje", ["pje"]),
    ("lui", "lɥi", ["lɥi"]),
    ("loi", "lwa", ["lwa"]),
    ("jouer", "ʒwe", ["ʒwe"]),
    ("nuit", "nɥi", ["nɥi"]),
    ("rien", "ʁjɛ̃", ["ʁjɛ̃"]),
    ("lion", "ljɔ̃", ["ljɔ̃"]),
    ("mouette", "mwɛt", ["mwɛt"]),
    ("alouette", "alwɛt", ["a", "lwɛt"]),
    ("fouet", "fwɛ", ["fwɛ"]),

    # ── Voyelles nasales ──
    ("chanson", "ʃɑ̃sɔ̃", ["ʃɑ̃", "sɔ̃"]),
    ("menton", "mɑ̃tɔ̃", ["mɑ̃", "tɔ̃"]),
    ("parfum", "paʁfœ̃", ["paʁ", "fœ̃"]),
    ("lundi", "lœ̃di", ["lœ̃", "di"]),
    ("peinture", "pɛ̃tyʁ", ["pɛ̃", "tyʁ"]),
    ("invention", "ɛ̃vɑ̃sjɔ̃", ["ɛ̃", "vɑ̃", "sjɔ̃"]),

    # ── Hiatus ──
    ("chaos", "kao", ["ka", "o"]),
    ("aéré", "aeʁe", ["a", "e", "ʁe"]),
    ("poète", "pɔɛt", ["pɔ", "ɛt"]),
    ("naïf", "naif", ["na", "if"]),
    ("pays", "pei", ["pe", "i"]),

    # ── Codas complexes ──
    ("arbre", "aʁbʁ", ["aʁbʁ"]),
    ("monstre", "mɔ̃stʁ", ["mɔ̃stʁ"]),
    ("texte", "tɛkst", ["tɛkst"]),
    ("ongle", "ɔ̃ɡl", ["ɔ̃ɡl"]),

    # ── Mots courants divers ──
    ("France", "fʁɑ̃s", ["fʁɑ̃s"]),
    ("musique", "myzik", ["my", "zik"]),
    ("école", "ekɔl", ["e", "kɔl"]),
    ("hôpital", "opital", ["o", "pi", "tal"]),
    ("bibliothèque", "biblijɔtɛk", ["bi", "bli", "jɔ", "tɛk"]),
    ("philosophie", "filɔzɔfi", ["fi", "lɔ", "zɔ", "fi"]),
    ("république", "ʁepyblik", ["ʁe", "py", "blik"]),
    ("fromage", "fʁɔmaʒ", ["fʁɔ", "maʒ"]),
    ("chocolatier", "ʃɔkɔlatje", ["ʃɔ", "kɔ", "la", "tje"]),
    ("ordinateur", "ɔʁdinatœʁ", ["ɔʁ", "di", "na", "tœʁ"]),
]


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(syl: LecturaSyllabeur, verbose: bool = False) -> dict:
    """Évalue le Syllabeur sur le jeu de test intégré."""
    total = len(_TEST_DATA)
    correct_count = 0
    correct_boundaries = 0
    total_boundaries = 0
    errors: list[tuple[str, str, list[str], list[str]]] = []

    for mot, ipa, expected_sylls in _TEST_DATA:
        produced = syl.syllabify_ipa(ipa)

        # Nombre de syllabes correct ?
        count_ok = len(produced) == len(expected_sylls)

        # Frontières correctes (comparaison exacte)
        boundary_ok = produced == expected_sylls

        if boundary_ok:
            correct_count += 1
            correct_boundaries += max(len(expected_sylls) - 1, 0)
        else:
            # Compter les frontières correctes partiellement
            # On compare les positions de coupure
            exp_boundaries = _get_boundaries(expected_sylls)
            prod_boundaries = _get_boundaries(produced)
            for b in exp_boundaries:
                if b in prod_boundaries:
                    correct_boundaries += 1

            errors.append((mot, ipa, expected_sylls, produced))

        total_boundaries += max(len(expected_sylls) - 1, 0)

    return {
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0,
        "boundary_total": total_boundaries,
        "boundary_correct": correct_boundaries,
        "boundary_accuracy": (correct_boundaries / total_boundaries
                              if total_boundaries else 0),
        "errors": errors,
    }


def _get_boundaries(sylls: list[str]) -> list[int]:
    """Retourne les positions (en phonèmes) des frontières syllabiques."""
    boundaries: list[int] = []
    pos = 0
    for s in sylls[:-1]:
        pos += len(s)
        boundaries.append(pos)
    return boundaries


def print_results(results: dict, verbose: bool = False) -> None:
    """Affiche les résultats."""
    print(f"\n{'=' * 60}")
    print(f"  Évaluation — Lectura Syllabeur v1.0")
    print(f"{'=' * 60}")

    print(f"\n  Mots corrects (syllabation exacte) : {results['accuracy']:.1%}"
          f"  ({results['correct']}/{results['total']})")
    print(f"  Frontières syllabiques correctes   : {results['boundary_accuracy']:.1%}"
          f"  ({results['boundary_correct']}/{results['boundary_total']})")

    # Catégories
    cats = {
        "1 syllabe": [], "2 syllabes": [], "3 syllabes": [],
        "4+ syllabes": [], "clusters": [], "semi-voyelles": [],
        "nasales": [], "hiatus": [], "codas": [], "divers": [],
    }
    for mot, ipa, expected, produced in results["errors"]:
        n = len(expected)
        if n == 1:
            cats["1 syllabe"].append(mot)
        elif n == 2:
            cats["2 syllabes"].append(mot)
        elif n == 3:
            cats["3 syllabes"].append(mot)
        else:
            cats["4+ syllabes"].append(mot)

    n_errors = len(results["errors"])
    print(f"\n  Erreurs : {n_errors}")

    if verbose and results["errors"]:
        print(f"\n  {'Mot':<20} {'IPA':<18} {'Attendu':<25} {'Obtenu':<25}")
        print(f"  {'-' * 88}")
        for mot, ipa, expected, produced in results["errors"][:30]:
            exp_str = ".".join(expected)
            prod_str = ".".join(produced)
            print(f"  {mot:<20} {ipa:<18} {exp_str:<25} {prod_str:<25}")

    if n_errors > 0 and not verbose:
        print(f"\n  Relancer avec --verbose pour le détail des erreurs")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation du Syllabeur Lectura")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Afficher le détail des erreurs")
    args = parser.parse_args()

    syl = LecturaSyllabeur()
    results = evaluate(syl, args.verbose)
    print_results(results, args.verbose)
    print()


if __name__ == "__main__":
    main()
