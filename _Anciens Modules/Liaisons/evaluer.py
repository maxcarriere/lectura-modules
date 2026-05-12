#!/usr/bin/env python3
"""Évaluation du module Liaisons Lectura sur un jeu de test intégré.

Évalue la classification des jonctions entre paires de mots :
  - Liaison grammaticale (obligatoire / facultative / interdite)
  - Enchaînement phonétique
  - Pas de liaison

Usage :
    python evaluer.py
    python evaluer.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

from lectura_liaisons import (
    LecturaLiaisons,
    LiaisonDecision,
    MotInfo,
)


# ══════════════════════════════════════════════════════════════════════════════
# Jeu de test intégré
# ══════════════════════════════════════════════════════════════════════════════

# Format : (mot1, mot2, kind_attendu, typ_attendu, latent_phoneme_attendu)
#   mot1/mot2 : (ortho, phone, [POS])
#   kind_attendu : "grammaticale" | "enchainement" | "none"
#   typ_attendu : "obligatoire" | "facultative" | "interdite" | "none"
#   latent_phoneme : "z" | "t" | "n" | "ʁ" | None


_TEST_DATA: list[tuple[
    tuple[str, str, list[str]],     # mot1
    tuple[str, str, list[str]],     # mot2
    str,                            # kind
    str,                            # typ
    str | None,                     # latent_phoneme
]] = [
    # ══════════════════════════════════════════════════════════════════════
    # LIAISONS OBLIGATOIRES
    # ══════════════════════════════════════════════════════════════════════

    # ART + NOM (voyelle)
    (("les", "le", ["ART:def"]), ("enfants", "ɑ̃fɑ̃", ["NOM"]),
     "grammaticale", "obligatoire", "z"),
    (("un", "œ̃", ["ART:ind"]), ("ami", "ami", ["NOM"]),
     "grammaticale", "obligatoire", "n"),
    (("des", "de", ["ART:ind"]), ("amis", "ami", ["NOM"]),
     "grammaticale", "obligatoire", "z"),
    (("les", "le", ["ART:def"]), ("arbres", "aʁbʁ", ["NOM"]),
     "grammaticale", "obligatoire", "z"),

    # ART + ADJ
    (("les", "le", ["ART:def"]), ("anciens", "ɑ̃sjɛ̃", ["ADJ"]),
     "grammaticale", "obligatoire", "z"),
    (("des", "de", ["ART:ind"]), ("énormes", "enɔʁm", ["ADJ"]),
     "grammaticale", "obligatoire", "z"),

    # PRO:per + VER
    (("nous", "nu", ["PRO:per"]), ("avons", "avɔ̃", ["VER"]),
     "grammaticale", "obligatoire", "z"),
    (("vous", "vu", ["PRO:per"]), ("êtes", "ɛt", ["AUX"]),
     "grammaticale", "obligatoire", "z"),
    (("elles", "ɛl", ["PRO:per"]), ("arrivent", "aʁiv", ["VER"]),
     "grammaticale", "obligatoire", "z"),

    # PRE + mot
    (("dans", "dɑ̃", ["PRE"]), ("un", "œ̃", ["ART:ind"]),
     "grammaticale", "obligatoire", "z"),
    (("sans", "sɑ̃", ["PRE"]), ("effort", "efɔʁ", ["NOM"]),
     "grammaticale", "obligatoire", "z"),
    (("chez", "ʃe", ["PRE"]), ("elle", "ɛl", ["PRO:per"]),
     "grammaticale", "obligatoire", "z"),
    (("en", "ɑ̃", ["PRE"]), ("avance", "avɑ̃s", ["NOM"]),
     "grammaticale", "obligatoire", "n"),

    # ADV + ADJ
    (("très", "tʁɛ", ["ADV"]), ("important", "ɛ̃pɔʁtɑ̃", ["ADJ"]),
     "grammaticale", "obligatoire", "z"),
    (("plus", "ply", ["ADV"]), ("utile", "ytil", ["ADJ"]),
     "grammaticale", "obligatoire", "z"),

    # ADJ + NOM
    (("petit", "pəti", ["ADJ"]), ("enfant", "ɑ̃fɑ̃", ["NOM"]),
     "grammaticale", "obligatoire", "t"),
    (("grand", "ɡʁɑ̃", ["ADJ"]), ("ami", "ami", ["NOM"]),
     "grammaticale", "obligatoire", "t"),
    (("gros", "ɡʁo", ["ADJ"]), ("ours", "uʁs", ["NOM"]),
     "grammaticale", "obligatoire", "z"),

    # "est" (toujours liaison)
    (("est", "ɛ", ["AUX"]), ("arrivé", "aʁive", ["VER"]),
     "grammaticale", "obligatoire", "t"),

    # VER/AUX + ADJ
    (("est", "ɛ", ["AUX"]), ("important", "ɛ̃pɔʁtɑ̃", ["ADJ"]),
     "grammaticale", "obligatoire", "t"),
    (("sont", "sɔ̃", ["AUX"]), ("arrivés", "aʁive", ["ADJ"]),
     "grammaticale", "obligatoire", "t"),

    # ══════════════════════════════════════════════════════════════════════
    # LIAISONS INTERDITES / ABSENTES
    # ══════════════════════════════════════════════════════════════════════

    # "et" — jamais de liaison (détecté comme interdite)
    (("et", "e", ["CON"]), ("alors", "alɔʁ", ["ADV"]),
     "grammaticale", "interdite", None),

    # h aspiré — bloque la liaison (module retourne none car h aspiré
    # empêche _starts_with_vowel_or_h_muet → pas de latent → none)
    (("les", "le", ["ART:def"]), ("haricots", "aʁiko", ["NOM"]),
     "none", "none", None),
    (("les", "le", ["ART:def"]), ("héros", "eʁo", ["NOM"]),
     "none", "none", None),

    # Mot2 commence par consonne — pas de liaison
    (("les", "le", ["ART:def"]), ("chats", "ʃa", ["NOM"]),
     "none", "none", None),
    (("un", "œ̃", ["ART:ind"]), ("chat", "ʃa", ["NOM"]),
     "none", "none", None),

    # "onze" — bloque la liaison
    (("les", "le", ["ART:def"]), ("onze", "ɔ̃z", ["NOM"]),
     "grammaticale", "interdite", None),

    # ══════════════════════════════════════════════════════════════════════
    # ENCHAÎNEMENTS
    # ══════════════════════════════════════════════════════════════════════

    # Consonne finale prononcée + voyelle suivante
    # "avec" (PRE) se termine par /k/ (consonne prononcée) → enchaînement
    (("avec", "avɛk", ["PRE"]), ("elle", "ɛl", ["PRO:per"]),
     "enchainement", "enchainement", None),

    # "il" se termine par /l/ (consonne prononcée) → enchaînement
    (("il", "il", ["PRO:per"]), ("arrive", "aʁiv", ["VER"]),
     "enchainement", "enchainement", None),

    # ══════════════════════════════════════════════════════════════════════
    # LIAISONS FACULTATIVES
    # ══════════════════════════════════════════════════════════════════════

    # NOM + ADJ (facultative)
    (("soldats", "sɔlda", ["NOM"]), ("anglais", "ɑ̃ɡlɛ", ["ADJ"]),
     "grammaticale", "facultative", "z"),

    # ══════════════════════════════════════════════════════════════════════
    # CAS PARTICULIERS — BON + voyelle (dénasalisation)
    # ══════════════════════════════════════════════════════════════════════

    (("bon", "bɔ̃", ["ADJ"]), ("ami", "ami", ["NOM"]),
     "grammaticale", "obligatoire", "n"),

    # ══════════════════════════════════════════════════════════════════════
    # MOTS INVARIABLES ET ADVERBES
    # ══════════════════════════════════════════════════════════════════════

    (("tout", "tu", ["ADV"]), ("entier", "ɑ̃tje", ["ADJ"]),
     "grammaticale", "obligatoire", "t"),
    (("moins", "mwɛ̃", ["ADV"]), ("important", "ɛ̃pɔʁtɑ̃", ["ADJ"]),
     "grammaticale", "obligatoire", "z"),

    # ══════════════════════════════════════════════════════════════════════
    # CORRECTIONS PFC
    # ══════════════════════════════════════════════════════════════════════

    # on + VER → obligatoire /n/
    (("on", "ɔ̃", ["PRO:per"]), ("a", "a", ["AUX"]),
     "grammaticale", "obligatoire", "n"),

    # ils + VER → obligatoire /z/ (consonant-final phone)
    (("ils", "il", ["PRO:per"]), ("ont", "ɔ̃", ["AUX"]),
     "grammaticale", "obligatoire", "z"),

    # quand + voyelle → obligatoire /t/
    (("quand", "kɑ̃", ["CON"]), ("il", "il", ["PRO:per"]),
     "grammaticale", "obligatoire", "t"),

    # NUM + NOM → obligatoire /z/
    (("deux", "dø", ["NUM"]), ("amis", "ami", ["NOM"]),
     "grammaticale", "obligatoire", "z"),
    (("trois", "tʁwa", ["NUM"]), ("enfants", "ɑ̃fɑ̃", ["NOM"]),
     "grammaticale", "obligatoire", "z"),

    # est + INTJ → obligatoire /t/ (always-liaison, 0% non-réalisation PFC)
    (("est", "ɛ", ["AUX"]), ("euh", "ø", ["INTJ"]),
     "grammaticale", "obligatoire", "t"),

    # ADV polysyllabique → pas de liaison obligatoire
    (("vraiment", "vʁɛmɑ̃", ["ADV"]), ("utile", "ytil", ["ADJ"]),
     "none", "none", None),

    # ══════════════════════════════════════════════════════════════════════
    # ALWAYS-LIAISON (court-circuit POS)
    # ══════════════════════════════════════════════════════════════════════

    # les + amis → obligatoire /z/ même avec POS vide
    (("les", "le", []), ("amis", "ami", []),
     "grammaticale", "obligatoire", "z"),

    # on + arrive → obligatoire /n/ même avec POS erroné (NOM)
    (("on", "ɔ̃", ["NOM"]), ("arrive", "aʁiv", ["VER"]),
     "grammaticale", "obligatoire", "n"),

    # les + haricots → none (h aspiré prioritaire)
    (("les", "le", []), ("haricots", "aʁiko", ["NOM"]),
     "none", "none", None),

    # les + onze → interdite (onze prioritaire)
    (("les", "le", []), ("onze", "ɔ̃z", ["NOM"]),
     "grammaticale", "interdite", None),

    # est + en → obligatoire /t/ (même sans POS mot2 compatible)
    (("est", "ɛ", []), ("en", "ɑ̃", ["PRE"]),
     "grammaticale", "obligatoire", "t"),

    # suis + allée → obligatoire /z/
    (("suis", "sɥi", []), ("allée", "ale", ["VER"]),
     "grammaticale", "obligatoire", "z"),

    # ont + eu → obligatoire /t/
    (("ont", "ɔ̃", []), ("eu", "y", ["VER"]),
     "grammaticale", "obligatoire", "t"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(lia: LecturaLiaisons, verbose: bool = False) -> dict:
    """Évalue le module Liaisons sur le jeu de test intégré."""
    total = len(_TEST_DATA)
    correct_kind = 0
    correct_typ = 0
    correct_latent = 0
    correct_full = 0
    errors: list[tuple[str, str, dict, dict]] = []

    for mot1_data, mot2_data, exp_kind, exp_typ, exp_latent in _TEST_DATA:
        w1 = MotInfo(ortho=mot1_data[0], phone=mot1_data[1], pos=mot1_data[2])
        w2 = MotInfo(ortho=mot2_data[0], phone=mot2_data[1], pos=mot2_data[2])

        decision = lia.classify(w1, w2)

        got_kind = decision.kind
        got_typ = decision.typ
        got_latent = decision.latent_phoneme

        kind_ok = got_kind == exp_kind
        typ_ok = got_typ == exp_typ
        latent_ok = got_latent == exp_latent

        if kind_ok:
            correct_kind += 1
        if typ_ok:
            correct_typ += 1
        if latent_ok:
            correct_latent += 1
        if kind_ok and typ_ok and latent_ok:
            correct_full += 1
        else:
            pair_str = f"{mot1_data[0]} + {mot2_data[0]}"
            pos_str = f"{mot1_data[2]} + {mot2_data[2]}"
            expected = {"kind": exp_kind, "typ": exp_typ, "latent": exp_latent}
            got = {"kind": got_kind, "typ": got_typ, "latent": got_latent}
            errors.append((pair_str, pos_str, expected, got))

    return {
        "total": total,
        "correct_full": correct_full,
        "correct_kind": correct_kind,
        "correct_typ": correct_typ,
        "correct_latent": correct_latent,
        "accuracy_full": correct_full / total if total else 0,
        "accuracy_kind": correct_kind / total if total else 0,
        "accuracy_typ": correct_typ / total if total else 0,
        "errors": errors,
    }


def print_results(results: dict, verbose: bool = False) -> None:
    """Affiche les résultats."""
    print(f"\n{'=' * 60}")
    print(f"  Évaluation — Lectura Liaisons v1.0")
    print(f"{'=' * 60}")

    print(f"\n  Classification complète (kind+typ+latent) : "
          f"{results['accuracy_full']:.1%}"
          f"  ({results['correct_full']}/{results['total']})")
    print(f"  Kind correct (gram/ench/none)             : "
          f"{results['accuracy_kind']:.1%}"
          f"  ({results['correct_kind']}/{results['total']})")
    print(f"  Type correct (oblig/fac/int/none)         : "
          f"{results['accuracy_typ']:.1%}"
          f"  ({results['correct_typ']}/{results['total']})")

    n_errors = len(results["errors"])
    print(f"\n  Erreurs : {n_errors}")

    if verbose and results["errors"]:
        print(f"\n  {'Paire':<25} {'POS':<30} {'Attendu':<30} {'Obtenu'}")
        print(f"  {'-' * 110}")
        for pair_str, pos_str, expected, got in results["errors"][:30]:
            exp_str = f"{expected['kind']}/{expected['typ']}/{expected['latent']}"
            got_str = f"{got['kind']}/{got['typ']}/{got['latent']}"
            print(f"  {pair_str:<25} {pos_str:<30} {exp_str:<30} {got_str}")

    if n_errors > 0 and not verbose:
        print(f"\n  Relancer avec --verbose pour le détail des erreurs")

    # Répartition par catégorie
    cats: dict[str, list[str]] = {
        "obligatoire": [], "facultative": [], "interdite": [],
        "enchainement": [], "none": [],
    }
    for mot1_data, mot2_data, exp_kind, exp_typ, _ in _TEST_DATA:
        cat = exp_typ if exp_kind == "grammaticale" else exp_kind
        cats.setdefault(cat, [])
        cats[cat].append(f"{mot1_data[0]}+{mot2_data[0]}")

    print(f"\n  Répartition du jeu de test :")
    for cat, items in cats.items():
        if items:
            print(f"    {cat:<15} : {len(items)} paires")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation du module Liaisons Lectura")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Afficher le détail des erreurs")
    args = parser.parse_args()

    lia = LecturaLiaisons()
    results = evaluate(lia, args.verbose)
    print_results(results, args.verbose)
    print()


if __name__ == "__main__":
    main()
