"""Tests pour les regles de ponctuation."""

from lectura_correcteur._types import TypeCorrection
from lectura_correcteur.syntaxe._ponctuation import verifier_ponctuation


def test_majuscule_debut_phrase():
    """Le premier mot doit avoir une majuscule."""
    tokens = ["bonjour", ",", "le", "chat"]
    corrections = verifier_ponctuation(tokens)
    assert tokens[0] == "Bonjour"
    # 1 majuscule + 1 point final
    assert len(corrections) == 2
    assert corrections[0].type_correction == TypeCorrection.SYNTAXE
    assert corrections[0].explication == "Majuscule en debut de phrase"
    assert corrections[1].corrige == "."


def test_majuscule_apres_point():
    """Apres un point, le mot suivant doit avoir une majuscule."""
    tokens = ["Fin", ".", "debut"]
    corrections = verifier_ponctuation(tokens)
    assert tokens[2] == "Debut"
    # 1 majuscule + 1 point final (dernier token etait "debut" pas ".")
    assert len(corrections) == 2


def test_deja_majuscule():
    """Un mot deja en majuscule ne doit pas generer de correction."""
    tokens = ["Bonjour", ".", "Le", "chat"]
    corrections = verifier_ponctuation(tokens)
    # Seulement le point final (pas de correction majuscule)
    assert len(corrections) == 1
    assert corrections[0].corrige == "."


def test_ponctuation_non_alpha():
    """Un token non-alpha apres un point ne doit pas etre modifie."""
    tokens = ["Fin", ".", "123"]
    corrections = verifier_ponctuation(tokens)
    assert tokens[2] == "123"
    # Seulement le point final (123 n'est pas ponctuation terminale)
    assert len(corrections) == 1
    assert corrections[0].corrige == "."


def test_pas_de_point_si_deja_present():
    """Si la phrase se termine par un point, pas de point ajoute."""
    tokens = ["Bonjour", "."]
    corrections = verifier_ponctuation(tokens)
    assert len(corrections) == 0
    assert tokens == ["Bonjour", "."]


def test_pas_de_point_si_exclamation():
    """Si la phrase se termine par !, pas de point ajoute."""
    tokens = ["Bonjour", "!"]
    corrections = verifier_ponctuation(tokens)
    assert len(corrections) == 0


def test_pas_de_point_si_interrogation():
    """Si la phrase se termine par ?, pas de point ajoute."""
    tokens = ["Quoi", "?"]
    corrections = verifier_ponctuation(tokens)
    assert len(corrections) == 0
