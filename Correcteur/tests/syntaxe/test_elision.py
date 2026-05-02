"""Tests pour l'elision."""

from lectura_correcteur.syntaxe._elision import appliquer_elision


def test_elision_que_il():
    """'que il' -> ["qu'", "il"] (2 tokens)."""
    tokens = ["parce", "que", "il"]
    result = appliquer_elision(tokens)
    assert result == ["parce", "qu'", "il"]


def test_elision_je_arrive():
    """'je arrive' -> ["j'", "arrive"] (2 tokens)."""
    tokens = ["je", "arrive"]
    result = appliquer_elision(tokens)
    assert result == ["j'", "arrive"]


def test_elision_de_un():
    """'de un' -> ["d'", "un"] (2 tokens)."""
    tokens = ["de", "un"]
    result = appliquer_elision(tokens)
    assert result == ["d'", "un"]


def test_elision_se_est():
    """'se est' -> ["s'", "est"] (2 tokens)."""
    tokens = ["se", "est"]
    result = appliquer_elision(tokens)
    assert result == ["s'", "est"]


def test_elision_la_ecole():
    """'la ecole' -> ["l'", "ecole"] (2 tokens)."""
    tokens = ["la", "ecole"]
    result = appliquer_elision(tokens)
    assert result == ["l'", "ecole"]


def test_pas_elision_devant_consonne():
    """'le chat' reste 'le chat' (pas d'elision devant consonne)."""
    tokens = ["le", "chat"]
    result = appliquer_elision(tokens)
    assert result == ["le", "chat"]


def test_elision_ne_a():
    """'ne a' -> ["n'", "a"] (2 tokens)."""
    tokens = ["ne", "a"]
    result = appliquer_elision(tokens)
    assert result == ["n'", "a"]


def test_pas_elision_mot_non_elidable():
    """'si il' reste 'si il' (si n'est pas elidable)."""
    tokens = ["si", "il"]
    result = appliquer_elision(tokens)
    assert result == ["si", "il"]
