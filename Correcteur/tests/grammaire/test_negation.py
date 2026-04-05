"""Tests pour les regles de negation."""

from lectura_correcteur.grammaire._negation import verifier_negation


def test_mange_pas_ne_mange_pas(mock_lexique):
    """'je mange pas' -> 'je ne mange pas'."""
    mots = ["je", "mange", "pas"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["je", "ne", "mange", "pas"]
    assert len(corrections) == 1


def test_dort_plus(mock_lexique):
    """'il dort plus' -> 'il ne dort plus'."""
    mots = ["il", "dort", "plus"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "dort", "plus"]
    assert len(corrections) == 1


def test_mange_jamais(mock_lexique):
    """'il mange jamais' -> 'il ne mange jamais'."""
    mots = ["il", "mange", "jamais"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "mange", "jamais"]
    assert len(corrections) == 1


def test_mange_rien(mock_lexique):
    """'il mange rien' -> 'il ne mange rien'."""
    mots = ["il", "mange", "rien"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "mange", "rien"]
    assert len(corrections) == 1


def test_pas_correction_si_ne_present(mock_lexique):
    """'il ne mange pas' -> pas de correction."""
    mots = ["il", "ne", "mange", "pas"]
    pos = ["PRO:per", "ADV", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "mange", "pas"]
    assert len(corrections) == 0
