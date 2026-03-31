"""Tests pour les regles de conjugaison."""

from lectura_correcteur.grammaire._conjugaison import verifier_conjugaisons


def test_ils_mange_ent(mock_lexique):
    """'ils mange' -> 'ils mangent'."""
    mots = ["ils", "mange"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "mange"],
    )
    assert result[1] == "mangent"
    assert len(corrections) >= 1


def test_elles_mange_ent(mock_lexique):
    """'elles mange' -> 'elles mangent'."""
    mots = ["elles", "mange"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["elles", "mange"],
    )
    assert result[1] == "mangent"


def test_pas_de_correction_si_deja_correct(mock_lexique):
    """'ils mangent' -> pas de correction."""
    mots = ["ils", "mangent"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "mangent"],
    )
    assert result == ["ils", "mangent"]
