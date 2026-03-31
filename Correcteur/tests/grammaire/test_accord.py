"""Tests pour les regles d'accord."""

from lectura_correcteur._types import TypeCorrection
from lectura_correcteur.grammaire._accord import verifier_accords


def test_pluriel_nom_apres_det(mock_lexique):
    """'les enfant' -> 'les enfants'."""
    mots = ["les", "enfant"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["les", "enfants"]
    assert len(corrections) == 1
    assert corrections[0].type_correction == TypeCorrection.GRAMMAIRE


def test_pluriel_deja_present(mock_lexique):
    """'les enfants' -> pas de correction."""
    mots = ["les", "enfants"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["les", "enfants"]
    assert len(corrections) == 0


def test_restaurer_ils(mock_lexique):
    """'il' (quand original etait 'ils') -> 'ils'."""
    mots = ["il", "mange"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique, originaux=["ils", "mange"],
    )
    assert result[0].lower() == "ils"


def test_regle4_det_nom_ver(mock_lexique):
    """'les enfants mange' -> 'les enfants mangent'."""
    mots = ["les", "enfants", "mange"]
    pos = ["ART:def", "NOM", "VER"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["les", "enfants", "mangent"]


def test_invariable_pas_modifie(mock_lexique):
    """Les mots invariables ne prennent pas de -s."""
    mots = ["les", "chose"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_accords(mots, pos, {}, mock_lexique)
    assert result == ["les", "chose"]
