"""Tests pour le verificateur orthographique."""

from lectura_correcteur._types import TypeCorrection
from lectura_correcteur.orthographe._verificateur import VerificateurOrthographe


def test_mot_connu(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["chat"])
    assert len(results) == 1
    assert results[0].dans_lexique is True
    assert results[0].type_correction == TypeCorrection.AUCUNE


def test_mot_inconnu(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["xyzabc"])
    assert len(results) == 1
    assert results[0].dans_lexique is False
    assert results[0].type_correction == TypeCorrection.HORS_LEXIQUE


def test_phrase_mixte(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["le", "chat", "manjer"])
    assert results[0].dans_lexique is True
    assert results[1].dans_lexique is True
    assert results[2].dans_lexique is False
    assert results[2].type_correction == TypeCorrection.HORS_LEXIQUE


def test_casse_insensible(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["Chat"])
    assert results[0].dans_lexique is True


def test_avec_morpho(mock_lexique):
    v = VerificateurOrthographe(mock_lexique)
    morpho = [{"pos": "NOM", "genre": "Masc", "nombre": "Sing"}]
    results = v.verifier_phrase(["chat"], morpho)
    assert results[0].pos == "NOM"
    assert results[0].morpho.get("genre") == "Masc"
