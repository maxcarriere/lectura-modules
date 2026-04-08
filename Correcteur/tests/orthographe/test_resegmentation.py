"""Tests pour la resegmentation."""

from lectura_correcteur.orthographe._resegmentation import resegmenter


def test_narrive_split(mock_lexique):
    """'narrive' devrait etre resegmente en ["n'", "arrive"]."""
    tokens = ["narrive"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["n'", "arrive"]


def test_mot_connu_pas_resegmente(mock_lexique):
    """Un mot connu du lexique ne doit pas etre resegmente."""
    tokens = ["chat"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["chat"]


def test_mot_court_pas_resegmente(mock_lexique):
    """Un token de moins de 3 caracteres ne doit pas etre resegmente."""
    tokens = ["ab"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["ab"]


def test_pas_de_faux_positif(mock_lexique):
    """Un mot inconnu qui ne commence pas par un clitique reste tel quel."""
    tokens = ["xyzabc"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["xyzabc"]


def test_lhomme_split(mock_lexique):
    """'lhomme' -> ["l'", "homme"] via split elargi (consonne h)."""
    tokens = ["lhomme"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["l'", "homme"]


def test_cest_split(mock_lexique):
    """'cest' -> ["c'", "est"] via split elargi."""
    tokens = ["cest"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["c'", "est"]


def test_quil_split(mock_lexique):
    """'quil' -> ["qu'", "il"] via split elargi."""
    tokens = ["quil"]
    result = resegmenter(tokens, mock_lexique)
    assert result == ["qu'", "il"]
