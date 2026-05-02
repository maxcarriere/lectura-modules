"""Tests pour les suggestions orthographiques."""

import pytest

from lectura_correcteur.orthographe._suggestions import suggerer, _est_variante_accent


@pytest.fixture
def lexique_accent(mock_lexique):
    """MockLexique enrichi avec des formes accentuees et ligatures."""
    from tests.conftest import MockLexique

    formes = mock_lexique._formes.copy()
    formes.update({
        "château": [{"ortho": "château", "cgram": "NOM", "freq": 25.0,
                     "genre": "m", "nombre": "s"}],
        "châteaux": [{"ortho": "châteaux", "cgram": "NOM", "freq": 10.0,
                      "genre": "m", "nombre": "p"}],
        "chapeau": [{"ortho": "chapeau", "cgram": "NOM", "freq": 30.0,
                     "genre": "m", "nombre": "s"}],
        "sœur": [{"ortho": "sœur", "cgram": "NOM", "freq": 25.0,
                  "genre": "f", "nombre": "s"}],
        "sœurs": [{"ortho": "sœurs", "cgram": "NOM", "freq": 10.0,
                   "genre": "f", "nombre": "p"}],
        "tôt": [{"ortho": "tôt", "cgram": "ADV", "freq": 20.0}],
        "tout": [{"ortho": "tout", "cgram": "ADV", "freq": 80.0}],
        "été": [{"ortho": "été", "cgram": "NOM", "freq": 30.0,
                 "genre": "m", "nombre": "s"}],
    })
    return MockLexique(formes=formes)


def test_est_variante_accent():
    """_est_variante_accent detecte les variantes accent-only."""
    assert _est_variante_accent("chateau", "château") is True
    assert _est_variante_accent("tot", "tôt") is True
    assert _est_variante_accent("ete", "été") is True
    assert _est_variante_accent("chateau", "chapeau") is False
    assert _est_variante_accent("chat", "chats") is False


def test_accent_priorite(lexique_accent):
    """'chateau' -> 'château' en premier, pas 'chapeau'."""
    results = suggerer("chateau", lexique_accent)
    assert len(results) > 0
    assert results[0] == "château"


def test_accent_priorite_tot(lexique_accent):
    """'tot' -> 'tôt' en premier, pas 'tout'."""
    results = suggerer("tot", lexique_accent)
    assert len(results) > 0
    assert results[0] == "tôt"


def test_ligature_oe(lexique_accent):
    """'soeur' -> 'sœur' (digraphe oe -> ligature œ comme d=1)."""
    results = suggerer("soeur", lexique_accent)
    assert "sœur" in results


# --- Phase 5 : G2P phonetique ---

class MockG2P:
    """G2P mock qui retourne une prononciation predeterminee."""
    def __init__(self, table: dict[str, str]):
        self._table = table

    def prononcer(self, mot: str) -> str | None:
        return self._table.get(mot.lower())


def test_g2p_phase5_homophone(mock_lexique):
    """Phase 5 : un mot hors-lexique dont le phone correspond a un mot connu."""
    # "fome" n'existe pas, mais sa prononciation "fam" correspond a "femme"
    g2p = MockG2P({"fome": "fam"})
    results = suggerer("fome", mock_lexique, g2p=g2p)
    assert "femme" in results


def test_g2p_phase5_sans_g2p(mock_lexique):
    """Sans G2P, phase 5 n'est pas activee (pas d'erreur)."""
    results = suggerer("fome", mock_lexique, g2p=None)
    # Pas de crash, juste pas de resultat G2P
    assert isinstance(results, list)


def test_g2p_phase5_pas_de_doublon(mock_lexique):
    """Phase 5 ne duplique pas les candidats deja trouves en phases 1-4."""
    # "chta" -> d=1 -> "chat" ; G2P dit phone "ʃa" -> homophone "chat" aussi
    g2p = MockG2P({"chta": "\u0283a"})
    results = suggerer("chta", mock_lexique, g2p=g2p)
    assert results.count("chat") <= 1
