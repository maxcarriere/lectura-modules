"""Tests pour les fonctions de recherche."""

from pathlib import Path

import pytest

from lectura_lexique import Lexique

DONNEES_DIR = Path(__file__).parent / "donnees"
TEST_CSV = DONNEES_DIR / "test.csv"


@pytest.fixture(scope="module")
def lexique():
    with Lexique(TEST_CSV) as lex:
        yield lex


def test_rechercher_regex_ortho(lexique):
    """rechercher('^man') trouve mange, manges, etc."""
    results = lexique.rechercher("^man")
    orthos = {r["ortho"].lower() for r in results}
    assert any(o.startswith("man") for o in orthos)
    assert len(results) > 0


def test_rechercher_regex_phone(lexique):
    """rechercher sur phone."""
    results = lexique.rechercher("ʃa", champ="phone")
    orthos = {r["ortho"].lower() for r in results}
    assert "chat" in orthos or "chas" in orthos


def test_rechercher_regex_invalide(lexique):
    """Un pattern regex invalide retourne []."""
    assert lexique.rechercher("[invalid") == []


def test_filtrer_nom(lexique):
    """filtrer(cgram='NOM') retourne des noms."""
    results = lexique.filtrer(cgram="NOM")
    assert len(results) > 0
    for r in results:
        assert r["cgram"].startswith("NOM")


def test_filtrer_genre_m(lexique):
    """filtrer(genre='m') retourne des mots masculins."""
    results = lexique.filtrer(genre="m")
    assert len(results) > 0
    for r in results:
        assert r.get("genre") == "m"


def test_filtrer_nombre_p(lexique):
    """filtrer(nombre='p') retourne des mots pluriels."""
    results = lexique.filtrer(nombre="p")
    assert len(results) > 0
    for r in results:
        assert r.get("nombre") == "p"


def test_filtrer_freq_min(lexique):
    """filtrer(freq_min=100) retourne des mots frequents."""
    results = lexique.filtrer(freq_min=100)
    assert len(results) > 0
    for r in results:
        assert float(r.get("freq", 0)) >= 100


def test_filtrer_combinaison(lexique):
    """filtrer multi-critere."""
    results = lexique.filtrer(cgram="NOM", genre="f", nombre="s")
    assert len(results) > 0
    for r in results:
        assert r["cgram"].startswith("NOM")
        assert r.get("genre") == "f"
        assert r.get("nombre") == "s"


def test_anagrammes_rime(lexique):
    """anagrammes('rime') trouve 'mire'."""
    results = lexique.anagrammes("rime")
    orthos = {r["ortho"].lower() for r in results}
    assert "mire" in orthos


def test_anagrammes_exclut_source(lexique):
    """anagrammes n'inclut pas le mot source."""
    results = lexique.anagrammes("rime")
    orthos = {r["ortho"].lower() for r in results}
    assert "rime" not in orthos


def test_anagrammes_inexistant(lexique):
    """anagrammes d'un mot sans anagramme retourne []."""
    results = lexique.anagrammes("dans")
    assert results == []
