"""Tests pour les fonctions de semantique."""

from pathlib import Path

import pytest

from lectura_lexique import Lexique

DONNEES_DIR = Path(__file__).parent / "donnees"
TEST_CSV = DONNEES_DIR / "test.csv"


@pytest.fixture(scope="module")
def lexique():
    with Lexique(TEST_CSV) as lex:
        yield lex


def test_synonymes_maison(lexique):
    """synonymes('maison') retourne demeure, habitation."""
    syns = lexique.synonymes("maison")
    assert "demeure" in syns
    assert "habitation" in syns


def test_synonymes_grand(lexique):
    """synonymes('grand') retourne vaste, immense."""
    syns = lexique.synonymes("grand")
    assert "vaste" in syns
    assert "immense" in syns


def test_synonymes_absent(lexique):
    """synonymes d'un mot sans synonymes retourne []."""
    syns = lexique.synonymes("le")
    assert syns == []


def test_synonymes_inexistant(lexique):
    """synonymes d'un mot inexistant retourne []."""
    assert lexique.synonymes("xyzabc") == []


def test_antonymes_grand(lexique):
    """antonymes('grand') retourne petit, minuscule."""
    ants = lexique.antonymes("grand")
    assert "petit" in ants
    assert "minuscule" in ants


def test_antonymes_petit(lexique):
    """antonymes('petit') retourne grand, immense."""
    ants = lexique.antonymes("petit")
    assert "grand" in ants
    assert "immense" in ants


def test_antonymes_absent(lexique):
    """antonymes d'un mot sans antonymes retourne []."""
    ants = lexique.antonymes("chat")
    assert ants == []


def test_definition_maison(lexique):
    """definition('maison') retourne la definition."""
    defs = lexique.definition("maison")
    assert len(defs) > 0
    assert "habitation" in defs[0].lower()


def test_definition_chat(lexique):
    """definition('chat') retourne la definition."""
    defs = lexique.definition("chat")
    assert len(defs) > 0
    assert "felin" in defs[0].lower() or "domestique" in defs[0].lower()


def test_definition_absent(lexique):
    """definition d'un mot sans definition retourne []."""
    defs = lexique.definition("le")
    assert defs == []
