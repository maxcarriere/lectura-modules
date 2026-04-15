"""Tests pour les fonctions de morphologie."""

from pathlib import Path

import pytest

from lectura_lexique import Lexique

DONNEES_DIR = Path(__file__).parent / "donnees"
TEST_CSV = DONNEES_DIR / "test.csv"


@pytest.fixture(scope="module")
def lexique():
    with Lexique(TEST_CSV) as lex:
        yield lex


def test_conjuguer_manger(lexique):
    """conjuguer('manger') retourne une table avec indicatif/present."""
    table = lexique.conjuguer("manger")
    assert "indicatif" in table
    assert "présent" in table["indicatif"]
    pst = table["indicatif"]["présent"]
    # Verifier quelques personnes
    assert pst.get("1s") == "mange"
    assert pst.get("2s") == "manges"
    assert pst.get("3s") == "mange"
    assert pst.get("1p") == "mangeons"
    assert pst.get("2p") == "mangez"
    assert pst.get("3p") == "mangent"


def test_conjuguer_imparfait(lexique):
    """conjuguer('manger') contient l'imparfait."""
    table = lexique.conjuguer("manger")
    assert "imparfait" in table.get("indicatif", {})
    imp = table["indicatif"]["imparfait"]
    assert imp.get("3s") == "mangeait"


def test_conjuguer_inexistant(lexique):
    """conjuguer d'un mot inexistant retourne un dict vide."""
    table = lexique.conjuguer("xyzabc")
    assert table == {}


def test_conjuguer_nom(lexique):
    """conjuguer d'un nom retourne un dict vide (pas de verbe)."""
    table = lexique.conjuguer("chat")
    assert table == {}


def test_formes_de_chat(lexique):
    """formes_de('chat') retourne chat et chats."""
    formes = lexique.formes_de("chat")
    orthos = {f["ortho"].lower() for f in formes}
    assert "chat" in orthos
    assert "chats" in orthos


def test_formes_de_manger(lexique):
    """formes_de('manger') retourne les formes verbales."""
    formes = lexique.formes_de("manger")
    orthos = {f["ortho"].lower() for f in formes}
    assert "mange" in orthos
    assert "mangeons" in orthos
    assert "mangez" in orthos


def test_formes_de_filtre_cgram(lexique):
    """formes_de avec filtre cgram."""
    formes = lexique.formes_de("manger", cgram="VER")
    assert len(formes) > 0
    for f in formes:
        assert f["cgram"].startswith("VER")


def test_lemme_de_chat(lexique):
    """lemme_de('chat') retourne 'chat'."""
    assert lexique.lemme_de("chat") == "chat"


def test_lemme_de_mange(lexique):
    """lemme_de('mange') retourne 'manger'."""
    assert lexique.lemme_de("mange") == "manger"


def test_lemme_de_inexistant(lexique):
    """lemme_de d'un mot inexistant retourne None."""
    assert lexique.lemme_de("xyzabc") is None
