"""Tests pour la classe Lexique."""

from pathlib import Path

import pytest

from lectura_lexique import Lexique

DONNEES_DIR = Path(__file__).parent / "donnees"
TEST_CSV = DONNEES_DIR / "test.csv"


@pytest.fixture(scope="module")
def lexique():
    with Lexique(TEST_CSV) as lex:
        yield lex


def test_existe_mots_courants(lexique):
    for mot in ("chat", "maison", "mange", "les", "est", "dans"):
        assert lexique.existe(mot), f"{mot} devrait exister"


def test_existe_insensible_casse(lexique):
    assert lexique.existe("Chat")
    assert lexique.existe("MAISON")


def test_inexistant(lexique):
    assert not lexique.existe("xyzabc123")
    assert not lexique.existe("manjer")


def test_homophones_chat(lexique):
    """Le phone 'ʃa' devrait retourner chat, chats et chas."""
    homos = lexique.homophones("\u0283a")
    orthos = {h["ortho"] for h in homos}
    assert "chat" in orthos
    assert len(homos) >= 2


def test_homophones_vide(lexique):
    assert lexique.homophones("xyzxyz") == []


def test_info(lexique):
    infos = lexique.info("chat")
    assert len(infos) > 0
    entry = infos[0]
    assert "ortho" in entry
    assert "cgram" in entry
    assert "phone" in entry


def test_frequence(lexique):
    assert lexique.frequence("le") > 0
    assert lexique.frequence("xyzabc123") == 0.0


def test_phone_de(lexique):
    phone = lexique.phone_de("chat")
    assert phone is not None
    assert "\u0283" in phone  # ʃ


def test_context_manager():
    with Lexique(TEST_CSV) as lex:
        assert lex.existe("chat")


def test_from_directory():
    lex = Lexique.from_directory(DONNEES_DIR)
    assert lex.existe("chat")
    lex.close()


