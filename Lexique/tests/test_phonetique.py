"""Tests pour les fonctions de phonetique."""

from pathlib import Path

import pytest

from lectura_lexique import Lexique

DONNEES_DIR = Path(__file__).parent / "donnees"
TEST_CSV = DONNEES_DIR / "test.csv"


@pytest.fixture(scope="module")
def lexique():
    with Lexique(TEST_CSV) as lex:
        yield lex


def test_rimes_maison(lexique):
    """rimes('maison') trouve des mots en -ɔ̃."""
    rimes = lexique.rimes("maison", nb_phonemes=1)
    orthos = {r["ortho"].lower() for r in rimes}
    # maison phone = mɛzɔ̃, dernier phoneme = ɔ̃
    # mangeons phone = mɑ̃ʒɔ̃, dernier phoneme = ɔ̃
    assert "mangeons" in orthos
    # son phone = sɔ̃
    assert "son" in orthos


def test_rimes_vide(lexique):
    """rimes d'un mot inexistant retourne []."""
    assert lexique.rimes("xyzabc") == []


def test_contient_son_a(lexique):
    """contient_son('a') trouve des mots avec /a/."""
    results = lexique.contient_son("a")
    orthos = {r["ortho"].lower() for r in results}
    # chat = ʃa, la = la, a = a
    assert "chat" in orthos or "la" in orthos or "a" in orthos
    assert len(results) > 0


def test_contient_son_vide(lexique):
    """contient_son('') retourne []."""
    assert lexique.contient_son("") == []


def test_mots_par_syllabes_1(lexique):
    """mots_par_syllabes(1) retourne les monosyllabes."""
    results = lexique.mots_par_syllabes(1)
    orthos = {r["ortho"].lower() for r in results}
    assert "chat" in orthos or "le" in orthos
    for r in results:
        nb = r.get("nb_syllabes")
        if nb is not None:
            # nb_syllabes peut etre string ou int selon le loader
            assert int(nb) == 1


def test_mots_par_syllabes_2(lexique):
    """mots_par_syllabes(2) retourne les dissyllabes."""
    results = lexique.mots_par_syllabes(2)
    orthos = {r["ortho"].lower() for r in results}
    assert "maison" in orthos or "petit" in orthos


def test_mots_par_syllabes_filtre_cgram(lexique):
    """mots_par_syllabes avec filtre POS."""
    results = lexique.mots_par_syllabes(2, cgram="NOM")
    for r in results:
        assert r["cgram"].startswith("NOM")
