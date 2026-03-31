"""Fixtures pour les tests lectura-lexique."""

from pathlib import Path

import pytest

from lectura_lexique import Lexique

DONNEES_DIR = Path(__file__).parent / "donnees"
TEST_CSV = DONNEES_DIR / "test.csv"


@pytest.fixture(scope="module")
def lexique_csv():
    """Lexique charge depuis le CSV de test."""
    with Lexique(TEST_CSV) as lex:
        yield lex
