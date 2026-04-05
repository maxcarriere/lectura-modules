"""Tests pour le module _loaders."""

from pathlib import Path

from lectura_lexique._loaders import iter_csv

DONNEES_DIR = Path(__file__).parent / "donnees"
TEST_CSV = DONNEES_DIR / "test.csv"


def test_iter_csv_produit_des_entrees():
    """iter_csv doit produire des entrees non vides."""
    entrees = list(iter_csv(TEST_CSV))
    assert len(entrees) == 34


def test_iter_csv_champs_canoniques():
    """Les champs doivent etre en noms canoniques."""
    entrees = list(iter_csv(TEST_CSV))
    premiere = entrees[0]
    assert "ortho" in premiere
    assert premiere["ortho"] == "chat"


def test_iter_csv_frequence_resolue():
    """La frequence doit etre resolue depuis freq_opensubs."""
    entrees = list(iter_csv(TEST_CSV))
    premiere = entrees[0]
    assert "freq" in premiere
    assert float(premiere["freq"]) > 0


def test_iter_csv_phone():
    """Le champ phone doit etre present."""
    entrees = list(iter_csv(TEST_CSV))
    premiere = entrees[0]
    assert "phone" in premiere
    assert premiere["phone"] == "\u0283a"  # ʃa
