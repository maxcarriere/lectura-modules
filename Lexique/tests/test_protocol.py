"""Tests pour le LexiqueProtocol."""

from lectura_lexique import Lexique, LexiqueProtocol


def test_lexique_satisfait_protocol():
    """La classe Lexique doit satisfaire le LexiqueProtocol."""
    assert issubclass(Lexique, LexiqueProtocol)


def test_instance_check():
    """isinstance doit fonctionner grace a runtime_checkable."""
    from pathlib import Path

    csv_path = Path(__file__).parent / "donnees" / "test.csv"
    with Lexique(csv_path) as lex:
        assert isinstance(lex, LexiqueProtocol)
