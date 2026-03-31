"""Fixtures pour les tests lectura-correcteur.

Fournit un MockLexique satisfaisant LexiqueProtocol pour les tests
unitaires qui n'ont pas besoin du vrai lexique SQLite.
"""

from __future__ import annotations

from typing import Any

import pytest


class MockLexique:
    """Lexique minimal pour les tests unitaires."""

    def __init__(self, formes: dict[str, list[dict[str, Any]]] | None = None):
        if formes is None:
            formes = _DONNEES_DEFAUT()
        self._formes = formes
        self._formes_set = frozenset(formes.keys())

    def existe(self, mot: str) -> bool:
        return mot.lower() in self._formes_set

    def info(self, mot: str) -> list[dict[str, Any]]:
        return self._formes.get(mot.lower(), [])

    def frequence(self, mot: str) -> float:
        entries = self._formes.get(mot.lower(), [])
        if not entries:
            return 0.0
        return max(float(e.get("freq", 0.0)) for e in entries)

    def phone_de(self, mot: str) -> str | None:
        entries = self._formes.get(mot.lower(), [])
        phones = [
            (e.get("phone", ""), float(e.get("freq", 0.0)))
            for e in entries
            if e.get("phone")
        ]
        if not phones:
            return None
        phones.sort(key=lambda x: -x[1])
        return phones[0][0]

    def homophones(self, phone: str) -> list[dict[str, Any]]:
        results = []
        for entries in self._formes.values():
            for e in entries:
                if e.get("phone") == phone:
                    results.append(e)
        return results

    def close(self) -> None:
        pass


def _DONNEES_DEFAUT() -> dict[str, list[dict[str, Any]]]:
    """Jeu de donnees par defaut pour les tests."""
    return {
        "chat": [{"ortho": "chat", "cgram": "NOM", "phone": "\u0283a",
                   "freq": 45.2, "genre": "m", "nombre": "s"}],
        "chats": [{"ortho": "chats", "cgram": "NOM", "phone": "\u0283a",
                    "freq": 12.1, "genre": "m", "nombre": "p"}],
        "maison": [{"ortho": "maison", "cgram": "NOM", "phone": "m\u025bz\u0254\u0303",
                     "freq": 38.7, "genre": "f", "nombre": "s"}],
        "maisons": [{"ortho": "maisons", "cgram": "NOM", "phone": "m\u025bz\u0254\u0303",
                      "freq": 8.2, "genre": "f", "nombre": "p"}],
        "mange": [{"ortho": "mange", "cgram": "VER", "phone": "m\u0251\u0303\u0292",
                    "freq": 15.4, "personne": "3", "nombre": "s"}],
        "manges": [{"ortho": "manges", "cgram": "VER", "phone": "m\u0251\u0303\u0292",
                     "freq": 3.2, "personne": "2", "nombre": "s"}],
        "mangent": [{"ortho": "mangent", "cgram": "VER", "phone": "m\u0251\u0303\u0292",
                      "freq": 5.1, "personne": "3", "nombre": "p"}],
        "enfant": [{"ortho": "enfant", "cgram": "NOM", "phone": "\u0251\u0303f\u0251\u0303",
                     "freq": 30.0, "genre": "m", "nombre": "s"}],
        "enfants": [{"ortho": "enfants", "cgram": "NOM", "phone": "\u0251\u0303f\u0251\u0303",
                      "freq": 20.0, "genre": "m", "nombre": "p"}],
        "pomme": [{"ortho": "pomme", "cgram": "NOM", "phone": "p\u0254m",
                    "freq": 5.0, "genre": "f", "nombre": "s"}],
        "pommes": [{"ortho": "pommes", "cgram": "NOM", "phone": "p\u0254m",
                     "freq": 3.0, "genre": "f", "nombre": "p"}],
        "le": [{"ortho": "le", "cgram": "ART:def", "phone": "l\u0259",
                 "freq": 890.5, "genre": "m", "nombre": "s"}],
        "la": [{"ortho": "la", "cgram": "ART:def", "phone": "la",
                 "freq": 720.3, "genre": "f", "nombre": "s"}],
        "les": [{"ortho": "les", "cgram": "ART:def", "phone": "le",
                  "freq": 650.1, "nombre": "p"}],
        "des": [{"ortho": "des", "cgram": "ART:ind", "phone": "de",
                  "freq": 400.2, "nombre": "p"}],
        "est": [{"ortho": "est", "cgram": "AUX", "phone": "\u025b",
                  "freq": 500.0, "personne": "3", "nombre": "s"}],
        "et": [{"ortho": "et", "cgram": "CON", "phone": "e",
                 "freq": 450.0}],
        "son": [{"ortho": "son", "cgram": "ADJ:pos", "phone": "s\u0254\u0303",
                  "freq": 120.0, "genre": "m", "nombre": "s"}],
        "sont": [{"ortho": "sont", "cgram": "AUX", "phone": "s\u0254\u0303",
                   "freq": 200.0, "personne": "3", "nombre": "p"}],
        "il": [{"ortho": "il", "cgram": "PRO:per", "phone": "il",
                 "freq": 800.0, "personne": "3", "nombre": "s"}],
        "ils": [{"ortho": "ils", "cgram": "PRO:per", "phone": "il",
                  "freq": 300.0, "personne": "3", "nombre": "p"}],
        "elle": [{"ortho": "elle", "cgram": "PRO:per", "phone": "\u025bl",
                   "freq": 600.0, "personne": "3", "nombre": "s"}],
        "elles": [{"ortho": "elles", "cgram": "PRO:per", "phone": "\u025bl",
                    "freq": 150.0, "personne": "3", "nombre": "p"}],
        "tu": [{"ortho": "tu", "cgram": "PRO:per", "phone": "ty",
                 "freq": 500.0, "personne": "2", "nombre": "s"}],
        "je": [{"ortho": "je", "cgram": "PRO:per", "phone": "\u0292\u0259",
                 "freq": 700.0, "personne": "1", "nombre": "s"}],
        "dans": [{"ortho": "dans", "cgram": "PRE", "phone": "d\u0251\u0303",
                   "freq": 150.0}],
        "belle": [{"ortho": "belle", "cgram": "ADJ", "phone": "b\u025bl",
                    "freq": 50.0, "genre": "f", "nombre": "s"}],
        "arrive": [{"ortho": "arrive", "cgram": "VER", "phone": "a\u0281iv",
                     "freq": 20.0, "personne": "3", "nombre": "s"}],
        "bonjour": [{"ortho": "bonjour", "cgram": "NOM", "phone": "b\u0254\u0303\u0292u\u0281",
                      "freq": 60.0}],
        "comment": [{"ortho": "comment", "cgram": "ADV", "phone": "k\u0254m\u0251\u0303",
                      "freq": 100.0}],
        "jouent": [{"ortho": "jouent", "cgram": "VER", "phone": "\u0292u",
                     "freq": 10.0, "personne": "3", "nombre": "p"}],
        "dort": [{"ortho": "dort", "cgram": "VER", "phone": "d\u0254\u0281",
                   "freq": 15.0, "personne": "3", "nombre": "s"}],
        "chose": [{"ortho": "chose", "cgram": "NOM", "phone": "\u0283oz",
                    "freq": 50.0, "nombre": "s"}],
    }


@pytest.fixture
def mock_lexique():
    """Fixture MockLexique avec donnees par defaut."""
    return MockLexique()
