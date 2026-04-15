"""Tests d'intégration pour pipeline_formules.py — pipeline Tokeniseur→G2P."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_formules import OptionsLecture
from lectura_nlp.pipeline_formules import (
    MotAnalyseG2P,
    ResultatPhraseG2P,
    analyser_phrase_complete,
)


# ── Mock tokens (simulent la sortie du Tokeniseur) ────────────────────

class MockType:
    def __init__(self, value: str):
        self.value = value

class MockFormuleType:
    def __init__(self, value: str):
        self.value = value

class MockToken:
    def __init__(self, type_val: str, text: str, span: tuple[int, int]):
        self.type = MockType(type_val)
        self.text = text
        self.span = span

class MockFormule(MockToken):
    def __init__(self, text: str, span: tuple[int, int],
                 formule_type: str = "nombre", children: list | None = None):
        super().__init__("formule", text, span)
        self.formule_type = MockFormuleType(formule_type)
        self.children = children or []


# ── Mock engine ────────────────────────────────────────────────────────

class MockEngine:
    """Moteur G2P factice pour les tests."""
    def analyser(self, tokens: list[str]) -> dict:
        g2p = []
        pos = []
        liaison = []
        for t in tokens:
            if t == "PLACEHOLDER":
                g2p.append("")
                pos.append("")
                liaison.append("")
            elif t.lower() == "le":
                g2p.append("lə")
                pos.append("ART:def")
                liaison.append("none")
            elif t.lower() == "numéro":
                g2p.append("nymeʁo")
                pos.append("NOM")
                liaison.append("none")
            else:
                g2p.append(t.lower())
                pos.append("NOM")
                liaison.append("none")
        return {
            "tokens": tokens,
            "g2p": g2p,
            "pos": pos,
            "liaison": liaison,
            "morpho": {},
        }


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyserPhraseComplete:
    def test_phrase_sans_formule(self):
        tokens = [
            MockToken("mot", "Le", (0, 2)),
            MockToken("mot", "chat", (3, 7)),
        ]
        result = analyser_phrase_complete(tokens, engine=MockEngine())
        assert len(result.mots) == 2
        assert result.mots[0].phone == "lə"
        assert not result.mots[0].est_formule
        assert not result.mots[1].est_formule

    def test_phrase_avec_formule_nombre(self):
        tokens = [
            MockToken("mot", "Le", (0, 2)),
            MockToken("mot", "numéro", (3, 9)),
            MockFormule("42", (10, 12), formule_type="nombre"),
        ]
        result = analyser_phrase_complete(tokens, engine=MockEngine())
        assert len(result.mots) == 3
        assert result.mots[0].phone == "lə"
        assert result.mots[2].est_formule
        assert result.mots[2].lecture is not None
        assert "quarante" in result.mots[2].lecture.display_fr

    def test_phrase_formule_sigle(self):
        tokens = [
            MockFormule("SNCF", (0, 4), formule_type="sigle"),
        ]
        result = analyser_phrase_complete(tokens)
        assert len(result.mots) == 1
        assert result.mots[0].est_formule
        assert "esse" in result.mots[0].lecture.display_fr

    def test_phrase_sans_engine(self):
        """Sans moteur neural, les mots normaux ont phone vide."""
        tokens = [
            MockToken("mot", "chat", (0, 4)),
            MockFormule("42", (5, 7), formule_type="nombre"),
        ]
        result = analyser_phrase_complete(tokens, engine=None)
        assert result.mots[0].phone == ""  # pas de G2P neural
        assert result.mots[1].phone != ""  # lecture algo

    def test_ponctuation_ignoree(self):
        tokens = [
            MockToken("mot", "Bonjour", (0, 7)),
            MockToken("ponctuation", ".", (7, 8)),
        ]
        result = analyser_phrase_complete(tokens, engine=MockEngine())
        assert len(result.mots) == 1  # ponctuation ignorée

    def test_proprietes_resultat(self):
        tokens = [
            MockToken("mot", "Le", (0, 2)),
            MockFormule("42", (3, 5), formule_type="nombre"),
        ]
        result = analyser_phrase_complete(tokens, engine=MockEngine())
        assert result.tokens == ["Le", "42"]
        assert len(result.phones) == 2
        assert len(result.formules) == 1
        assert 1 in result.formules

    def test_phrase_vide(self):
        result = analyser_phrase_complete([], engine=MockEngine())
        assert len(result.mots) == 0

    def test_multiple_formules(self):
        tokens = [
            MockFormule("42", (0, 2), formule_type="nombre"),
            MockToken("mot", "et", (3, 5)),
            MockFormule("SNCF", (6, 10), formule_type="sigle"),
        ]
        result = analyser_phrase_complete(tokens, engine=MockEngine())
        assert len(result.mots) == 3
        assert result.mots[0].est_formule
        assert not result.mots[1].est_formule
        assert result.mots[2].est_formule
