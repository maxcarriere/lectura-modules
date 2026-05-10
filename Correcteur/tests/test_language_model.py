"""Tests pour ScorerNgram (_language_model.py)."""

from __future__ import annotations

import math
import sqlite3
import tempfile
from pathlib import Path

import pytest

from lectura_correcteur._language_model import ScorerNgram, _LOG_PROB_OOV


@pytest.fixture
def ngram_db(tmp_path):
    """Cree une base n-gram SQLite minimale pour les tests."""
    db_path = str(tmp_path / "test_ngram.db")
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE unigrams (
            word TEXT PRIMARY KEY,
            count INTEGER,
            prob REAL
        )
    """)
    conn.execute("""
        CREATE TABLE bigrams (
            w1 TEXT, w2 TEXT,
            count INTEGER,
            prob REAL,
            backoff REAL,
            PRIMARY KEY (w1, w2)
        )
    """)
    conn.execute("""
        CREATE TABLE trigrams (
            w1 TEXT, w2 TEXT, w3 TEXT,
            count INTEGER,
            prob REAL,
            backoff REAL,
            PRIMARY KEY (w1, w2, w3)
        )
    """)

    # Unigrammes
    unigrams = [
        ("le", 1000, -1.0),
        ("chat", 500, -1.3),
        ("mange", 200, -1.7),
        ("la", 900, -1.05),
        ("souris", 100, -2.0),
        ("les", 800, -1.1),
        ("chats", 300, -1.5),
        ("mangent", 150, -1.8),
        ("des", 700, -1.15),
        ("<s>", 5000, -0.3),
        ("</s>", 5000, -0.3),
    ]
    conn.executemany(
        "INSERT INTO unigrams VALUES (?, ?, ?)", unigrams,
    )

    # Bigrammes
    bigrams = [
        ("le", "chat", 200, -0.7, -0.3),
        ("chat", "mange", 80, -0.8, -0.2),
        ("mange", "la", 50, -1.0, -0.1),
        ("la", "souris", 40, -1.1, -0.2),
        ("<s>", "le", 300, -0.5, -0.3),
        ("<s>", "les", 250, -0.6, -0.3),
        ("les", "chats", 150, -0.7, -0.2),
        ("chats", "mangent", 60, -0.9, -0.2),
        ("mangent", "des", 40, -1.0, -0.1),
    ]
    conn.executemany(
        "INSERT INTO bigrams VALUES (?, ?, ?, ?, ?)", bigrams,
    )

    # Trigrammes
    trigrams = [
        ("le", "chat", "mange", 50, -0.5, -0.2),
        ("chat", "mange", "la", 30, -0.6, -0.1),
        ("mange", "la", "souris", 20, -0.7, 0.0),
        ("<s>", "le", "chat", 100, -0.3, -0.2),
        ("les", "chats", "mangent", 40, -0.5, -0.1),
    ]
    conn.executemany(
        "INSERT INTO trigrams VALUES (?, ?, ?, ?, ?, ?)", trigrams,
    )

    conn.commit()
    conn.close()
    return db_path


class TestScorerNgramBasic:
    """Tests de base pour ScorerNgram."""

    def test_construction(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        assert lm._max_order == 3
        lm.close()

    def test_max_order_detection(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        assert lm._max_order == 3
        lm.close()

    def test_close(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        lm.close()


class TestScoreMotEnContexte:
    """Tests pour score_mot_en_contexte avec backoff."""

    def test_trigram_hit(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        # "le chat mange" -> trigram (le, chat, mange) = -0.5
        score = lm.score_mot_en_contexte("mange", ["le", "chat"])
        assert score == pytest.approx(-0.5)
        lm.close()

    def test_bigram_hit(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        # "le chat" -> bigram (le, chat) = -0.7
        score = lm.score_mot_en_contexte("chat", ["le"])
        assert score == pytest.approx(-0.7)
        lm.close()

    def test_unigram_hit(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        # No context -> unigram "chat" = -1.3
        score = lm.score_mot_en_contexte("chat", [])
        assert score == pytest.approx(-1.3)
        lm.close()

    def test_backoff_trigram_to_bigram(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        # Trigram ("la", "souris", "mange") doesn't exist
        # Should backoff to bigram: backoff(la, souris) + P(mange|souris)
        # backoff(la, souris) is not in trigrams -> 0.0
        # P(mange|souris) bigram doesn't exist -> backoff(souris) + P(mange)
        score = lm.score_mot_en_contexte("mange", ["la", "souris"])
        assert isinstance(score, float)
        assert score < 0  # should be a negative log-prob
        lm.close()

    def test_oov_word(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        score = lm.score_mot_en_contexte("xyzinexistant", [])
        assert score == pytest.approx(_LOG_PROB_OOV)
        lm.close()

    def test_case_insensitive(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        s1 = lm.score_mot_en_contexte("Chat", ["Le"])
        s2 = lm.score_mot_en_contexte("chat", ["le"])
        assert s1 == pytest.approx(s2)
        lm.close()


class TestScorePhrase:
    """Tests pour score_phrase."""

    def test_known_phrase(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        score = lm.score_phrase(["le", "chat", "mange"])
        assert isinstance(score, float)
        assert score < 0  # sum of negative log-probs
        lm.close()

    def test_empty_phrase(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        assert lm.score_phrase([]) == 0.0
        lm.close()

    def test_better_phrase_scores_higher(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        # "le chat mange" should score better than random order
        s1 = lm.score_phrase(["le", "chat", "mange"])
        s2 = lm.score_phrase(["mange", "le", "chat"])
        # s1 should be higher (less negative)
        assert s1 > s2
        lm.close()


class TestScorerCandidats:
    """Tests pour scorer_candidats."""

    def test_basic_ranking(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        # After "le", "chat" should score better than "souris"
        result = lm.scorer_candidats(
            ["chat", "souris"], ["le"], [],
        )
        assert len(result) == 2
        assert result[0][0] == "chat"
        assert result[0][1] > result[1][1]
        lm.close()

    def test_with_right_context(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        # "chat" before "mange" should score better than "souris" before "mange"
        result = lm.scorer_candidats(
            ["chat", "souris"], ["le"], ["mange"],
        )
        assert result[0][0] == "chat"
        lm.close()

    def test_empty_candidats(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        result = lm.scorer_candidats([], ["le"], [])
        assert result == []
        lm.close()

    def test_returns_sorted(self, ngram_db):
        lm = ScorerNgram(ngram_db)
        result = lm.scorer_candidats(
            ["chat", "souris", "mange"], ["le"], [],
        )
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
        lm.close()
