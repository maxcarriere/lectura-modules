"""Tests pour le surlignage formules et la validation spans."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_aligneur import (
    EventFormule,
    GroupeLecture,
    GroupePhonologique,
    LectureFormule,
    MotAnalyse,
    Phoneme,
    ResultatGroupe,
    Syllabe,
    lecture_depuis_g2p,
    syllabifier_groupes,
    _valider_spans_formule,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_lecture(events_data: list[tuple[str, str, tuple[int, int]]]) -> LectureFormule:
    """Crée une LectureFormule à partir de (ortho, phone, span_source)."""
    events = [
        EventFormule(ortho=o, phone=p, span_source=s)
        for o, p, s in events_data
    ]
    display = "-".join(e.ortho for e in events)
    return LectureFormule(display_fr=display, events=events)


def _make_formule_groupe(lecture: LectureFormule) -> GroupeLecture:
    """Crée un GroupeLecture de type formule."""
    return GroupeLecture(
        mots=[],
        phone_groupe="",
        span=(0, 0),
        est_formule=True,
        lecture=lecture,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tests mode progressif (chaque event = 1 syllabe)
# ══════════════════════════════════════════════════════════════════════════════

class TestModeProgressif:
    def test_chaque_event_est_une_syllabe(self):
        lecture = _make_lecture([
            ("quarante", "kaʁɑ̃t", (0, 2)),
            ("deux",     "dø",     (2, 3)),
        ])
        groupe = _make_formule_groupe(lecture)
        resultats = syllabifier_groupes([groupe])

        assert len(resultats) == 1
        syllabes = resultats[0].syllabes
        assert len(syllabes) == 2
        assert syllabes[0].phone == "kaʁɑ̃t"
        assert syllabes[0].ortho == "quarante"
        assert syllabes[1].phone == "dø"
        assert syllabes[1].ortho == "deux"

    def test_spans_preserves(self):
        lecture = _make_lecture([
            ("cent",     "sɑ̃",   (5, 6)),
            ("quarante", "kaʁɑ̃t", (6, 7)),
            ("deux",     "dø",    (7, 8)),
        ])
        groupe = _make_formule_groupe(lecture)
        resultats = syllabifier_groupes([groupe])

        syllabes = resultats[0].syllabes
        assert syllabes[0].span == (5, 6)
        assert syllabes[1].span == (6, 7)
        assert syllabes[2].span == (7, 8)


# ══════════════════════════════════════════════════════════════════════════════
# Tests validation spans
# ══════════════════════════════════════════════════════════════════════════════

class TestValiderSpans:
    def test_spans_valides(self):
        lecture = _make_lecture([
            ("un", "ɛ̃", (0, 1)),
            ("deux", "dø", (1, 3)),
        ])
        _valider_spans_formule(lecture)  # Ne doit pas lever d'exception

    def test_span_start_sup_end(self):
        lecture = LectureFormule(
            display_fr="erreur",
            events=[EventFormule(ortho="x", phone="x", span_source=(5, 3))],
        )
        with pytest.raises(ValueError, match="incohérent"):
            _valider_spans_formule(lecture)

    def test_span_egal(self):
        """span (3, 3) est valide (longueur 0)."""
        lecture = _make_lecture([("x", "x", (3, 3))])
        _valider_spans_formule(lecture)  # OK


# ══════════════════════════════════════════════════════════════════════════════
# Tests conversion G2P → Syllabeur
# ══════════════════════════════════════════════════════════════════════════════

class _MockG2PResult:
    """Simule un LectureFormuleResult du module G2P."""
    def __init__(self):
        self.display_fr = "quarante-deux"
        self.events = [
            type("Evt", (), {"ortho": "quarante", "phone": "kaʁɑ̃t", "span_source": (0, 2)})(),
            type("Evt", (), {"ortho": "deux", "phone": "dø", "span_source": (2, 3)})(),
        ]

class TestLectureDepuisG2P:
    def test_conversion_basique(self):
        g2p_result = _MockG2PResult()
        lecture = lecture_depuis_g2p(g2p_result)

        assert isinstance(lecture, LectureFormule)
        assert lecture.display_fr == "quarante-deux"
        assert len(lecture.events) == 2
        assert lecture.events[0].ortho == "quarante"
        assert lecture.events[0].phone == "kaʁɑ̃t"
        assert lecture.events[0].span_source == (0, 2)
        assert isinstance(lecture.events[0], EventFormule)

    def test_conversion_vide(self):
        """Objet sans events."""
        empty = type("R", (), {"display_fr": "", "events": []})()
        lecture = lecture_depuis_g2p(empty)
        assert lecture.display_fr == ""
        assert lecture.events == []


# ══════════════════════════════════════════════════════════════════════════════
# Tests lectures via dict
# ══════════════════════════════════════════════════════════════════════════════

class TestLecturesFormuleDict:
    def test_lecture_fournie_par_dict(self):
        """Formule sans lecture pré-attachée, fournie via le dict."""
        groupe = GroupeLecture(
            mots=[],
            phone_groupe="",
            span=(0, 2),
            est_formule=True,
            lecture=None,
        )
        lecture = _make_lecture([
            ("quarante", "kaʁɑ̃t", (0, 1)),
            ("deux", "dø", (1, 2)),
        ])
        resultats = syllabifier_groupes([groupe], lectures_formules={0: lecture})

        assert len(resultats[0].syllabes) == 2
        assert groupe.lecture is not None  # attaché par side-effect
