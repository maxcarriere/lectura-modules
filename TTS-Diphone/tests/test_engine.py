"""Tests pour lectura_tts_diphone.engine."""

import pytest


def test_synth_mode_enum():
    from lectura_tts_diphone.engine import SynthMode
    assert SynthMode.FLUIDE.value == "FLUIDE"
    assert SynthMode("MOT_A_MOT") == SynthMode.MOT_A_MOT


def test_build_diphone_chain():
    from lectura_tts_diphone.engine import DiphoneEngine
    chain = DiphoneEngine.build_diphone_chain(["b", "a"])
    assert chain == ["#-b", "b-a", "a-#"]


def test_build_diphone_chain_single():
    from lectura_tts_diphone.engine import DiphoneEngine
    chain = DiphoneEngine.build_diphone_chain(["a"])
    assert chain == ["#-a", "a-#"]


def test_build_diphone_chain_empty():
    from lectura_tts_diphone.engine import DiphoneEngine
    chain = DiphoneEngine.build_diphone_chain([])
    assert chain == []


def test_compute_phone_durations():
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    engine = DiphoneEngine()
    durs = engine.compute_phone_durations(["b", "a"], SynthMode.FLUIDE)
    assert len(durs) == 2
    assert durs[0] < durs[1]  # consonant < vowel (with final lengthening)


def test_compute_phone_durations_syllabes():
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    engine = DiphoneEngine()
    durs = engine.compute_phone_durations(["b", "a"], SynthMode.SYLLABES)
    assert durs[0] == 80.0
    assert durs[1] == 350.0


def test_phone_durs_to_diphone_durs():
    from lectura_tts_diphone.engine import DiphoneEngine
    engine = DiphoneEngine()
    chain = ["#-b", "b-a", "a-#"]
    phone_durs = [50.0, 120.0]
    di_durs = engine._phone_durs_to_diphone_durs(chain, phone_durs)
    assert len(di_durs) == 3
    assert di_durs[0] == 25.0  # #-b = phone[0]/2
    assert di_durs[2] == 60.0  # a-# = phone[-1]/2


def test_f0_targets_syllabes():
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    engine = DiphoneEngine()
    f0s = engine.compute_f0_targets(["b", "a", "l"], SynthMode.SYLLABES)
    assert f0s == [190.0, 190.0, 190.0]


def test_f0_targets_fluide():
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    engine = DiphoneEngine()
    f0s = engine.compute_f0_targets(["b", "a", "l"], SynthMode.FLUIDE)
    assert len(f0s) == 3
    assert f0s[0] > f0s[-1]  # declination


def test_pause_defaults():
    from lectura_tts_diphone.engine import DiphoneEngine
    engine = DiphoneEngine()
    assert engine._get_pause_ms("period") == 550.0
    assert engine._get_pause_ms("comma") == 50.0
    assert engine._get_pause_ms("unknown") == 60.0


def test_group_f0_contour_question():
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    f0s = DiphoneEngine._group_f0_contour(
        ["b", "a", "l", "a"],
        SynthMode.FLUIDE,
        {"group_idx": 0, "n_groups": 1, "boundary": "question", "base_f0": 200.0},
    )
    assert len(f0s) == 4
    # Interrogative: last phone should be higher
    assert f0s[-1] > f0s[0]


def test_engine_not_loaded():
    """Engine sans load ne crashe pas a l'init."""
    from lectura_tts_diphone.engine import DiphoneEngine
    engine = DiphoneEngine()
    assert engine.loaded is False
    assert engine.diphones == {}
