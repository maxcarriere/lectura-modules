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
    assert di_durs[2] == 80.0  # a-# = max(phone[-1]/2, 80) — plancher terminal


def test_f0_targets_syllabes():
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    engine = DiphoneEngine()
    f0s = engine.compute_f0_targets(["b", "a", "l"], SynthMode.SYLLABES)
    assert f0s == [175.0, 175.0, 175.0]


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


# ── Tests prosodie AP ───────────────────────────────────────────────


def test_segment_aps_single_word():
    """Un seul mot = 1 AP."""
    from lectura_tts_diphone.engine import DiphoneEngine
    aps = DiphoneEngine._segment_aps(["b", "ɔ̃", "ʒ", "u", "ʁ"], [])
    assert aps == [(0, 5)]


def test_segment_aps_merge_short():
    """Mot court (<=2 phones) fusionne avec le mot suivant."""
    from lectura_tts_diphone.engine import DiphoneEngine
    # "le" (2 phones) + "chat" (3 phones) → 1 AP
    aps = DiphoneEngine._segment_aps(["l", "ə", "ʃ", "a", "t"], [2])
    assert len(aps) == 1
    assert aps[0] == (0, 5)


def test_segment_aps_two_words():
    """Deux mots longs = 2 APs."""
    from lectura_tts_diphone.engine import DiphoneEngine
    # "bonjour" (4 phones) + "comment" (4 phones)
    phones = ["b", "ɔ̃", "ʒ", "u", "k", "ɔ", "m", "ɑ̃"]
    aps = DiphoneEngine._segment_aps(phones, [4])
    assert len(aps) == 2
    assert aps[0] == (0, 4)
    assert aps[1] == (4, 8)


def test_f0_ap_pattern_lh_star():
    """AP courte: F0 finale > F0 initiale (pattern LH*)."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    # Single AP, continuation (non-final group)
    phones = ["b", "a", "l"]
    info = {"group_idx": 0, "n_groups": 2, "boundary": "none",
            "base_f0": 175.0, "macro_expressivity": 1.0}
    f0s = DiphoneEngine._group_f0_contour(phones, SynthMode.FLUIDE, info)
    # The vowel "a" at index 1 is the only syllable; consonants interpolate
    # With continuation pattern, H* should make final phone relatively high
    assert f0s[2] >= f0s[0]  # final >= initial (continuation rise)


def test_f0_ap_continuation_vs_final():
    """AP non-finale monte (continuation), AP finale descend (declaratif)."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    # Two words = 2 APs
    phones = ["b", "ɔ̃", "ʒ", "u", "k", "ɔ", "m", "ɑ̃"]
    info_cont = {"group_idx": 0, "n_groups": 2, "boundary": "none",
                 "base_f0": 175.0, "macro_expressivity": 1.0}
    f0s_cont = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info_cont, word_boundaries=[4])

    info_final = {"group_idx": 0, "n_groups": 1, "boundary": "period",
                  "base_f0": 175.0, "macro_expressivity": 1.0}
    f0s_final = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info_final, word_boundaries=[4])

    # In continuation, the last AP's final vowel should be higher
    # In declarative, the last AP's final vowel should be lower
    # Find the last vowel (index 7, "ɑ̃")
    assert f0s_cont[7] > f0s_final[7]


def test_microprosody_voiceless_stop():
    """Voyelle apres /p/ (sourde) a F0 plus haut que apres /b/ (voisee)."""
    from lectura_tts_diphone.engine import _ONSET_F0_PERTURB
    assert _ONSET_F0_PERTURB["p"] > 0  # sourde → raise F0
    assert _ONSET_F0_PERTURB["b"] < 0  # voisee → lower F0
    assert _ONSET_F0_PERTURB["p"] > _ONSET_F0_PERTURB["b"]


def test_duration_accent():
    """Voyelle accentuee (AP-finale) plus longue que non-accentuee."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    engine = DiphoneEngine()
    # "bala" — 2 vowels, second is AP-final accent
    phones = ["b", "a", "l", "a"]
    # Without accent
    durs_no_accent = engine.compute_phone_durations(phones, SynthMode.FLUIDE)
    # With accent on last vowel (index 3)
    durs_accent = engine.compute_phone_durations(
        phones, SynthMode.FLUIDE, accent_positions={3})
    # Accented vowel should be longer
    assert durs_accent[3] > durs_no_accent[3]
    # Non-accented vowel should be short (90ms base)
    assert durs_accent[1] == 90.0  # no final lengthening on non-last phone


# ── Tests pipeline timbre ──────────────────────────────────────────


def test_compress_ap_identity():
    """gamma=1.0 retourne input inchange."""
    import numpy as np
    from lectura_tts_diphone._world import compress_aperiodicity

    ap = np.random.rand(10, 50).astype(np.float64)
    result = compress_aperiodicity(ap, gamma=1.0)
    np.testing.assert_array_equal(result, ap)


def test_compress_ap_reduces():
    """AP=0.1 descend apres compression (gamma=2.0)."""
    import numpy as np
    from lectura_tts_diphone._world import compress_aperiodicity

    ap = np.full((5, 50), 0.1, dtype=np.float64)
    result = compress_aperiodicity(ap, gamma=2.0)
    # 0.1^2 = 0.01 (or similar with freq-dependent gamma)
    assert np.all(result < ap)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_compress_ap_freq_dependent():
    """Bin 1kHz plus comprime que bin 10kHz."""
    import numpy as np
    from lectura_tts_diphone._world import SIWIS_SR, compress_aperiodicity

    n_bins = 100
    ap = np.full((1, n_bins), 0.3, dtype=np.float64)
    result = compress_aperiodicity(ap, gamma=2.0, sr=SIWIS_SR)
    freq_per_bin = (SIWIS_SR / 2) / max(1, n_bins - 1)
    bin_1k = int(1000 / freq_per_bin)
    bin_10k = int(10000 / freq_per_bin)
    if bin_10k < n_bins:
        # Basse freq a gamma plus fort → valeur plus petite
        assert result[0, bin_1k] < result[0, bin_10k]


def test_sharpen_identity():
    """gain=1.0 retourne input inchange."""
    import numpy as np
    from lectura_tts_diphone._world import sharpen_formants

    sp = np.random.rand(10, 64).astype(np.float64) + 0.01
    result = sharpen_formants(sp, gain=1.0)
    np.testing.assert_array_equal(result, sp)


def test_sharpen_increases_contrast():
    """Ratio max/min augmente apres sharpening."""
    import numpy as np
    from lectura_tts_diphone._world import sharpen_formants

    # Spectre avec structure : 2 pics formantiques
    sp = np.ones((5, 64), dtype=np.float64) * 0.1
    sp[:, 10] = 1.0  # pic F1
    sp[:, 25] = 0.8  # pic F2
    ratio_before = sp[0].max() / sp[0].min()
    result = sharpen_formants(sp, gain=1.5)
    ratio_after = result[0].max() / result[0].min()
    assert ratio_after > ratio_before


def test_vtln_identity():
    """alpha=1.0 retourne input inchange."""
    import numpy as np
    from lectura_tts_diphone._world import warp_vtln

    sp = np.random.rand(10, 64).astype(np.float64) + 0.01
    result = warp_vtln(sp, alpha=1.0)
    np.testing.assert_array_equal(result, sp)


def test_vtln_shifts_peak():
    """alpha>1 deplace le pic spectral vers le haut."""
    import numpy as np
    from lectura_tts_diphone._world import warp_vtln

    sp = np.ones((1, 128), dtype=np.float64) * 0.01
    peak_bin = 30
    sp[0, peak_bin] = 1.0
    # Warping alpha=1.15 : pic monte en frequence
    result = warp_vtln(sp, alpha=1.15, sr=44100)
    new_peak = np.argmax(result[0])
    assert new_peak > peak_bin


def test_existing_tests_pass():
    """Verifier que les constantes fondamentales n'ont pas change."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    # SynthMode
    assert SynthMode.FLUIDE.value == "FLUIDE"
    # Diphone chain
    assert DiphoneEngine.build_diphone_chain(["a"]) == ["#-a", "a-#"]
    # SYLLABES durations unchanged
    engine = DiphoneEngine()
    durs = engine.compute_phone_durations(["b", "a"], SynthMode.SYLLABES)
    assert durs[0] == 80.0
    assert durs[1] == 350.0
    # F0 SYLLABES flat
    f0s = engine.compute_f0_targets(["b", "a"], SynthMode.SYLLABES)
    assert f0s == [175.0, 175.0]
    # Pause defaults
    assert engine._get_pause_ms("period") == 550.0


# ── Tests concat_diphones boundaries ─────────────────────────────


def test_concat_diphones_returns_boundaries():
    """concat_diphones retourne un 4-tuple avec les frontieres."""
    import numpy as np
    from lectura_tts_diphone._world import concat_diphones

    n_bins = 50
    seg1 = {
        "f0": np.ones(10, dtype=np.float64) * 200,
        "sp": np.ones((10, n_bins), dtype=np.float64) * 0.5,
        "ap": np.ones((10, n_bins), dtype=np.float64) * 0.3,
    }
    seg2 = {
        "f0": np.ones(8, dtype=np.float64) * 180,
        "sp": np.ones((8, n_bins), dtype=np.float64) * 0.4,
        "ap": np.ones((8, n_bins), dtype=np.float64) * 0.2,
    }

    result = concat_diphones([seg1, seg2])
    assert len(result) == 4
    f0, sp, ap, boundaries = result
    assert len(boundaries) == 2
    # First segment starts at 0
    assert boundaries[0][0] == 0
    # Second segment starts within the output
    assert boundaries[1][0] >= 0
    assert boundaries[1][1] <= len(f0)


def test_concat_diphones_single_segment():
    """Un seul segment retourne boundaries = [(0, n)]."""
    import numpy as np
    from lectura_tts_diphone._world import concat_diphones

    seg = {
        "f0": np.ones(15, dtype=np.float64) * 200,
        "sp": np.ones((15, 50), dtype=np.float64),
        "ap": np.ones((15, 50), dtype=np.float64) * 0.5,
    }

    f0, sp, ap, boundaries = concat_diphones([seg])
    assert boundaries == [(0, 15)]
    assert len(f0) == 15


def test_concat_diphones_boundaries_cover_all_frames():
    """Les frontieres couvrent tout le signal concatene."""
    import numpy as np
    from lectura_tts_diphone._world import concat_diphones

    n_bins = 30
    segments = []
    for n in [12, 10, 14, 8]:
        segments.append({
            "f0": np.ones(n, dtype=np.float64) * 180,
            "sp": np.ones((n, n_bins), dtype=np.float64) * 0.5,
            "ap": np.ones((n, n_bins), dtype=np.float64) * 0.3,
        })

    f0, sp, ap, boundaries = concat_diphones(segments)
    assert len(boundaries) == 4
    # First boundary starts at 0
    assert boundaries[0][0] == 0
    # All boundaries are within the output range
    for start, end in boundaries:
        assert start >= 0
        assert end <= len(f0)
        assert start < end


# ── Tests postfilter ─────────────────────────────────────────────


def test_postfilter_phone2id():
    """Vocabulaire phone2id contient les phones principaux."""
    from lectura_tts_diphone._postfilter import PHONE2ID, N_PHONES
    assert "#" in PHONE2ID
    assert "a" in PHONE2ID
    assert "b" in PHONE2ID
    assert N_PHONES == len(PHONE2ID)


def test_postfilter_parse_diphone_key():
    """Parse diphone key en IDs."""
    from lectura_tts_diphone._postfilter import _parse_diphone_key, PHONE2ID
    a_id, b_id = _parse_diphone_key("b-a")
    assert a_id == PHONE2ID["b"]
    assert b_id == PHONE2ID["a"]


def test_postfilter_parse_silence_key():
    """Parse diphone key avec silence (#)."""
    from lectura_tts_diphone._postfilter import _parse_diphone_key, PHONE2ID
    a_id, b_id = _parse_diphone_key("#-b")
    assert a_id == PHONE2ID["#"]
    assert b_id == PHONE2ID["b"]


def test_postfilter_parse_unknown_phone():
    """Phone inconnu mappe vers 0 (silence)."""
    from lectura_tts_diphone._postfilter import _parse_diphone_key
    a_id, b_id = _parse_diphone_key("X-Y")
    assert a_id == 0
    assert b_id == 0


def test_engine_postfilter_none_by_default():
    """Engine sans modele ONNX a _postfilter = None."""
    from lectura_tts_diphone.engine import DiphoneEngine
    engine = DiphoneEngine()
    assert not hasattr(engine, '_postfilter') or engine._postfilter is None


# ── Tests timbre cepstral ────────────────────────────────────────


def test_extract_timbre_signature():
    """Signature extraite a la bonne forme (1D, n_bins)."""
    import numpy as np
    from lectura_tts_diphone._world import extract_timbre_signature

    sp = np.random.rand(20, 64).astype(np.float64) + 0.01
    sig = extract_timbre_signature(sp)
    assert sig.ndim == 1
    assert sig.shape[0] == 64


def test_apply_timbre_identity():
    """blend=0 et texture=0 ne changent rien."""
    import numpy as np
    from lectura_tts_diphone._world import apply_timbre

    sp = np.random.rand(10, 64).astype(np.float64) + 0.01
    sig = np.zeros(64, dtype=np.float64)
    result = apply_timbre(sp, sig, blend=0.0, texture=0.0)
    np.testing.assert_allclose(result, sp, rtol=1e-6)


def test_apply_timbre_full():
    """blend=1 modifie les coefficients de tilt spectral."""
    import numpy as np
    from lectura_tts_diphone._world import apply_timbre, extract_timbre_signature

    sp = np.random.rand(10, 64).astype(np.float64) + 0.01
    # Signature avec tilt spectral fort
    sig = np.zeros(64, dtype=np.float64)
    sig[0] = 1.0  # energie
    sig[1] = -0.5  # tilt
    sig[2] = -0.3  # tilt

    result = apply_timbre(sp, sig, blend=1.0, texture=0.0)
    # Le resultat doit differer de l'original
    assert not np.allclose(result, sp, rtol=1e-3)


def test_apply_timbre_preserves_formants():
    """Coefficients cepstraux 3-15 (formants) restent inchanges."""
    import numpy as np
    from scipy.fft import dct
    from lectura_tts_diphone._world import apply_timbre

    sp = np.random.rand(10, 64).astype(np.float64) + 0.01
    sig = np.zeros(64, dtype=np.float64)
    sig[0] = 2.0
    sig[1] = -1.0
    sig[20] = 0.5  # texture

    # Extraire les coefficients cepstraux avant
    log_sp = np.log(np.maximum(sp, 1e-10))
    ceps_before = dct(log_sp, type=2, axis=1, norm='ortho')

    result = apply_timbre(sp, sig, blend=1.0, texture=1.0,
                          formant_low=3, formant_high=16)

    # Extraire les coefficients cepstraux apres
    log_result = np.log(np.maximum(result, 1e-10))
    ceps_after = dct(log_result, type=2, axis=1, norm='ortho')

    # Les coefficients 3-15 doivent etre identiques
    np.testing.assert_allclose(
        ceps_after[:, 3:16], ceps_before[:, 3:16], rtol=1e-6)


def test_list_signatures():
    """Signatures pre-calculees accessibles."""
    from lectura_tts_diphone._timbre import list_signatures

    sigs = list_signatures()
    assert isinstance(sigs, list)
    assert len(sigs) >= 3
    assert "neutre" in sigs
    assert "homme" in sigs
    assert "enfant" in sigs


def test_load_signature_builtin():
    """Chargement d'une signature pre-calculee par nom."""
    import numpy as np
    from lectura_tts_diphone._timbre import load_signature

    sig = load_signature("homme")
    assert isinstance(sig, np.ndarray)
    assert sig.ndim == 1
    # Tilt spectral negatif pour voix masculine
    assert sig[1] < 0


def test_load_signature_unknown_raises():
    """Nom inconnu leve ValueError."""
    import pytest
    from lectura_tts_diphone._timbre import load_signature

    with pytest.raises(ValueError, match="Signature inconnue"):
        load_signature("voix_inexistante")


def test_base_f0_scaling():
    """base_f0 ajuste les F0 targets dans synthesize_groups."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode

    # Test que group_info recoit le bon base_f0
    phones = ["b", "a", "l"]
    info_default = {
        "group_idx": 0, "n_groups": 1, "boundary": "period",
        "base_f0": 175.0, "macro_expressivity": 1.0,
    }
    info_low = {
        "group_idx": 0, "n_groups": 1, "boundary": "period",
        "base_f0": 120.0, "macro_expressivity": 1.0,
    }

    f0s_default = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info_default)
    f0s_low = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info_low)

    # F0 avec base_f0=120 doit etre globalement plus bas
    import numpy as np
    assert np.mean(f0s_low) < np.mean(f0s_default)


def test_timbre_signature_dimensions():
    """Les signatures builtin ont la bonne dimension."""
    import numpy as np
    from lectura_tts_diphone._timbre import BUILTIN_SIGNATURES

    for name, sig in BUILTIN_SIGNATURES.items():
        assert isinstance(sig, np.ndarray), f"{name}: pas un ndarray"
        assert sig.ndim == 1, f"{name}: pas 1D"
        assert len(sig) == 400, f"{name}: taille {len(sig)} != 400"
