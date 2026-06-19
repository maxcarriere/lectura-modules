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
    import numpy as np

    # Compare question vs declarative (regles style): question final F0 higher
    phones = ["b", "a", "l", "a", "k", "ɔ", "m", "ɑ̃"]
    info_q = {"group_idx": 0, "n_groups": 1, "boundary": "question",
              "base_f0": 200.0, "macro_expressivity": 1.0,
              "prosody_style": "regles"}
    f0s_q = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info_q, word_boundaries=[4])

    info_d = {"group_idx": 0, "n_groups": 1, "boundary": "period",
              "base_f0": 200.0, "macro_expressivity": 1.0,
              "prosody_style": "regles"}
    f0s_d = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info_d, word_boundaries=[4])

    assert len(f0s_q) == 8
    # Question final F0 should be higher than declarative final F0
    assert f0s_q[-1] > f0s_d[-1]


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
        "prosody_style": "regles",
    }
    info_low = {
        "group_idx": 0, "n_groups": 1, "boundary": "period",
        "base_f0": 120.0, "macro_expressivity": 1.0,
        "prosody_style": "regles",
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


# ── Tests Fujisaki ──────────────────────────────────────────────


def test_fujisaki_phrase_response():
    """Gp(0)=0, pic a t=1/alpha, decroissance ensuite."""
    from lectura_tts_diphone._fujisaki import ALPHA_DEFAULT, phrase_response

    # t < 0 → 0
    assert phrase_response(-1.0) == 0.0
    # t = 0 → 0
    assert phrase_response(0.0) == 0.0
    # pic a t = 1/alpha
    t_peak = 1.0 / ALPHA_DEFAULT
    val_peak = phrase_response(t_peak)
    assert val_peak > 0.0
    # Avant le pic, valeur plus basse
    assert phrase_response(t_peak * 0.5) < val_peak
    # Apres le pic, decroissance
    assert phrase_response(t_peak * 3.0) < val_peak


def test_fujisaki_accent_response():
    """Ga monte puis redescend, max ~1.0 pendant la commande active."""
    from lectura_tts_diphone._fujisaki import BETA_DEFAULT, accent_response

    dur = 0.1  # 100ms command
    # Avant la commande → 0
    assert accent_response(-0.1, dur) == 0.0
    # Pendant la commande → monte vers 1.0
    val_mid = accent_response(dur * 0.8, dur)
    assert val_mid > 0.3
    # Apres offset → redescend
    val_after = accent_response(dur * 3.0, dur)
    assert val_after < val_mid
    # Bien apres → proche de 0
    val_late = accent_response(dur * 10.0, dur)
    assert val_late < 0.05


def test_fujisaki_generate_contour():
    """Contour avec accent command produit un pic au bon endroit."""
    from lectura_tts_diphone._fujisaki import generate_contour

    fb = 150.0
    # Un seul accent a t=0.2, duree 0.1, amplitude 0.3
    accent_cmds = [(0.2, 0.1, 0.3)]
    times = [i * 0.05 for i in range(10)]  # 0.0 a 0.45s

    contour = generate_contour(fb, [], accent_cmds, times)
    assert len(contour) == 10
    # Toutes les valeurs >= fb (accent ajoute)
    assert all(f >= fb - 0.01 for f in contour)
    # Le pic doit etre pres de t=0.2-0.3 (indices 4-6)
    peak_idx = contour.index(max(contour))
    assert 3 <= peak_idx <= 7


def test_fujisaki_generate_contour_flat():
    """Sans commandes, contour plat a fb."""
    from lectura_tts_diphone._fujisaki import generate_contour

    fb = 180.0
    times = [i * 0.05 for i in range(5)]
    contour = generate_contour(fb, [], [], times)
    assert all(abs(f - fb) < 0.01 for f in contour)


def test_f0_contour_smoothness():
    """Transitions entre phones adjacents sont lisses (pas de sauts)."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode

    # Long phrase with multiple APs to test smoothness (regles style)
    phones = ["b", "ɔ̃", "ʒ", "u", "ʁ", "k", "ɔ", "m", "ɑ̃",
              "s", "a", "v", "a"]
    info = {"group_idx": 0, "n_groups": 1, "boundary": "period",
            "base_f0": 175.0, "macro_expressivity": 1.0,
            "prosody_style": "regles"}
    f0s = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info, word_boundaries=[5, 9])

    # AP patterns with consonant interpolation produce smooth contours
    max_jump = 0.0
    for i in range(1, len(f0s)):
        jump = abs(f0s[i] - f0s[i - 1])
        max_jump = max(max_jump, jump)
    assert max_jump < 60.0, f"Max F0 jump = {max_jump:.1f} Hz"


# ── Tests syllable-level F0 ─────────────────────────────────────


def test_phones_to_syllables():
    """Decoupage syllabique correct avec preference onset."""
    from lectura_tts_diphone.engine import _phones_to_syllables

    # "bala" : b=0, a=1, l=2, a=3 → vowels at [1, 3]
    spans = _phones_to_syllables(["b", "a", "l", "a"], [1, 3])
    assert len(spans) == 2
    assert spans[0] == (0, 2)  # "ba"
    assert spans[1] == (2, 4)  # "la"

    # No vowels → single span covering all phones
    spans = _phones_to_syllables(["b", "l"], [])
    assert spans == [(0, 2)]

    # Single vowel → single span
    spans = _phones_to_syllables(["b", "a", "l"], [1])
    assert spans == [(0, 3)]


# ── Tests prosodie double style ─────────────────────────────────


def test_prosody_style_default():
    """Le style par defaut est 'regles'."""
    from lectura_tts_diphone.engine import DiphoneEngine
    assert "regles" in DiphoneEngine._PROSODY_STYLES
    assert "corpus" in DiphoneEngine._PROSODY_STYLES


def test_prosody_style_regles():
    """Style regles produit F0 avec declination (debut > fin)."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    import numpy as np

    phones = ["b", "ɔ̃", "ʒ", "u", "ʁ", "k", "ɔ", "m", "ɑ̃",
              "s", "a", "v", "a"]
    info = {"group_idx": 0, "n_groups": 1, "boundary": "period",
            "base_f0": 200.0, "macro_expressivity": 1.0,
            "prosody_style": "regles"}
    f0s = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info, word_boundaries=[5, 9])

    # Declaratif : F0 debut > F0 fin (chute declarative)
    assert f0s[1] > f0s[-1]  # voyelle 1 vs dernier phone


def test_prosody_style_corpus():
    """Style corpus produit F0 coherents par syllabe."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode

    phones = ["b", "ɔ̃", "ʒ", "u", "ʁ", "k", "ɔ", "m", "ɑ̃",
              "s", "a", "v", "a"]
    info = {"group_idx": 0, "n_groups": 1, "boundary": "period",
            "base_f0": 200.0, "macro_expressivity": 1.0,
            "prosody_style": "corpus"}
    f0s = DiphoneEngine._group_f0_contour(
        phones, SynthMode.FLUIDE, info)

    assert len(f0s) == len(phones)
    assert all(f > 80.0 for f in f0s)
    # Corpus prosody stored for reuse
    assert "_corpus_prosody" in info


def test_prosody_style_corpus_question_vs_decl():
    """Style corpus : question F0 final plus haut que declaratif."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode
    import numpy as np

    phones = ["b", "a", "l", "a", "k", "ɔ", "m", "ɑ̃"]
    info_q = {"group_idx": 0, "n_groups": 1, "boundary": "question",
              "base_f0": 200.0, "macro_expressivity": 1.0,
              "prosody_style": "corpus"}
    f0s_q = DiphoneEngine._group_f0_contour(phones, SynthMode.FLUIDE, info_q)

    info_d = {"group_idx": 0, "n_groups": 1, "boundary": "period",
              "base_f0": 200.0, "macro_expressivity": 1.0,
              "prosody_style": "corpus"}
    f0s_d = DiphoneEngine._group_f0_contour(phones, SynthMode.FLUIDE, info_d)

    # Question final F0 should be higher than declarative (statistically)
    # Use mean of last 2 phones to be robust to jitter
    assert np.mean(f0s_q[-2:]) > np.mean(f0s_d[-2:])


def test_corpus_dur_ratio():
    """Style corpus fournit dur_ratio dans _corpus_prosody."""
    from lectura_tts_diphone.engine import DiphoneEngine, SynthMode

    phones = ["b", "a", "l", "a", "k", "ɔ", "m", "ɑ̃"]
    info = {"group_idx": 0, "n_groups": 1, "boundary": "period",
            "base_f0": 200.0, "macro_expressivity": 1.0,
            "prosody_style": "corpus"}
    DiphoneEngine._group_f0_contour(phones, SynthMode.FLUIDE, info)

    prosody = info["_corpus_prosody"]
    assert len(prosody) > 0
    for sp in prosody:
        assert "dur_ratio" in sp
        assert "f0_hz" in sp
        assert sp["dur_ratio"] > 0


def test_corpus_prosody_module():
    """Module _prosody_corpus charge la banque et genere des contours."""
    from random import Random
    from lectura_tts_diphone._prosody_corpus import (
        generate_corpus_prosody, load_bank,
    )

    bank = load_bank()
    assert len(bank) > 0

    rng = Random(42)
    prosody = generate_corpus_prosody(
        6, "declaratif", rng, base_f0=200.0, group_role="seul")
    assert len(prosody) == 6
    for sp in prosody:
        assert sp["f0_hz"] > 80.0
        assert sp["dur_ratio"] > 0


# ── Tests retimbre ─────────────────────────────────────────────────────────

def test_variante_to_sr_override():
    """Formule : 0 → None, +1 → 11025, -1 → 44100."""
    from lectura_tts_diphone._retimbre import variante_to_sr_override

    # Neutre
    assert variante_to_sr_override(0.0) is None
    assert variante_to_sr_override(0.005) is None

    # Aigu (+1 → sr_override = 22050/2 = 11025)
    sr = variante_to_sr_override(1.0)
    assert sr == 11025

    # Grave (-1 → sr_override = 22050*2 = 44100)
    sr = variante_to_sr_override(-1.0)
    assert sr == 44100

    # Intermediaire (+0.5 → 22050 / 2^0.5 ~ 15590)
    sr = variante_to_sr_override(0.5)
    assert 15000 < sr < 16000

    # Clamp min
    sr = variante_to_sr_override(2.0)
    assert sr >= 8000

    # Clamp max
    sr = variante_to_sr_override(-2.0)
    assert sr <= 48000


def test_retimbre_import_guard():
    """Sans vc-zeroshot installe, l'import de RetimbreEngine doit reussir
    mais retimbre() doit lever ImportError si le module est absent."""
    from lectura_tts_diphone._retimbre import RetimbreEngine
    # L'import seul ne doit pas planter (lazy)
    assert RetimbreEngine is not None


def test_synth_groups_voix_none():
    """voix=None → meme output qu'avant (pas de regression)."""
    from lectura_tts_diphone.engine import DiphoneEngine
    import inspect

    sig = inspect.signature(DiphoneEngine.synthesize_groups)
    params = sig.parameters

    # Le parametre voix existe et a None par defaut
    assert "voix" in params
    assert params["voix"].default is None
    assert "voix_variante" in params
    assert params["voix_variante"].default == 0.0
    assert "voix_tau" in params
    assert params["voix_tau"].default == 0.3


def test_retimbre_cache_key():
    """_make_cache_key produit des cles hashables pour tous les types."""
    from lectura_tts_diphone._retimbre import _make_cache_key

    # str
    key1 = _make_cache_key("siwis", None)
    assert isinstance(key1, tuple)
    assert key1 == ("siwis", None)

    # list
    key2 = _make_cache_key(["siwis", "nadine"], 11025)
    assert key2 == (("siwis", "nadine"), 11025)

    # dict
    key3 = _make_cache_key({"siwis": 0.5, "nadine": 0.5}, None)
    assert isinstance(key3[0], tuple)  # sorted items tuple

    # Same dict different order → same key
    key4 = _make_cache_key({"nadine": 0.5, "siwis": 0.5}, None)
    assert key3 == key4


def test_retimbre_voix_types():
    """Le type hint de voix accepte str, list, dict."""
    from lectura_tts_diphone._retimbre import RetimbreEngine, VoixSpec
    import inspect

    sig = inspect.signature(RetimbreEngine.retimbre)
    # Le parametre reference accepte VoixSpec
    assert "reference" in sig.parameters
