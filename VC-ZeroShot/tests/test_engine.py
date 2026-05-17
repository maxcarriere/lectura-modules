"""Tests pour engine.py --- ZeroShotEngine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lectura_vc_zeroshot._openvoice import OV_SR
from lectura_vc_zeroshot._presets import (
    blend_presets,
    has_preset,
    list_presets,
    load_preset,
)
from lectura_vc_zeroshot.engine import ZeroShotEngine, blend_se, _is_se


@pytest.fixture
def models_dir(tmp_path):
    """Cree un repertoire de modeles factice avec fichiers OpenVoice."""
    (tmp_path / "openvoice_se.onnx").write_bytes(b"fake")
    (tmp_path / "openvoice_vc.onnx").write_bytes(b"fake")
    return tmp_path


class TestZeroShotEngineInit:
    def test_basic_init(self, models_dir):
        engine = ZeroShotEngine(models_dir=models_dir)
        assert engine.models_dir == models_dir
        assert engine._converter is None  # lazy

    def test_missing_models_dir(self, tmp_path, monkeypatch):
        """Aucun modele trouve -> FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.delenv("LECTURA_MODELS_DIR", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "fakehome")
        # Patch le fallback package modeles pour qu'il ne trouve rien
        monkeypatch.setattr(
            "lectura_vc_zeroshot._chargeur._PACKAGE_MODELS",
            tmp_path / "nopkg",
        )
        with pytest.raises(FileNotFoundError):
            ZeroShotEngine(models_dir=nonexistent)

    def test_incomplete_models(self, tmp_path):
        """Repertoire sans les modeles OpenVoice -> FileNotFoundError."""
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "dummy.txt").write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="Modeles OpenVoice manquants"):
            ZeroShotEngine(models_dir=tmp_path)


class TestSrOverrideFactor:
    """Teste le calcul du factor de decalage formants."""

    def test_neutral_no_override(self):
        """sr_override=None -> pas de trick, factor=1."""
        # Le factor est implicit : OV_SR / OV_SR = 1
        factor = OV_SR / OV_SR
        assert factor == 1.0

    def test_acute_override(self):
        """sr_override=11025 -> factor 2x (formants montes)."""
        sr_override = 11025
        factor = OV_SR / sr_override
        assert abs(factor - 2.0) < 0.01

    def test_grave_override(self):
        """sr_override=44100 -> factor 0.5x (formants baisses)."""
        sr_override = 44100
        factor = OV_SR / sr_override
        assert abs(factor - 0.5) < 0.01


class TestConvertValidation:
    def test_convert_calls_converter(self, models_dir):
        """Verifie que convert() appelle le converter sous-jacent."""
        engine = ZeroShotEngine(models_dir=models_dir)

        # Mock le converter
        mock_conv = MagicMock()
        mock_se = np.zeros((1, 256, 1), dtype=np.float32)
        mock_audio = np.zeros(OV_SR, dtype=np.float32)
        mock_conv.extract_se.return_value = mock_se
        mock_conv.convert.return_value = (mock_audio, OV_SR)
        engine._converter = mock_conv

        audio = np.random.randn(16000).astype(np.float32)
        ref = np.random.randn(16000).astype(np.float32)

        result, sr = engine.convert(audio, ref, sr_in=16000, ref_sr=16000)

        assert sr == OV_SR
        assert mock_conv.extract_se.call_count == 2  # src + tgt
        assert mock_conv.convert.call_count == 1

    def test_convert_with_sr_override(self, models_dir):
        """Verifie que sr_override active le trick SR."""
        engine = ZeroShotEngine(models_dir=models_dir)

        mock_conv = MagicMock()
        mock_se = np.zeros((1, 256, 1), dtype=np.float32)
        mock_audio = np.zeros(OV_SR, dtype=np.float32)
        mock_conv.extract_se.return_value = mock_se
        mock_conv.convert.return_value = (mock_audio, OV_SR)
        engine._converter = mock_conv

        audio = np.random.randn(16000).astype(np.float32)
        ref = np.random.randn(16000).astype(np.float32)

        result, sr = engine.convert(
            audio, ref, sr_in=16000, ref_sr=16000, sr_override=11025,
        )

        assert sr == OV_SR
        # src SE via extract_se, tgt SE via _extract_se_with_sr_trick
        assert mock_conv.extract_se.call_count >= 1
        assert mock_conv.convert.call_count == 1


class TestExtractSe:
    def test_extract_se_delegates(self, models_dir):
        """extract_se() delegue au converter."""
        engine = ZeroShotEngine(models_dir=models_dir)

        mock_conv = MagicMock()
        mock_se = np.zeros((1, 256, 1), dtype=np.float32)
        mock_conv.extract_se.return_value = mock_se
        engine._converter = mock_conv

        audio = np.random.randn(16000).astype(np.float32)
        se = engine.extract_se(audio, sr=16000)

        np.testing.assert_array_equal(se, mock_se)
        mock_conv.extract_se.assert_called_once()


class TestConvertFromSe:
    def test_convert_from_se_delegates(self, models_dir):
        """convert_from_se() delegue au converter."""
        engine = ZeroShotEngine(models_dir=models_dir)

        mock_conv = MagicMock()
        mock_audio = np.zeros(OV_SR, dtype=np.float32)
        mock_conv.convert.return_value = (mock_audio, OV_SR)
        engine._converter = mock_conv

        audio = np.random.randn(16000).astype(np.float32)
        src_se = np.zeros((1, 256, 1), dtype=np.float32)
        tgt_se = np.ones((1, 256, 1), dtype=np.float32)

        result, sr = engine.convert_from_se(audio, src_se, tgt_se, sr_in=16000)
        assert sr == OV_SR
        mock_conv.convert.assert_called_once()


# -- Presets -------------------------------------------------------------------


class TestPresets:
    def test_list_presets(self):
        """list_presets() retourne les 6 locuteurs."""
        presets = list_presets()
        assert len(presets) >= 6
        assert "siwis" in presets
        assert "bernard" in presets

    def test_has_preset(self):
        assert has_preset("siwis") is True
        assert has_preset("toto_inconnu") is False

    def test_load_preset_shape(self):
        se = load_preset("siwis")
        assert se.shape == (1, 256, 1)
        assert se.dtype == np.float32

    def test_load_preset_missing(self):
        with pytest.raises(FileNotFoundError, match="Preset 'xyz'"):
            load_preset("xyz")

    def test_blend_presets_equal(self):
        blend = blend_presets({"siwis": 1.0, "nadine": 1.0})
        se_s = load_preset("siwis")
        se_n = load_preset("nadine")
        expected = (se_s + se_n) / 2
        np.testing.assert_allclose(blend, expected, atol=1e-6)

    def test_blend_presets_weighted(self):
        blend = blend_presets({"siwis": 0.5, "nadine": 0.3, "ezwa": 0.2})
        se_s = load_preset("siwis")
        se_n = load_preset("nadine")
        se_e = load_preset("ezwa")
        expected = se_s * 0.5 + se_n * 0.3 + se_e * 0.2
        np.testing.assert_allclose(blend, expected, atol=1e-6)

    def test_blend_presets_empty_raises(self):
        with pytest.raises(ValueError, match="Au moins un"):
            blend_presets({})


# -- blend_se ------------------------------------------------------------------


class TestBlendSe:
    def test_blend_equal_weights(self):
        se1 = np.ones((1, 256, 1), dtype=np.float32) * 2
        se2 = np.ones((1, 256, 1), dtype=np.float32) * 4
        result = blend_se([se1, se2])
        np.testing.assert_allclose(result, np.full((1, 256, 1), 3.0), atol=1e-6)

    def test_blend_custom_weights(self):
        se1 = np.ones((1, 256, 1), dtype=np.float32) * 0
        se2 = np.ones((1, 256, 1), dtype=np.float32) * 10
        result = blend_se([se1, se2], weights=[0.8, 0.2])
        expected = se2 * 0.2  # 0*0.8 + 10*0.2 = 2
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_blend_single(self):
        se = np.random.randn(1, 256, 1).astype(np.float32)
        result = blend_se([se])
        np.testing.assert_array_equal(result, se)

    def test_blend_mismatched_lengths_raises(self):
        se = np.zeros((1, 256, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Nombre de poids"):
            blend_se([se, se], weights=[1.0])


# -- _is_se helper -------------------------------------------------------------


class TestIsSe:
    def test_se_shape(self):
        assert _is_se(np.zeros((1, 256, 1))) is True

    def test_audio_shape(self):
        assert _is_se(np.zeros(16000)) is False

    def test_wrong_dim(self):
        assert _is_se(np.zeros((256,))) is False
        assert _is_se(np.zeros((1, 128, 1))) is False


# -- resolve_target_se ---------------------------------------------------------


class TestResolveTargetSe:
    def test_resolve_preset_name(self, models_dir):
        """resolve_target_se('siwis') charge le preset."""
        engine = ZeroShotEngine(models_dir=models_dir)
        se = engine.resolve_target_se("siwis")
        assert se.shape == (1, 256, 1)
        expected = load_preset("siwis")
        np.testing.assert_array_equal(se, expected)

    def test_resolve_direct_se(self, models_dir):
        """Un SE (1,256,1) passe directement est retourne tel quel."""
        engine = ZeroShotEngine(models_dir=models_dir)
        direct_se = np.random.randn(1, 256, 1).astype(np.float32)
        result = engine.resolve_target_se(direct_se)
        np.testing.assert_array_equal(result, direct_se)

    def test_resolve_list(self, models_dir):
        """Liste de presets → moyenne des SE."""
        engine = ZeroShotEngine(models_dir=models_dir)
        se = engine.resolve_target_se(["siwis", "nadine"])
        expected = blend_se([load_preset("siwis"), load_preset("nadine")])
        np.testing.assert_allclose(se, expected, atol=1e-6)

    def test_resolve_dict_weighted(self, models_dir):
        """Dict pondere → blend."""
        engine = ZeroShotEngine(models_dir=models_dir)
        se = engine.resolve_target_se({"siwis": 0.7, "bernard": 0.3})
        expected = blend_presets({"siwis": 0.7, "bernard": 0.3})
        np.testing.assert_allclose(se, expected, atol=1e-6)

    def test_resolve_audio_array(self, models_dir):
        """Un ndarray 1D est traite comme audio → extract_se()."""
        engine = ZeroShotEngine(models_dir=models_dir)
        mock_conv = MagicMock()
        mock_se = np.ones((1, 256, 1), dtype=np.float32)
        mock_conv.extract_se.return_value = mock_se
        engine._converter = mock_conv

        audio = np.random.randn(16000).astype(np.float32)
        result = engine.resolve_target_se(audio, ref_sr=16000)
        np.testing.assert_array_equal(result, mock_se)

    def test_engine_available_presets(self, models_dir):
        """available_presets() retourne la liste des presets."""
        presets = ZeroShotEngine.available_presets()
        assert isinstance(presets, list)
        assert "siwis" in presets

    def test_engine_get_preset_se(self, models_dir):
        """get_preset_se() charge un preset."""
        se = ZeroShotEngine.get_preset_se("siwis")
        assert se.shape == (1, 256, 1)

    def test_engine_blend_preset_se(self, models_dir):
        """blend_preset_se() melange des presets."""
        se = ZeroShotEngine.blend_preset_se({"siwis": 0.5, "nadine": 0.5})
        assert se.shape == (1, 256, 1)

    def test_convert_accepts_preset_name(self, models_dir):
        """convert() avec reference='siwis' utilise le preset."""
        engine = ZeroShotEngine(models_dir=models_dir)
        mock_conv = MagicMock()
        mock_se = np.zeros((1, 256, 1), dtype=np.float32)
        mock_audio = np.zeros(OV_SR, dtype=np.float32)
        mock_conv.extract_se.return_value = mock_se
        mock_conv.convert.return_value = (mock_audio, OV_SR)
        engine._converter = mock_conv

        audio = np.random.randn(16000).astype(np.float32)
        result, sr = engine.convert(audio, reference="siwis", sr_in=16000)
        assert sr == OV_SR
        # extract_se appele 1 seule fois (src) — tgt vient du preset
        assert mock_conv.extract_se.call_count == 1

    def test_convert_accepts_dict(self, models_dir):
        """convert() avec reference dict utilise le blend."""
        engine = ZeroShotEngine(models_dir=models_dir)
        mock_conv = MagicMock()
        mock_se = np.zeros((1, 256, 1), dtype=np.float32)
        mock_audio = np.zeros(OV_SR, dtype=np.float32)
        mock_conv.extract_se.return_value = mock_se
        mock_conv.convert.return_value = (mock_audio, OV_SR)
        engine._converter = mock_conv

        audio = np.random.randn(16000).astype(np.float32)
        result, sr = engine.convert(
            audio,
            reference={"siwis": 0.5, "nadine": 0.5},
            sr_in=16000,
        )
        assert sr == OV_SR
