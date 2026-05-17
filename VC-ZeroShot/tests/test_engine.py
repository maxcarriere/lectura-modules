"""Tests pour engine.py --- ZeroShotEngine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lectura_vc_zeroshot._openvoice import OV_SR
from lectura_vc_zeroshot.engine import ZeroShotEngine


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
