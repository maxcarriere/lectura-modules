"""Tests pour engine.py v2 --- VCEngine facade unifiee."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lectura_vc import RVC_SPEAKERS
from lectura_vc.engine import VCEngine


class TestRetroCompat:
    """Tests de retro-compatibilite des imports."""

    def test_import_creer_engine(self):
        from lectura_vc import creer_engine
        assert callable(creer_engine)

    def test_import_convertir(self):
        from lectura_vc import convertir
        assert callable(convertir)

    def test_import_vcengine(self):
        from lectura_vc import VCEngine
        assert VCEngine is not None

    def test_import_rvc_speakers(self):
        from lectura_vc import RVC_SPEAKERS
        assert isinstance(RVC_SPEAKERS, list)
        assert "ezwa" in RVC_SPEAKERS
        assert len(RVC_SPEAKERS) == 6

    def test_version_v2(self):
        from lectura_vc import __version__
        assert __version__.startswith("2.")


class TestModeResolution:
    def test_auto_with_speaker(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        mode = engine._resolve_mode("ezwa", None, "auto")
        assert mode == "rvc"

    def test_auto_with_reference(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        mode = engine._resolve_mode(None, "ref.wav", "auto")
        assert mode == "zeroshot"

    def test_auto_with_both(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        mode = engine._resolve_mode("ezwa", "ref.wav", "auto")
        assert mode == "cascade"

    def test_auto_with_nothing(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        with pytest.raises(ValueError, match="necessite"):
            engine._resolve_mode(None, None, "auto")

    def test_explicit_mode(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        mode = engine._resolve_mode("ezwa", "ref.wav", "rvc")
        assert mode == "rvc"


class TestConvertValidation:
    def test_zeroshot_no_reference(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        engine.default_speaker = None
        engine._models_dir = None
        engine._locuteurs_engine = None
        engine._zeroshot_engine = MagicMock()

        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="necessite"):
            engine.convert(audio, mode="zeroshot", sr_in=16000)

    def test_unknown_mode(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        engine.default_speaker = None
        engine._models_dir = None
        engine._locuteurs_engine = None
        engine._zeroshot_engine = None

        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="inconnu"):
            engine.convert(audio, mode="invalid", sr_in=16000)

    def test_rvc_delegates_to_locuteurs(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        engine.default_speaker = None
        engine._models_dir = None
        engine._zeroshot_engine = None

        mock_loc = MagicMock()
        mock_loc.convert.return_value = (np.zeros(48000, dtype=np.float32), 48000)
        engine._locuteurs_engine = mock_loc

        audio = np.random.randn(16000).astype(np.float32)
        result, sr = engine.convert(audio, speaker="ezwa", mode="rvc", sr_in=16000)

        assert sr == 48000
        mock_loc.convert.assert_called_once()

    def test_zeroshot_delegates_to_zeroshot(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        engine.default_speaker = None
        engine._models_dir = None
        engine._locuteurs_engine = None

        mock_zs = MagicMock()
        mock_zs.convert.return_value = (np.zeros(22050, dtype=np.float32), 22050)
        engine._zeroshot_engine = mock_zs

        audio = np.random.randn(16000).astype(np.float32)
        ref = np.random.randn(16000).astype(np.float32)
        result, sr = engine.convert(audio, reference=ref, mode="zeroshot", sr_in=16000, ref_sr=16000)

        assert sr == 22050
        mock_zs.convert.assert_called_once()

    def test_cascade_delegates_to_both(self):
        engine = VCEngine.__new__(VCEngine)
        engine.mode = "auto"
        engine.default_speaker = None
        engine._models_dir = None

        mock_loc = MagicMock()
        mock_loc.convert.return_value = (np.zeros(48000, dtype=np.float32), 48000)
        engine._locuteurs_engine = mock_loc

        mock_zs = MagicMock()
        mock_zs.convert.return_value = (np.zeros(22050, dtype=np.float32), 22050)
        engine._zeroshot_engine = mock_zs

        audio = np.random.randn(16000).astype(np.float32)
        ref = np.random.randn(16000).astype(np.float32)
        result, sr = engine.convert(
            audio, speaker="ezwa", reference=ref,
            mode="cascade", sr_in=16000, ref_sr=16000,
        )

        assert sr == 22050
        mock_loc.convert.assert_called_once()
        mock_zs.convert.assert_called_once()
