"""Tests pour les engines d'inference (mock/unit)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestOnnxTTSEngineMatcha:
    """Tests pour OnnxTTSEngine avec modele Matcha-Conformer mocke."""

    def _make_engine(self, tmp_path):
        """Cree un engine avec un repertoire de modeles Matcha factice."""
        from lectura_tts_monospeaker.inference_onnx import OnnxTTSEngine

        config = {
            "model": {
                "type": "matcha-conformer",
                "d_model": 256,
                "n_mels": 80,
                "n_style_dims": 5,
                "n_ode_steps": 4,
            },
            "audio": {"sample_rate": 22050, "hop_length": 256},
            "enhance": {
                "spectral_alpha": 0.20,
                "temporal_alpha": 0.20,
                "noise_gate_threshold": -8.0,
                "silence_val": -11.5,
                "fade_frames": 5,
            },
            "embeddings": {
                "pitch_emb_weight": [0.01] * 256,
                "pitch_emb_bias": [0.0] * 256,
                "energy_emb_weight": [0.01] * 256,
                "energy_emb_bias": [0.0] * 256,
            },
        }
        vocab_data = {
            "vocab": ["<PAD>", "<UNK>", "#", "|", "b", "\u0254\u0303", "\u0292", "u", "\u0281"],
            "phone2id": {"<PAD>": 0, "<UNK>": 1, "#": 2, "|": 3,
                         "b": 4, "\u0254\u0303": 5, "\u0292": 6, "u": 7, "\u0281": 8},
        }

        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "phoneme_vocab.json").write_text(json.dumps(vocab_data))
        (tmp_path / "matcha_encoder.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "matcha_unet.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "hifigan.onnx").write_bytes(b"FAKE_ONNX")

        return OnnxTTSEngine(tmp_path)

    @patch("onnxruntime.InferenceSession")
    def test_synthesize_phonemes_matcha(self, mock_session_cls, tmp_path):
        """Verifie que synthesize_phonemes fonctionne avec le chemin Matcha."""
        engine = self._make_engine(tmp_path)

        T = 7  # SIL + 5 phones + SIL
        D = 256

        # Mock encoder (Matcha)
        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]

        # Mock UNet (Matcha) — dynamic T_mel based on input
        mock_unet = MagicMock()
        def _unet_run(_, inputs):
            t_mel = inputs["x_t"].shape[2]
            return [np.random.randn(1, 80, t_mel).astype(np.float32) - 5.0]
        mock_unet.run.side_effect = _unet_run

        # Mock hifigan — dynamic T_audio based on mel length
        mock_hifigan = MagicMock()
        def _hifi_run(_, inputs):
            t_mel = inputs["mel"].shape[2]
            return [np.random.randn(1, 1, t_mel * 256).astype(np.float32) * 0.5]
        mock_hifigan.run.side_effect = _hifi_run

        mock_session_cls.side_effect = [mock_encoder, mock_unet, mock_hifigan]

        result = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281b")

        assert result.sample_rate == 22050
        assert isinstance(result.samples, np.ndarray)
        assert result.samples.dtype == np.float32
        # UNet should have been called n_ode_steps=4 times
        assert mock_unet.run.call_count == 4

    @patch("onnxruntime.InferenceSession")
    def test_synthesize_phonemes_style_preset(self, mock_session_cls, tmp_path):
        """Verifie que les style presets s'appliquent correctement."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 256

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        mock_unet = MagicMock()
        def _unet_run(_, inputs):
            t_mel = inputs["x_t"].shape[2]
            return [np.random.randn(1, 80, t_mel).astype(np.float32) - 5.0]
        mock_unet.run.side_effect = _unet_run
        mock_hifigan = MagicMock()
        def _hifi_run(_, inputs):
            t_mel = inputs["mel"].shape[2]
            return [np.random.randn(1, 1, t_mel * 256).astype(np.float32) * 0.5]
        mock_hifigan.run.side_effect = _hifi_run
        mock_session_cls.side_effect = [mock_encoder, mock_unet, mock_hifigan]

        result = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281b", style="narratif")

        assert result.sample_rate == 22050
        assert len(result.samples) > 0

        # Verify style_vector was passed to encoder
        call_args = mock_encoder.run.call_args[0][1]
        assert "style_vector" in call_args
        style_passed = call_args["style_vector"]
        # Should match narratif preset
        np.testing.assert_allclose(style_passed[0], [0.0, -0.2, -0.2, 0.0, 0.0])

    @patch("onnxruntime.InferenceSession")
    def test_synthesize_phonemes_n_ode_steps(self, mock_session_cls, tmp_path):
        """Verifie que n_ode_steps est respecte."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 256

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        mock_unet = MagicMock()
        def _unet_run(_, inputs):
            t_mel = inputs["x_t"].shape[2]
            return [np.random.randn(1, 80, t_mel).astype(np.float32) - 5.0]
        mock_unet.run.side_effect = _unet_run
        mock_hifigan = MagicMock()
        def _hifi_run(_, inputs):
            t_mel = inputs["mel"].shape[2]
            return [np.random.randn(1, 1, t_mel * 256).astype(np.float32) * 0.5]
        mock_hifigan.run.side_effect = _hifi_run
        mock_session_cls.side_effect = [mock_encoder, mock_unet, mock_hifigan]

        result = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281b", n_ode_steps=8)

        # UNet should have been called 8 times
        assert mock_unet.run.call_count == 8

    @patch("onnxruntime.InferenceSession")
    def test_resolve_style_priority(self, mock_session_cls, tmp_path):
        """Verifie la priorite : explicit vector > named preset > neutre."""
        engine = self._make_engine(tmp_path)
        # Force load config
        engine._config = json.loads((tmp_path / "config.json").read_text())

        # Explicit vector takes priority
        sv, overrides = engine._resolve_style(
            style="narratif", style_vector=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert sv == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert overrides == {}

        # Named preset
        sv, overrides = engine._resolve_style(style="dialogue")
        assert sv == [0.3, 0.2, 0.2, 0.0, 1.0]
        assert "duration_scale" in overrides

        # Default neutre
        sv, overrides = engine._resolve_style()
        assert sv == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert overrides == {}


class TestOnnxTTSEngineFastPitch:
    """Tests pour OnnxTTSEngine avec modele FastPitch legacy mocke."""

    def _make_engine(self, tmp_path):
        """Cree un engine avec un repertoire de modeles FastPitch factice."""
        from lectura_tts_monospeaker.inference_onnx import OnnxTTSEngine

        config = {
            "model": {"d_model": 128, "n_mels": 80},
            "audio": {"sample_rate": 22050, "hop_length": 256},
            "enhance": {
                "spectral_alpha": 0.20,
                "temporal_alpha": 0.20,
                "noise_gate_threshold": -8.0,
                "silence_val": -11.5,
                "fade_frames": 5,
            },
            "embeddings": {
                "pitch_emb_weight": [0.01] * 128,
                "pitch_emb_bias": [0.0] * 128,
                "energy_emb_weight": [0.01] * 128,
                "energy_emb_bias": [0.0] * 128,
            },
        }
        vocab_data = {
            "vocab": ["<PAD>", "<UNK>", "#", "|", "b", "\u0254\u0303", "\u0292", "u", "\u0281"],
            "phone2id": {"<PAD>": 0, "<UNK>": 1, "#": 2, "|": 3,
                         "b": 4, "\u0254\u0303": 5, "\u0292": 6, "u": 7, "\u0281": 8},
        }

        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "phoneme_vocab.json").write_text(json.dumps(vocab_data))
        (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "hifigan.onnx").write_bytes(b"FAKE_ONNX")

        return OnnxTTSEngine(tmp_path)

    @patch("onnxruntime.InferenceSession")
    def test_synthesize_phonemes_fastpitch(self, mock_session_cls, tmp_path):
        """Verifie que le chemin FastPitch legacy fonctionne encore."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 128

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]

        mock_decoder = MagicMock()
        def _dec_run(_, inputs):
            t_mel = inputs["decoder_in"].shape[1]
            return [np.random.randn(1, 80, t_mel).astype(np.float32) - 5.0]
        mock_decoder.run.side_effect = _dec_run

        mock_hifigan = MagicMock()
        def _hifi_run(_, inputs):
            t_mel = inputs["mel"].shape[2]
            return [np.random.randn(1, 1, t_mel * 256).astype(np.float32) * 0.5]
        mock_hifigan.run.side_effect = _hifi_run

        mock_session_cls.side_effect = [mock_encoder, mock_decoder, mock_hifigan]

        result = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281b")

        assert result.sample_rate == 22050
        assert isinstance(result.samples, np.ndarray)
        assert result.samples.dtype == np.float32

    @patch("onnxruntime.InferenceSession")
    def test_fastpitch_style_overrides_prosody_only(self, mock_session_cls, tmp_path):
        """En mode FastPitch, les presets appliquent les overrides prosodiques
        mais pas le style_vector (pas supporte par le modele)."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 128

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        mock_decoder = MagicMock()
        def _dec_run(_, inputs):
            t_mel = inputs["decoder_in"].shape[1]
            return [np.random.randn(1, 80, t_mel).astype(np.float32) - 5.0]
        mock_decoder.run.side_effect = _dec_run
        mock_hifigan = MagicMock()
        def _hifi_run(_, inputs):
            t_mel = inputs["mel"].shape[2]
            return [np.random.randn(1, 1, t_mel * 256).astype(np.float32) * 0.5]
        mock_hifigan.run.side_effect = _hifi_run
        mock_session_cls.side_effect = [mock_encoder, mock_decoder, mock_hifigan]

        # Should not crash even with style preset
        result = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281b", style="narratif")

        assert len(result.samples) > 0
        # FastPitch encoder should NOT receive style_vector
        call_args = mock_encoder.run.call_args[0][1]
        assert "style_vector" not in call_args


class TestApiTTSEngine:
    """Tests pour ApiTTSEngine."""

    def test_init_defaults(self):
        from lectura_tts_monospeaker.inference_api import ApiTTSEngine

        engine = ApiTTSEngine()
        assert "lectura.world" in engine._url

    def test_init_custom(self):
        from lectura_tts_monospeaker.inference_api import ApiTTSEngine

        engine = ApiTTSEngine(api_url="http://localhost:8000", api_key="test-key")
        assert engine._url == "http://localhost:8000"
        assert engine._key == "test-key"


class TestStylePresets:
    """Tests pour les presets de style."""

    def test_all_presets_exist(self):
        from lectura_tts_monospeaker.inference_onnx import STYLE_PRESETS

        expected = {"neutre", "narratif", "dialogue", "expressif",
                    "meditatif", "rapide", "lent"}
        assert set(STYLE_PRESETS.keys()) == expected

    def test_preset_structure(self):
        from lectura_tts_monospeaker.inference_onnx import STYLE_PRESETS

        for name, preset in STYLE_PRESETS.items():
            assert "style_vector" in preset, f"{name} manque style_vector"
            assert len(preset["style_vector"]) == 5, \
                f"{name} style_vector should be 5-dim"
            for key in ("pitch_range", "energy_scale",
                        "duration_scale", "pause_scale"):
                assert key in preset, f"{name} manque {key}"

    def test_neutre_is_zero(self):
        from lectura_tts_monospeaker.inference_onnx import STYLE_PRESETS

        neutre = STYLE_PRESETS["neutre"]
        assert neutre["style_vector"] == [0.0, 0.0, 0.0, 0.0, 0.0]
