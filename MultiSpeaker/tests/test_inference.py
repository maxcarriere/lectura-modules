"""Tests pour les engines d'inference (mock/unit)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestOnnxTTSEngineLegacy:
    """Tests pour OnnxTTSEngine en mode legacy (per-speaker encoders)."""

    def _make_engine(self, tmp_path, speaker="siwis"):
        """Cree un engine avec un repertoire de modeles factice (legacy)."""
        from lectura_tts_multispeaker.inference_onnx import OnnxTTSEngine

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
        (tmp_path / f"encoder_{speaker}.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "decoder.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "hifigan.onnx").write_bytes(b"FAKE_ONNX")

        return OnnxTTSEngine(tmp_path, speaker=speaker)

    @patch("onnxruntime.InferenceSession")
    def test_synthesize_phonemes_shape(self, mock_session_cls, tmp_path):
        """Verifie que synthesize_phonemes retourne un TTSResult."""
        engine = self._make_engine(tmp_path)

        T = 7  # SIL + 5 phones + SIL
        D = 128
        T_mel = 50

        # Mock encoder
        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]

        # Mock decoder
        mock_decoder = MagicMock()
        mock_decoder.run.return_value = [
            np.random.randn(1, 80, T_mel).astype(np.float32) - 5.0,
        ]

        # Mock hifigan
        mock_hifigan = MagicMock()
        mock_hifigan.run.return_value = [
            np.random.randn(1, 1, T_mel * 256).astype(np.float32) * 0.5,
        ]

        # Load order: hifigan, decoder, encoder
        mock_session_cls.side_effect = [mock_hifigan, mock_decoder, mock_encoder]

        result = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")

        assert result.sample_rate == 22050
        assert isinstance(result.samples, np.ndarray)
        assert result.samples.dtype == np.float32
        assert len(result.phoneme_timings) == 5  # b o~ Z u R

    @patch("onnxruntime.InferenceSession")
    def test_set_speaker(self, mock_session_cls, tmp_path):
        """Verifie que set_speaker change le speaker."""
        engine = self._make_engine(tmp_path, speaker="siwis")
        # Add another speaker encoder
        (tmp_path / "encoder_bernard.onnx").write_bytes(b"FAKE_ONNX")

        assert engine.speaker == "siwis"
        engine.set_speaker("bernard")
        assert engine.speaker == "bernard"


class TestOnnxTTSEngineUnified:
    """Tests pour OnnxTTSEngine en mode unifie (encoder.onnx unique)."""

    def _make_engine(self, tmp_path, speaker="siwis"):
        """Cree un engine avec un repertoire de modeles factice (unified)."""
        from lectura_tts_multispeaker.inference_onnx import OnnxTTSEngine

        config = {
            "model": {"d_model": 128, "n_mels": 80, "n_style_dims": 5, "n_speakers": 6},
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
        speakers_data = {
            "speakers": [
                {"id": 0, "name": "siwis", "gender": "female", "label": "Siwis"},
                {"id": 3, "name": "bernard", "gender": "male", "label": "Bernard"},
            ],
            "default": "siwis",
        }

        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "phoneme_vocab.json").write_text(json.dumps(vocab_data))
        (tmp_path / "speakers.json").write_text(json.dumps(speakers_data))
        # Unified: single encoder.onnx
        (tmp_path / "encoder.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "decoder.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "hifigan.onnx").write_bytes(b"FAKE_ONNX")

        return OnnxTTSEngine(tmp_path, speaker=speaker)

    @patch("onnxruntime.InferenceSession")
    def test_unified_detected(self, mock_session_cls, tmp_path):
        """Verifie que le mode unifie est detecte."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 128
        T_mel = 50

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        mock_decoder = MagicMock()
        mock_decoder.run.return_value = [
            np.random.randn(1, 80, T_mel).astype(np.float32) - 5.0,
        ]
        mock_hifigan = MagicMock()
        mock_hifigan.run.return_value = [
            np.random.randn(1, 1, T_mel * 256).astype(np.float32) * 0.5,
        ]

        # Load order: hifigan, decoder, encoder (FastPitch unified)
        mock_session_cls.side_effect = [mock_hifigan, mock_decoder, mock_encoder]

        result = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")
        assert engine._unified is True
        assert result.sample_rate == 22050

        # Verify encoder was called with 4 inputs (phone_ids, phrase_type, speaker_id, style_vector)
        call_args = mock_encoder.run.call_args
        inputs = call_args[1] if call_args[1] else call_args[0][1]
        assert "speaker_id" in inputs
        assert "style_vector" in inputs

    @patch("onnxruntime.InferenceSession")
    def test_set_speaker_no_reload(self, mock_session_cls, tmp_path):
        """En mode unifie, set_speaker ne recharge pas l'encodeur."""
        engine = self._make_engine(tmp_path)

        mock_session = MagicMock()
        # Load order: hifigan, decoder, encoder
        mock_session_cls.side_effect = [MagicMock(), MagicMock(), mock_session]

        # Force load
        engine._ensure_loaded()
        encoder_after_load = engine._encoder

        # Change speaker — should NOT reload encoder
        engine.set_speaker("bernard")
        assert engine.speaker == "bernard"
        assert engine._speaker_id == 3
        assert engine._encoder is encoder_after_load  # same object

    @patch("onnxruntime.InferenceSession")
    def test_style_vector_passed(self, mock_session_cls, tmp_path):
        """Verifie que style_vector est passe a l'encodeur."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 128
        T_mel = 50

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        mock_decoder = MagicMock()
        mock_decoder.run.return_value = [
            np.random.randn(1, 80, T_mel).astype(np.float32) - 5.0,
        ]
        mock_hifigan = MagicMock()
        mock_hifigan.run.return_value = [
            np.random.randn(1, 1, T_mel * 256).astype(np.float32) * 0.5,
        ]

        mock_session_cls.side_effect = [mock_hifigan, mock_decoder, mock_encoder]

        engine.synthesize_phonemes(
            "b\u0254\u0303\u0292u\u0281",
            style_vector=[1.5, 0.5, 0.3, 0.0, 0.0],
        )

        call_args = mock_encoder.run.call_args
        inputs = call_args[1] if call_args[1] else call_args[0][1]
        np.testing.assert_array_almost_equal(
            inputs["style_vector"], [[1.5, 0.5, 0.3, 0.0, 0.0]],
        )

    @patch("onnxruntime.InferenceSession")
    def test_style_preset_resolved(self, mock_session_cls, tmp_path):
        """Verifie que style='expressif' est resolu en vecteur."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 128
        T_mel = 50

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        mock_decoder = MagicMock()
        mock_decoder.run.return_value = [
            np.random.randn(1, 80, T_mel).astype(np.float32) - 5.0,
        ]
        mock_hifigan = MagicMock()
        mock_hifigan.run.return_value = [
            np.random.randn(1, 1, T_mel * 256).astype(np.float32) * 0.5,
        ]

        mock_session_cls.side_effect = [mock_hifigan, mock_decoder, mock_encoder]

        engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281", style="expressif")

        call_args = mock_encoder.run.call_args
        inputs = call_args[1] if call_args[1] else call_args[0][1]
        np.testing.assert_array_almost_equal(
            inputs["style_vector"], [[1.0, 1.0, 0.0, 0.0, 0.0]],
        )


class TestApiTTSEngine:
    """Tests pour ApiTTSEngine multi-speaker."""

    def test_init_defaults(self):
        from lectura_tts_multispeaker.inference_api import ApiTTSEngine

        engine = ApiTTSEngine()
        assert "lectura.world" in engine._url
        assert engine.speaker == "siwis"

    def test_init_custom(self):
        from lectura_tts_multispeaker.inference_api import ApiTTSEngine

        engine = ApiTTSEngine(
            api_url="http://localhost:8000",
            api_key="test-key",
            speaker="bernard",
        )
        assert engine._url == "http://localhost:8000"
        assert engine._key == "test-key"
        assert engine.speaker == "bernard"

    def test_set_speaker(self):
        from lectura_tts_multispeaker.inference_api import ApiTTSEngine

        engine = ApiTTSEngine(speaker="siwis")
        engine.set_speaker("ezwa")
        assert engine.speaker == "ezwa"


class TestOnnxTTSEngineMatcha:
    """Tests pour OnnxTTSEngine en mode Matcha-Conformer (per-speaker encoders)."""

    def _make_engine(self, tmp_path, speaker="siwis"):
        """Cree un engine avec un repertoire de modeles factice (Matcha)."""
        from lectura_tts_multispeaker.inference_onnx import OnnxTTSEngine

        config = {
            "model": {
                "type": "matcha-conformer",
                "d_model": 384,
                "n_mels": 80,
                "n_style_dims": 5,
                "n_speakers": 6,
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
                "pitch_emb_weight": [0.01] * 384,
                "pitch_emb_bias": [0.0] * 384,
                "energy_emb_weight": [0.01] * 384,
                "energy_emb_bias": [0.0] * 384,
            },
        }
        vocab_data = {
            "vocab": ["<PAD>", "<UNK>", "#", "|", "b", "\u0254\u0303", "\u0292", "u", "\u0281"],
            "phone2id": {"<PAD>": 0, "<UNK>": 1, "#": 2, "|": 3,
                         "b": 4, "\u0254\u0303": 5, "\u0292": 6, "u": 7, "\u0281": 8},
        }
        speakers_data = {
            "speakers": [
                {"id": 0, "name": "siwis"},
                {"id": 1, "name": "ezwa"},
                {"id": 3, "name": "bernard"},
            ],
            "default": "siwis",
        }

        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "phoneme_vocab.json").write_text(json.dumps(vocab_data))
        (tmp_path / "speakers.json").write_text(json.dumps(speakers_data))
        (tmp_path / f"matcha_encoder_{speaker}.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "matcha_unet.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "hifigan.onnx").write_bytes(b"FAKE_ONNX")

        return OnnxTTSEngine(tmp_path, speaker=speaker)

    @patch("onnxruntime.InferenceSession")
    def test_matcha_detected(self, mock_session_cls, tmp_path):
        """Verifie que le type matcha-conformer est detecte."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 384

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        # UNet mock returns shape matching input x_t
        mock_unet = MagicMock()
        mock_unet.run.side_effect = lambda _, inputs: [
            np.random.randn(*inputs["x_t"].shape).astype(np.float32) * 0.01
        ]
        mock_hifigan = MagicMock()
        mock_hifigan.run.side_effect = lambda _, inputs: [
            np.random.randn(1, 1, inputs["mel"].shape[2] * 256).astype(np.float32) * 0.5
        ]

        # Load order: hifigan, unet, encoder
        mock_session_cls.side_effect = [mock_hifigan, mock_unet, mock_encoder]

        result = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")

        assert engine._model_type == "matcha-conformer"
        assert engine._unet is not None
        assert result.sample_rate == 22050
        assert isinstance(result.samples, np.ndarray)

        # UNet should be called n_ode_steps (4) times for ODE loop
        assert mock_unet.run.call_count == 4

    @patch("onnxruntime.InferenceSession")
    def test_n_ode_steps(self, mock_session_cls, tmp_path):
        """Verifie que n_ode_steps controle le nombre d'appels UNet."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 384

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        mock_unet = MagicMock()
        mock_unet.run.side_effect = lambda _, inputs: [
            np.random.randn(*inputs["x_t"].shape).astype(np.float32) * 0.01
        ]
        mock_hifigan = MagicMock()
        mock_hifigan.run.side_effect = lambda _, inputs: [
            np.random.randn(1, 1, inputs["mel"].shape[2] * 256).astype(np.float32) * 0.5
        ]

        mock_session_cls.side_effect = [mock_hifigan, mock_unet, mock_encoder]

        engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281", n_ode_steps=8)

        # UNet should be called 8 times (not default 4)
        assert mock_unet.run.call_count == 8

    @patch("onnxruntime.InferenceSession")
    def test_duration_noise(self, mock_session_cls, tmp_path):
        """Verifie que duration_noise modifie les durees."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 384

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.ones((1, T, D), dtype=np.float32) * 0.1,
            np.ones((1, T), dtype=np.float32) * 2.0,
            np.ones((1, T), dtype=np.float32) * 0.5,
            np.ones((1, T), dtype=np.float32) * 0.5,
        ]
        mock_unet = MagicMock()
        mock_unet.run.side_effect = lambda _, inputs: [
            np.random.randn(*inputs["x_t"].shape).astype(np.float32) * 0.01
        ]
        mock_hifigan = MagicMock()
        mock_hifigan.run.side_effect = lambda _, inputs: [
            np.random.randn(1, 1, inputs["mel"].shape[2] * 256).astype(np.float32) * 0.5
        ]

        mock_session_cls.side_effect = [mock_hifigan, mock_unet, mock_encoder]

        result = engine.synthesize_phonemes(
            "b\u0254\u0303\u0292u\u0281",
            duration_noise=0.15,
        )
        assert isinstance(result.samples, np.ndarray)

    @patch("onnxruntime.InferenceSession")
    def test_style_vector_passed_matcha(self, mock_session_cls, tmp_path):
        """Verifie que style_vector est passe a l'encodeur Matcha."""
        engine = self._make_engine(tmp_path)

        T = 7
        D = 384

        mock_encoder = MagicMock()
        mock_encoder.run.return_value = [
            np.random.randn(1, T, D).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
            np.random.randn(1, T).astype(np.float32),
        ]
        mock_unet = MagicMock()
        mock_unet.run.side_effect = lambda _, inputs: [
            np.random.randn(*inputs["x_t"].shape).astype(np.float32) * 0.01
        ]
        mock_hifigan = MagicMock()
        mock_hifigan.run.side_effect = lambda _, inputs: [
            np.random.randn(1, 1, inputs["mel"].shape[2] * 256).astype(np.float32) * 0.5
        ]

        mock_session_cls.side_effect = [mock_hifigan, mock_unet, mock_encoder]

        engine.synthesize_phonemes(
            "b\u0254\u0303\u0292u\u0281",
            style_vector=[1.0, 0.5, 0.0, 0.0, 0.0],
        )

        # Per-speaker Matcha encoder gets style_vector as input
        call_args = mock_encoder.run.call_args
        inputs = call_args[1] if call_args[1] else call_args[0][1]
        assert "style_vector" in inputs
        np.testing.assert_array_almost_equal(
            inputs["style_vector"], [[1.0, 0.5, 0.0, 0.0, 0.0]],
        )

    @patch("onnxruntime.InferenceSession")
    def test_set_speaker_matcha(self, mock_session_cls, tmp_path):
        """Verifie que set_speaker recharge l'encodeur en mode Matcha per-speaker."""
        engine = self._make_engine(tmp_path, speaker="siwis")
        # Add another speaker encoder
        (tmp_path / "matcha_encoder_bernard.onnx").write_bytes(b"FAKE_ONNX")

        assert engine.speaker == "siwis"
        engine.set_speaker("bernard")
        assert engine.speaker == "bernard"
