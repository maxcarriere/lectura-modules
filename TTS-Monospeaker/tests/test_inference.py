"""Tests pour les engines d'inference (mock/unit)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestOnnxTTSEngine:
    """Tests pour OnnxTTSEngine avec sessions ONNX mockees."""

    def _make_engine(self, tmp_path):
        """Cree un engine avec un repertoire de modeles factice."""
        from lectura_tts_monospeaker.inference_onnx import OnnxTTSEngine

        # Creer les fichiers config/vocab
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
            "vocab": ["<PAD>", "<UNK>", "#", "|", "b", "ɔ̃", "ʒ", "u", "ʁ"],
            "phone2id": {"<PAD>": 0, "<UNK>": 1, "#": 2, "|": 3,
                         "b": 4, "ɔ̃": 5, "ʒ": 6, "u": 7, "ʁ": 8},
        }

        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "phoneme_vocab.json").write_text(json.dumps(vocab_data))
        # Creer des fichiers ONNX factices
        (tmp_path / "fastpitch_encoder.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "fastpitch_decoder.onnx").write_bytes(b"FAKE_ONNX")
        (tmp_path / "hifigan.onnx").write_bytes(b"FAKE_ONNX")

        return OnnxTTSEngine(tmp_path)

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
            np.random.randn(1, T, D).astype(np.float32),  # enc_out
            np.random.randn(1, T).astype(np.float32),      # dur_pred
            np.random.randn(1, T).astype(np.float32),      # pitch_pred
            np.random.randn(1, T).astype(np.float32),      # energy_pred
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

        mock_session_cls.side_effect = [mock_encoder, mock_decoder, mock_hifigan]

        result = engine.synthesize_phonemes("bɔ̃ʒuʁ")

        assert result.sample_rate == 22050
        assert isinstance(result.samples, np.ndarray)
        assert result.samples.dtype == np.float32
        assert len(result.phoneme_timings) == 5  # b ɔ̃ ʒ u ʁ


class TestApiTTSEngine:
    """Tests pour ApiTTSEngine."""

    def test_init_defaults(self):
        from lectura_tts_monospeaker.inference_api import ApiTTSEngine

        engine = ApiTTSEngine()
        assert "lec-tu-ra.com" in engine._url

    def test_init_custom(self):
        from lectura_tts_monospeaker.inference_api import ApiTTSEngine

        engine = ApiTTSEngine(api_url="http://localhost:8000", api_key="test-key")
        assert engine._url == "http://localhost:8000"
        assert engine._key == "test-key"
