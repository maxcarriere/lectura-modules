"""Tests du module lectura-ctc.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ── Tests _mel.py ─────────────────────────────────────────────────────


class TestMel:
    """Tests de l'extraction mel-spectrogram."""

    def test_shape_basique(self):
        """Verifie la shape de sortie pour 1 seconde d'audio."""
        from lectura_ctc._mel import mel_spectrogram

        audio = np.zeros(16000, dtype=np.float32)
        mel = mel_spectrogram(audio, sr=16000)
        assert mel.ndim == 4
        assert mel.shape[0] == 1  # batch
        assert mel.shape[1] == 1  # canal
        assert mel.shape[2] == 80  # n_mels
        assert mel.shape[3] > 0  # frames > 0

    def test_frames_count(self):
        """Verifie le nombre de frames (100 fps a 16kHz, hop=160)."""
        from lectura_ctc._mel import mel_spectrogram

        audio = np.random.randn(16000).astype(np.float32)
        mel = mel_spectrogram(audio, sr=16000)
        # ~100 frames/s pour 1s d'audio
        n_frames = mel.shape[3]
        assert 99 <= n_frames <= 101, f"Attendu ~100 frames, obtenu {n_frames}"

    def test_dtype_float32(self):
        """Verifie que le mel est en float32."""
        from lectura_ctc._mel import mel_spectrogram

        audio = np.random.randn(8000).astype(np.float32)
        mel = mel_spectrogram(audio, sr=16000)
        assert mel.dtype == np.float32

    def test_sr_invalide(self):
        """Verifie qu'un sample rate != 16000 leve une erreur."""
        from lectura_ctc._mel import mel_spectrogram

        audio = np.zeros(8000, dtype=np.float32)
        with pytest.raises(ValueError, match="16000"):
            mel_spectrogram(audio, sr=8000)

    def test_audio_2d_invalide(self):
        """Verifie qu'un audio 2D leve une erreur."""
        from lectura_ctc._mel import mel_spectrogram

        audio = np.zeros((2, 8000), dtype=np.float32)
        with pytest.raises(ValueError, match="1D"):
            mel_spectrogram(audio, sr=16000)

    def test_filtres_mel_shape(self):
        """Verifie la shape de la matrice de filtres mel."""
        from lectura_ctc._mel import _creer_filtres_mel

        fb = _creer_filtres_mel()
        assert fb.shape == (80, 257)  # (n_mels, n_fft//2+1)


# ── Tests _decode.py ──────────────────────────────────────────────────


class TestDecode:
    """Tests du decodage CTC greedy."""

    def test_greedy_simple(self):
        """Decodage d'une sequence simple sans repetitions."""
        from lectura_ctc._decode import ctc_greedy_decode

        # 5 frames, vocab_size=4
        # Frame 0: blank(0), Frame 1: id=1, Frame 2: blank, Frame 3: id=2, Frame 4: blank
        logits = [
            [10.0, 0.0, 0.0, 0.0],  # blank
            [0.0, 10.0, 0.0, 0.0],  # id=1
            [10.0, 0.0, 0.0, 0.0],  # blank
            [0.0, 0.0, 10.0, 0.0],  # id=2
            [10.0, 0.0, 0.0, 0.0],  # blank
        ]
        result = ctc_greedy_decode(logits, blank_id=0)
        assert result == [1, 2]

    def test_greedy_repetitions(self):
        """Les repetitions consecutives sont fusionnees."""
        from lectura_ctc._decode import ctc_greedy_decode

        logits = [
            [0.0, 10.0, 0.0],  # id=1
            [0.0, 10.0, 0.0],  # id=1 (repetition → ignore)
            [0.0, 10.0, 0.0],  # id=1 (repetition → ignore)
            [0.0, 0.0, 10.0],  # id=2
        ]
        result = ctc_greedy_decode(logits, blank_id=0)
        assert result == [1, 2]

    def test_greedy_meme_token_avec_blank(self):
        """Repetition du meme token separee par un blank → 2 occurrences."""
        from lectura_ctc._decode import ctc_greedy_decode

        logits = [
            [0.0, 10.0, 0.0],  # id=1
            [10.0, 0.0, 0.0],  # blank
            [0.0, 10.0, 0.0],  # id=1
        ]
        result = ctc_greedy_decode(logits, blank_id=0)
        assert result == [1, 1]

    def test_greedy_tout_blank(self):
        """Silence complet → sequence vide."""
        from lectura_ctc._decode import ctc_greedy_decode

        logits = [
            [10.0, 0.0],
            [10.0, 0.0],
            [10.0, 0.0],
        ]
        result = ctc_greedy_decode(logits, blank_id=0)
        assert result == []

    def test_ids_vers_phones(self):
        """Conversion IDs → chaine IPA."""
        from lectura_ctc._decode import ids_vers_phones

        vocab_inv = {0: "[PAD]", 1: "[UNK]", 2: "|", 3: "b", 4: "ɔ̃"}
        result = ids_vers_phones([3, 4, 2, 3], vocab_inv)
        assert result == "b ɔ̃ | b"

    def test_ids_vers_phones_tokens_speciaux(self):
        """Les tokens [PAD] et [UNK] sont ignores."""
        from lectura_ctc._decode import ids_vers_phones

        vocab_inv = {0: "[PAD]", 1: "[UNK]", 2: "a"}
        result = ids_vers_phones([0, 2, 1], vocab_inv)
        assert result == "a"


# ── Tests vocab ───────────────────────────────────────────────────────


class TestVocab:
    """Tests du fichier vocab_phones.json."""

    def test_vocab_existe(self):
        """Verifie que le vocab est embarque."""
        vocab_path = Path(__file__).parent.parent / "src" / "lectura_ctc" / "data" / "vocab_phones.json"
        assert vocab_path.exists(), f"vocab_phones.json introuvable : {vocab_path}"

    def test_vocab_contenu(self):
        """Verifie le contenu du vocab."""
        vocab_path = Path(__file__).parent.parent / "src" / "lectura_ctc" / "data" / "vocab_phones.json"
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        assert "[PAD]" in vocab
        assert vocab["[PAD]"] == 0
        assert "|" in vocab  # separateur de mots
        assert len(vocab) >= 50  # au moins 50 tokens


# ── Tests integration ONNX ────────────────────────────────────────────


class TestOnnxIntegration:
    """Tests d'integration avec le modele ONNX (skip si absent)."""

    @pytest.fixture
    def onnx_disponible(self):
        """Skip si onnxruntime ou le modele n'est pas disponible."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pytest.skip("onnxruntime non installe")
        modele = Path(__file__).parent.parent / "src" / "lectura_ctc" / "modeles" / "phone_ctc_int8.onnx"
        if not modele.exists():
            pytest.skip("Modele ONNX non disponible")

    def test_transcrire_silence(self, onnx_disponible):
        """Transcription d'un silence (doit retourner une chaine vide ou quasi-vide)."""
        from lectura_ctc import creer_engine

        engine = creer_engine(mode="onnx")
        audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcrire(audio)
        assert isinstance(result, str)

    def test_transcrire_bruit(self, onnx_disponible):
        """Transcription de bruit blanc (doit retourner une chaine sans crash)."""
        from lectura_ctc import creer_engine

        engine = creer_engine(mode="onnx")
        audio = np.random.randn(32000).astype(np.float32) * 0.1
        result = engine.transcrire(audio)
        assert isinstance(result, str)

    def test_transcrire_batch(self, onnx_disponible):
        """Transcription batch."""
        from lectura_ctc import creer_engine

        engine = creer_engine(mode="onnx")
        audios = [
            np.zeros(16000, dtype=np.float32),
            np.random.randn(8000).astype(np.float32) * 0.05,
        ]
        results = engine.transcrire_batch(audios)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_transcrire_sinus(self, onnx_disponible):
        """Transcription d'un sinus (doit retourner quelque chose, pas crasher)."""
        from lectura_ctc import creer_engine

        engine = creer_engine(mode="onnx")
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = engine.transcrire(audio)
        assert isinstance(result, str)


# ── Tests factory creer_engine ────────────────────────────────────────


class TestFactory:
    """Tests de la factory creer_engine."""

    def test_mode_api(self):
        """creer_engine(mode='api') retourne un ApiCTCEngine."""
        from lectura_ctc import creer_engine

        engine = creer_engine(mode="api")
        assert "ApiCTCEngine" in type(engine).__name__

    def test_mode_auto_sans_modele(self, tmp_path, monkeypatch):
        """mode='auto' sans modele nulle part → fallback API."""
        import lectura_ctc
        from lectura_ctc import creer_engine
        # Forcer la cascade a ne rien trouver
        monkeypatch.setattr(lectura_ctc, "_resoudre_modeles_dir", lambda models_dir=None: None)
        engine = creer_engine(mode="auto")
        assert "ApiCTCEngine" in type(engine).__name__

    def test_mode_onnx_sans_modele(self, tmp_path, monkeypatch):
        """mode='onnx' sans modele ni fallback leve RuntimeError."""
        import lectura_ctc
        from lectura_ctc import creer_engine
        # Forcer la cascade et le fallback a ne rien trouver
        monkeypatch.setattr(lectura_ctc, "_resoudre_modeles_dir", lambda models_dir=None: None)
        monkeypatch.setattr(lectura_ctc, "_MODELES_DIR", tmp_path)
        with pytest.raises(RuntimeError):
            creer_engine(mode="onnx")
