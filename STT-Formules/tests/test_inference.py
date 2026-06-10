"""Tests du moteur d'inference FormulaCTC (Phase 3).

- _mel.py : compute_log_mel sur sinus synthetique, load_audio
- _inference.py : _ctc_greedy_decode, OnnxFormulaEngine (integration)
- __init__.py : creer_engine, transcrire (convenience)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lectura_stt_formules._mel import SAMPLE_RATE, compute_log_mel, load_audio
from lectura_stt_formules._inference import (
    OnnxFormulaEngine,
    _ctc_greedy_decode,
    _resoudre_modeles_dir,
)

# Dossier des exports ONNX (a cote du code source)
_EXPORTS_DIR = Path(__file__).resolve().parent.parent / "training" / "exports"
_HAS_MODEL = (
    (_EXPORTS_DIR / "formula_ctc_int8.onnx").is_file()
    or (_EXPORTS_DIR / "formula_ctc.onnx").is_file()
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_sine_wav(
    tmp_path: Path,
    freq: float = 440.0,
    duration: float = 0.5,
    sr: int = SAMPLE_RATE,
) -> Path:
    """Genere un WAV sinus dans tmp_path et retourne le chemin."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    path = tmp_path / "test_sine.wav"
    sf.write(str(path), data, sr)
    return path


# ──────────────────────────────────────────────
# Tests _mel.py
# ──────────────────────────────────────────────

class TestComputeLogMel:
    """Tests pour compute_log_mel."""

    def test_shape_sinus(self):
        """Un sinus 440 Hz de 0.5s → shape (1, 80, T) avec T > 0."""
        duration = 0.5
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration),
                        endpoint=False, dtype=np.float32)
        waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

        mel = compute_log_mel(waveform)

        assert mel.ndim == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80
        assert mel.shape[2] > 0
        assert mel.dtype == np.float32

    def test_expected_frames(self):
        """Verifie que le nombre de frames est coherent (~100 fps)."""
        duration = 1.0
        n_samples = int(SAMPLE_RATE * duration)
        waveform = np.zeros(n_samples, dtype=np.float32)
        mel = compute_log_mel(waveform)

        # ~100 frames/s pour hop_length=160 @ 16kHz
        expected_frames = n_samples // 160 + 1
        # Tolerance de quelques frames (padding)
        assert abs(mel.shape[2] - expected_frames) <= 3

    def test_log_values(self):
        """Les valeurs log-mel ne doivent pas etre +inf ou NaN."""
        t = np.linspace(0, 0.3, int(SAMPLE_RATE * 0.3),
                        endpoint=False, dtype=np.float32)
        waveform = 0.3 * np.sin(2 * np.pi * 1000 * t)
        mel = compute_log_mel(waveform)

        assert np.all(np.isfinite(mel))


class TestLoadAudio:
    """Tests pour load_audio."""

    def test_load_wav_16k(self, tmp_path):
        """Charge un WAV 16kHz → shape (N,), float32."""
        wav_path = _make_sine_wav(tmp_path, sr=SAMPLE_RATE)
        data = load_audio(wav_path)

        assert data.ndim == 1
        assert data.dtype == np.float32
        assert len(data) > 0

    def test_load_wav_resample(self, tmp_path):
        """Charge un WAV 44.1kHz, resample en 16kHz."""
        sr_orig = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr_orig * duration),
                        endpoint=False, dtype=np.float32)
        data = 0.5 * np.sin(2 * np.pi * 440 * t)
        wav_path = tmp_path / "test_44k.wav"
        sf.write(str(wav_path), data, sr_orig)

        resampled = load_audio(wav_path, sample_rate=SAMPLE_RATE)

        expected_samples = int(duration * SAMPLE_RATE)
        assert abs(len(resampled) - expected_samples) <= 1
        assert resampled.dtype == np.float32

    def test_load_stereo(self, tmp_path):
        """Charge un WAV stereo → mono."""
        duration = 0.3
        n = int(SAMPLE_RATE * duration)
        stereo = np.column_stack([
            np.sin(np.linspace(0, 2 * np.pi * 440, n, dtype=np.float32)),
            np.sin(np.linspace(0, 2 * np.pi * 880, n, dtype=np.float32)),
        ])
        wav_path = tmp_path / "stereo.wav"
        sf.write(str(wav_path), stereo, SAMPLE_RATE)

        mono = load_audio(wav_path)
        assert mono.ndim == 1
        assert len(mono) == n


# ──────────────────────────────────────────────
# Tests _inference.py
# ──────────────────────────────────────────────

class TestCtcGreedyDecode:
    """Tests pour _ctc_greedy_decode."""

    def test_simple_sequence(self):
        """Logits synthetiques → decodage correct."""
        # 5 frames, 4 tokens (0=blank, 1, 2, 3)
        # Sequence voulue : [1, 2, 3]
        logits = np.zeros((5, 4), dtype=np.float32)
        logits[0, 1] = 10.0  # frame 0 -> token 1
        logits[1, 1] = 10.0  # frame 1 -> token 1 (repetition)
        logits[2, 2] = 10.0  # frame 2 -> token 2
        logits[3, 0] = 10.0  # frame 3 -> blank
        logits[4, 3] = 10.0  # frame 4 -> token 3

        decoded = _ctc_greedy_decode(logits, blank_id=0)
        assert decoded == [1, 2, 3]

    def test_all_blanks(self):
        """Si tout est blank, le decodage est vide."""
        logits = np.zeros((10, 5), dtype=np.float32)
        logits[:, 0] = 10.0  # Tout = blank
        decoded = _ctc_greedy_decode(logits, blank_id=0)
        assert decoded == []

    def test_repeated_tokens(self):
        """Les tokens repetes consecutivement sont collapses."""
        logits = np.zeros((6, 3), dtype=np.float32)
        logits[0, 1] = 10.0  # token 1
        logits[1, 1] = 10.0  # token 1 (collapse)
        logits[2, 0] = 10.0  # blank
        logits[3, 1] = 10.0  # token 1 (nouveau, apres blank)
        logits[4, 2] = 10.0  # token 2
        logits[5, 2] = 10.0  # token 2 (collapse)

        decoded = _ctc_greedy_decode(logits, blank_id=0)
        assert decoded == [1, 1, 2]

    def test_realistic_vocab(self):
        """Test avec des token IDs du vocabulaire reel (VINGT=20, DEUX=4)."""
        logits = np.full((8, 87), -10.0, dtype=np.float32)
        # "vingt-deux" = VINGT(20) DEUX(4)
        logits[0, 20] = 5.0  # VINGT
        logits[1, 20] = 5.0  # VINGT (repetition)
        logits[2, 20] = 5.0  # VINGT (repetition)
        logits[3, 0] = 5.0   # blank
        logits[4, 0] = 5.0   # blank
        logits[5, 4] = 5.0   # DEUX
        logits[6, 4] = 5.0   # DEUX (repetition)
        logits[7, 0] = 5.0   # blank

        decoded = _ctc_greedy_decode(logits, blank_id=0)
        assert decoded == [20, 4]  # VINGT, DEUX


class TestResoudreModelesDir:
    """Tests pour _resoudre_modeles_dir."""

    def test_explicit_dir(self, tmp_path):
        """Un dossier explicite avec un modele ONNX est detecte."""
        onnx_file = tmp_path / "formula_ctc_int8.onnx"
        onnx_file.write_bytes(b"fake onnx model")

        result = _resoudre_modeles_dir(tmp_path)
        assert result == tmp_path

    def test_no_model(self, tmp_path):
        """Un dossier sans modele retourne None."""
        result = _resoudre_modeles_dir(tmp_path)
        assert result is None

    def test_fp32_fallback(self, tmp_path):
        """Detecte aussi le modele FP32 si INT8 absent."""
        onnx_file = tmp_path / "formula_ctc.onnx"
        onnx_file.write_bytes(b"fake onnx model")

        result = _resoudre_modeles_dir(tmp_path)
        assert result == tmp_path


# ──────────────────────────────────────────────
# Tests d'integration (skip si modele absent)
# ──────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_MODEL, reason="Modele ONNX absent")
class TestOnnxFormulaEngine:
    """Tests d'integration avec le modele ONNX reel."""

    def test_load_model(self):
        """Le moteur se charge sans erreur."""
        engine = OnnxFormulaEngine(models_dir=_EXPORTS_DIR)
        assert engine.model_path.exists()

    def test_transcrire_wav(self, tmp_path):
        """Transcrit un WAV synthetique (sinus) → retourne un dict valide."""
        wav_path = _make_sine_wav(tmp_path, freq=440, duration=1.0)
        engine = OnnxFormulaEngine(models_dir=_EXPORTS_DIR)
        result = engine.transcrire(wav_path)

        assert "tokens" in result
        assert "names" in result
        assert "logits" in result
        assert isinstance(result["tokens"], list)
        assert isinstance(result["names"], list)
        assert len(result["tokens"]) == len(result["names"])
        assert result["logits"].ndim == 2
        assert result["logits"].shape[1] == 87

    def test_transcrire_mel(self):
        """Transcrit un mel pre-calcule → retourne un dict valide."""
        engine = OnnxFormulaEngine(models_dir=_EXPORTS_DIR)

        # Mel synthetique : (1, 80, 100) = ~1 seconde
        mel = np.random.randn(1, 80, 100).astype(np.float32)
        result = engine.transcrire_mel(mel)

        assert "tokens" in result
        assert "names" in result
        assert "logits" in result

    def test_model_prefers_int8(self):
        """Si les deux modeles existent, INT8 est prefere."""
        engine = OnnxFormulaEngine(models_dir=_EXPORTS_DIR)
        if (_EXPORTS_DIR / "formula_ctc_int8.onnx").is_file():
            assert engine.model_path.name == "formula_ctc_int8.onnx"


@pytest.mark.skipif(not _HAS_MODEL, reason="Modele ONNX absent")
class TestPublicAPI:
    """Tests de l'API publique creer_engine / transcrire."""

    def test_creer_engine(self):
        """creer_engine retourne un OnnxFormulaEngine."""
        from lectura_stt_formules import creer_engine

        engine = creer_engine(models_dir=_EXPORTS_DIR)
        assert isinstance(engine, OnnxFormulaEngine)

    def test_creer_engine_invalid_mode(self):
        """creer_engine avec un mode invalide leve ValueError."""
        from lectura_stt_formules import creer_engine

        with pytest.raises(ValueError, match="mode"):
            creer_engine(mode="api", models_dir=_EXPORTS_DIR)

    def test_transcrire_convenience(self, tmp_path):
        """transcrire() cree un engine et transcrit."""
        from lectura_stt_formules import transcrire

        wav_path = _make_sine_wav(tmp_path, freq=440, duration=0.5)
        result = transcrire(wav_path, models_dir=_EXPORTS_DIR)

        assert "tokens" in result
        assert "names" in result
