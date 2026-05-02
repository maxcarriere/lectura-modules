"""Tests d'integration — necessitent les modeles ONNX et/ou lectura-g2p.

Ces tests sont marques avec pytest.mark.integration et ne sont executes
que si les modeles sont disponibles.
"""

import pytest

try:
    import onnxruntime
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from lectura_tts_monospeaker._chargeur import find_models_dir
    HAS_MODELS = find_models_dir() is not None
except Exception:
    HAS_MODELS = False

try:
    import lectura_nlp
    HAS_G2P = True
except ImportError:
    HAS_G2P = False


integration = pytest.mark.skipif(
    not (HAS_ORT and HAS_MODELS),
    reason="Necessite onnxruntime + modeles ONNX installes"
)


@integration
class TestOnnxIntegration:
    """Tests d'integration avec les vrais modeles ONNX."""

    def test_synthesize_phonemes_bonjour(self):
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")
        result = engine.synthesize_phonemes("bɔ̃ʒuʁ")

        assert result.sample_rate == 22050
        assert len(result.samples) > 0
        assert result.samples.max() <= 1.0
        assert result.samples.min() >= -1.0
        assert len(result.phoneme_timings) == 5

    def test_synthesize_phonemes_prosody(self):
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")

        result_normal = engine.synthesize_phonemes("bɔ̃ʒuʁ", duration_scale=1.0)
        result_slow = engine.synthesize_phonemes("bɔ̃ʒuʁ", duration_scale=2.0)

        # Plus lent → plus de samples
        assert len(result_slow.samples) > len(result_normal.samples)

    def test_synthesize_phonemes_phrase_types(self):
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")

        for pt in [0, 1, 2, 3]:
            result = engine.synthesize_phonemes("bɔ̃ʒuʁ", phrase_type=pt)
            assert len(result.samples) > 0

    @pytest.mark.skipif(not HAS_G2P, reason="Necessite lectura-g2p")
    def test_synthesize_text(self):
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")
        result = engine.synthesize("Bonjour le monde")

        assert result.sample_rate == 22050
        assert len(result.samples) > 0

    @pytest.mark.skipif(not HAS_G2P, reason="Necessite lectura-g2p")
    def test_synthetiser_convenience(self):
        from lectura_tts_monospeaker import synthetiser
        import numpy as np

        audio = synthetiser("Bonjour")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0


@integration
class TestLecturaTTSIntegration:
    """Test d'integration avec le registre lectura_tts."""

    def test_create_engine_lectura_mono(self):
        """Verifie que le moteur est accessible via lectura_tts."""
        try:
            from lectura_tts import create_engine, is_available

            if is_available("lectura-mono"):
                engine = create_engine("lectura-mono")
                result = engine.synthesize_phonemes("bɔ̃ʒuʁ")
                assert len(result.samples) > 0
            else:
                pytest.skip("lectura-mono non disponible dans le registre")
        except ImportError:
            pytest.skip("lectura_tts non installe")
