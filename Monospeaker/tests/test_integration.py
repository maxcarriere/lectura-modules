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
    import lectura_phonemiseur
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
        result = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281")

        assert result.sample_rate == 22050
        assert len(result.samples) > 0
        assert result.samples.max() <= 1.0
        assert result.samples.min() >= -1.0

    def test_synthesize_phonemes_prosody(self):
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")

        result_normal = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281", duration_scale=1.0)
        result_slow = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281", duration_scale=2.0)

        # Plus lent -> plus de samples
        assert len(result_slow.samples) > len(result_normal.samples)

    def test_synthesize_phonemes_phrase_types(self):
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")

        for pt in [0, 1, 2, 3]:
            result = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281", phrase_type=pt)
            assert len(result.samples) > 0

    def test_synthesize_phonemes_style_presets(self):
        """Teste tous les presets de style."""
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")

        for style_name in ("neutre", "narratif", "dialogue", "expressif",
                           "meditatif", "rapide", "lent"):
            result = engine.synthesize_phonemes(
                "\u0254\u0303\u0292u\u0281", style=style_name)
            assert len(result.samples) > 0, f"Style {style_name} a echoue"

    def test_synthesize_phonemes_explicit_style_vector(self):
        """Teste un vecteur de style explicite."""
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")
        result = engine.synthesize_phonemes(
            "\u0254\u0303\u0292u\u0281",
            style_vector=[0.5, 0.0, 0.0, 0.0, 0.0],
        )
        assert len(result.samples) > 0

    def test_synthesize_phonemes_n_ode_steps(self):
        """Teste differents n_ode_steps."""
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")

        result_fast = engine.synthesize_phonemes(
            "\u0254\u0303\u0292u\u0281", n_ode_steps=2)
        result_quality = engine.synthesize_phonemes(
            "\u0254\u0303\u0292u\u0281", n_ode_steps=8)

        assert len(result_fast.samples) > 0
        assert len(result_quality.samples) > 0

    @pytest.mark.skipif(not HAS_G2P, reason="Necessite lectura-g2p")
    def test_synthesize_text(self):
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")
        result = engine.synthesize("Bonjour le monde")

        assert result.sample_rate == 22050
        assert len(result.samples) > 0

    @pytest.mark.skipif(not HAS_G2P, reason="Necessite lectura-g2p")
    def test_synthesize_text_with_style(self):
        from lectura_tts_monospeaker import creer_engine

        engine = creer_engine(mode="local")
        result = engine.synthesize("Bonjour le monde", style="narratif")

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

    @pytest.mark.skipif(not HAS_G2P, reason="Necessite lectura-g2p")
    def test_synthetiser_with_style(self):
        from lectura_tts_monospeaker import synthetiser
        import numpy as np

        audio = synthetiser("Bonjour", style="narratif")
        assert isinstance(audio, np.ndarray)
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
                result = engine.synthesize_phonemes("\u0254\u0303\u0292u\u0281")
                assert len(result.samples) > 0
            else:
                pytest.skip("lectura-mono non disponible dans le registre")
        except ImportError:
            pytest.skip("lectura_tts non installe")
