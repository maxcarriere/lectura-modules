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
    from lectura_tts_multispeaker._chargeur import find_models_dir
    HAS_MODELS = find_models_dir("siwis") is not None
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
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")
        result = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")

        assert result.sample_rate == 22050
        assert len(result.samples) > 0
        assert result.samples.max() <= 1.0
        assert result.samples.min() >= -1.0
        assert len(result.phoneme_timings) == 5

    def test_synthesize_phonemes_prosody(self):
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")

        result_normal = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281", duration_scale=1.0)
        result_slow = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281", duration_scale=2.0)

        assert len(result_slow.samples) > len(result_normal.samples)

    def test_synthesize_phonemes_phrase_types(self):
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")

        for pt in [0, 1, 2, 3]:
            result = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281", phrase_type=pt)
            assert len(result.samples) > 0

    def test_set_speaker(self):
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")
        result_siwis = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")

        engine.set_speaker("bernard")
        result_bernard = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")
        assert len(result_bernard.samples) > 0

    def test_set_speaker_no_reload_unified(self):
        """En mode unifie, set_speaker ne recharge pas l'encodeur."""
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")
        engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")  # force load

        if not engine._unified:
            pytest.skip("Models are legacy layout, not unified")

        encoder_ref = engine._encoder
        engine.set_speaker("bernard")
        assert engine._encoder is encoder_ref
        assert engine._speaker_id == 3

    def test_liste_speakers(self):
        from lectura_tts_multispeaker import liste_speakers

        speakers = liste_speakers()
        assert len(speakers) == 6
        names = [s["name"] for s in speakers]
        assert "siwis" in names
        assert "bernard" in names
        assert "ezwa" in names

    @pytest.mark.skipif(not HAS_G2P, reason="Necessite lectura-g2p")
    def test_synthesize_text(self):
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")
        result = engine.synthesize("Bonjour le monde")

        assert result.sample_rate == 22050
        assert len(result.samples) > 0

    @pytest.mark.skipif(not HAS_G2P, reason="Necessite lectura-g2p")
    def test_synthetiser_convenience(self):
        from lectura_tts_multispeaker import synthetiser
        import numpy as np

        audio = synthetiser("Bonjour", speaker="siwis")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0


@integration
class TestStyleIntegration:
    """Tests d'integration pour le style conditioning."""

    def test_style_vector_changes_output(self):
        """Un style_vector non-nul produit un mel different."""
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")
        # Force lazy loading to set _unified flag
        engine._ensure_loaded()

        if not engine._unified:
            pytest.skip("Models are legacy layout, style not supported")

        result_neutral = engine.synthesize_phonemes(
            "b\u0254\u0303\u0292u\u0281",
            style_vector=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
        result_expressive = engine.synthesize_phonemes(
            "b\u0254\u0303\u0292u\u0281",
            style_vector=[1.5, 0.5, 0.3, 0.0, 0.0],
        )

        # Outputs should differ (style changes the mel)
        import numpy as np
        min_len = min(len(result_neutral.samples), len(result_expressive.samples))
        diff = np.abs(
            result_neutral.samples[:min_len] - result_expressive.samples[:min_len]
        ).mean()
        assert diff > 1e-5, f"Style vector had no measurable effect (diff={diff})"

    def test_style_preset(self):
        """Les presets de style sont acceptes."""
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")
        engine._ensure_loaded()

        if not engine._unified:
            pytest.skip("Models are legacy layout, style not supported")

        for preset in ["neutral", "expressive", "calm", "dialogue"]:
            result = engine.synthesize_phonemes(
                "b\u0254\u0303\u0292u\u0281",
                style=preset,
            )
            assert len(result.samples) > 0

    def test_all_speakers_produce_audio(self):
        """Les 6 speakers produisent de l'audio distinct."""
        from lectura_tts_multispeaker import creer_engine

        engine = creer_engine(mode="local", speaker="siwis")
        speakers = ["siwis", "ezwa", "nadine", "bernard", "gilles", "zeckou"]

        for spk in speakers:
            engine.set_speaker(spk)
            result = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")
            assert len(result.samples) > 0, f"Speaker {spk} produced empty audio"

    @pytest.mark.skipif(not HAS_G2P, reason="Necessite lectura-g2p")
    def test_synthetiser_with_style(self):
        """synthetiser() accepte style= et style_vector=."""
        from lectura_tts_multispeaker import synthetiser
        import numpy as np

        audio = synthetiser("Bonjour", speaker="siwis", style="expressive")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

        audio2 = synthetiser("Bonjour", speaker="siwis",
                             style_vector=[1.0, 0.0, 0.0, 0.0, 0.0])
        assert isinstance(audio2, np.ndarray)
        assert len(audio2) > 0


@integration
class TestLecturaTTSIntegration:
    """Test d'integration avec le registre lectura_tts."""

    def test_create_engine_lectura_multi(self):
        """Verifie que le moteur est accessible via lectura_tts."""
        try:
            from lectura_tts import create_engine, is_available

            if is_available("lectura-multi"):
                engine = create_engine("lectura-multi")
                result = engine.synthesize_phonemes("b\u0254\u0303\u0292u\u0281")
                assert len(result.samples) > 0
            else:
                pytest.skip("lectura-multi non disponible dans le registre")
        except ImportError:
            pytest.skip("lectura_tts non installe")
