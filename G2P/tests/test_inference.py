"""Tests de cohérence entre les 3 moteurs d'inférence.

Vérifie que ONNX, NumPy et Pure Python produisent les mêmes résultats.
Ces tests nécessitent un modèle entraîné dans modeles/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_MODELS_DIR = Path(__file__).resolve().parent.parent / "modeles"
_VOCAB_PATH = _MODELS_DIR / "unifie_vocab.json"
_WEIGHTS_PATH = _MODELS_DIR / "unifie_weights.json"
_ONNX_PATH = _MODELS_DIR / "unifie.onnx"


def _has_trained_model():
    return _VOCAB_PATH.exists() and _WEIGHTS_PATH.exists()


def _has_onnx_model():
    return _ONNX_PATH.exists()


@pytest.fixture
def numpy_engine():
    if not _has_trained_model():
        pytest.skip("Modèle non disponible")
    from lectura_nlp.inference_numpy import NumpyInferenceEngine
    return NumpyInferenceEngine(_WEIGHTS_PATH, _VOCAB_PATH)


@pytest.fixture
def pure_engine():
    if not _has_trained_model():
        pytest.skip("Modèle non disponible")
    from lectura_nlp.inference_pure import PureInferenceEngine
    return PureInferenceEngine(_WEIGHTS_PATH, _VOCAB_PATH)


@pytest.fixture
def onnx_engine():
    if not _has_onnx_model():
        pytest.skip("Modèle ONNX non disponible")
    from lectura_nlp.inference_onnx import OnnxInferenceEngine
    return OnnxInferenceEngine(_ONNX_PATH, _VOCAB_PATH)


TEST_SENTENCES = [
    ["les", "enfants", "sont", "arrivés"],
    ["bonjour"],
    ["le", "chat", "mange"],
]


class TestNumpyEngine:
    def test_empty(self, numpy_engine):
        result = numpy_engine.analyser([])
        assert result["tokens"] == []

    def test_single_word(self, numpy_engine):
        result = numpy_engine.analyser(["bonjour"])
        assert len(result["g2p"]) == 1
        assert len(result["pos"]) == 1
        assert len(result["liaison"]) == 1

    def test_sentence(self, numpy_engine):
        tokens = ["les", "enfants", "sont", "arrivés"]
        result = numpy_engine.analyser(tokens)
        assert len(result["g2p"]) == 4
        assert len(result["pos"]) == 4


class TestPureEngine:
    def test_empty(self, pure_engine):
        result = pure_engine.analyser([])
        assert result["tokens"] == []

    def test_single_word(self, pure_engine):
        result = pure_engine.analyser(["bonjour"])
        assert len(result["g2p"]) == 1
        assert len(result["pos"]) == 1

    def test_sentence(self, pure_engine):
        tokens = ["les", "enfants", "sont", "arrivés"]
        result = pure_engine.analyser(tokens)
        assert len(result["g2p"]) == 4


class TestCrossEngine:
    """Vérifie la cohérence entre NumPy et Pure Python."""

    def test_numpy_vs_pure(self, numpy_engine, pure_engine):
        for tokens in TEST_SENTENCES:
            np_result = numpy_engine.analyser(tokens)
            pure_result = pure_engine.analyser(tokens)

            assert np_result["g2p"] == pure_result["g2p"], (
                f"G2P mismatch for {tokens}: "
                f"{np_result['g2p']} != {pure_result['g2p']}"
            )
            assert np_result["pos"] == pure_result["pos"], (
                f"POS mismatch for {tokens}"
            )
            assert np_result["liaison"] == pure_result["liaison"], (
                f"Liaison mismatch for {tokens}"
            )

    def test_numpy_vs_onnx(self, numpy_engine, onnx_engine):
        for tokens in TEST_SENTENCES:
            np_result = numpy_engine.analyser(tokens)
            onnx_result = onnx_engine.analyser(tokens)

            # ONNX may have slight differences due to ONNX export
            # but argmax results should be identical
            assert np_result["g2p"] == onnx_result["g2p"], (
                f"G2P mismatch (numpy vs onnx) for {tokens}: "
                f"{np_result['g2p']} != {onnx_result['g2p']}"
            )
