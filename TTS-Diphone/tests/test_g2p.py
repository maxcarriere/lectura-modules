"""Tests pour lectura_tts_diphone.g2p."""

import pytest


def test_g2p_protocol():
    """G2PBackend est un Protocol runtime-checkable."""
    from lectura_tts_diphone.g2p import G2PBackend, CallableG2P

    def dummy(text):
        return [{"phones": ["a"], "boundary": "none"}]

    backend = CallableG2P(dummy)
    assert isinstance(backend, G2PBackend)


def test_callable_g2p():
    """CallableG2P wraps un callable."""
    from lectura_tts_diphone.g2p import CallableG2P

    def my_g2p(text):
        return [{"phones": ["b", "a"], "boundary": "period"}]

    backend = CallableG2P(my_g2p)
    result = backend.phonemize("ba")
    assert len(result) == 1
    assert result[0]["phones"] == ["b", "a"]
    assert result[0]["boundary"] == "period"


def test_callable_g2p_empty():
    """CallableG2P retourne une liste vide."""
    from lectura_tts_diphone.g2p import CallableG2P

    backend = CallableG2P(lambda t: [])
    assert backend.phonemize("") == []


def test_api_g2p_init():
    """LecturaApiG2P s'initialise avec URL."""
    from lectura_tts_diphone.g2p import LecturaApiG2P

    backend = LecturaApiG2P(api_url="http://example.com:8000")
    assert backend._api_url == "http://example.com:8000"


def test_api_g2p_default_url():
    """LecturaApiG2P URL par defaut."""
    from lectura_tts_diphone.g2p import LecturaApiG2P

    backend = LecturaApiG2P()
    assert "localhost" in backend._api_url


def test_auto_detect_api_preference():
    """auto_detect_g2p avec preference='api' retourne ApiG2P."""
    from lectura_tts_diphone.g2p import auto_detect_g2p, LecturaApiG2P

    backend = auto_detect_g2p(preference="api")
    assert isinstance(backend, LecturaApiG2P)


def test_auto_detect_auto():
    """auto_detect_g2p sans preference retourne un backend valide."""
    from lectura_tts_diphone.g2p import auto_detect_g2p, G2PBackend

    backend = auto_detect_g2p()
    assert isinstance(backend, G2PBackend)


def test_local_g2p_import_error():
    """auto_detect_g2p(preference='local') leve ImportError si pas installe."""
    from lectura_tts_diphone.g2p import auto_detect_g2p
    try:
        import lecteur_syllabique  # noqa: F401
        pytest.skip("lecteur_syllabique est installe")
    except ImportError:
        with pytest.raises(ImportError):
            auto_detect_g2p(preference="local")
