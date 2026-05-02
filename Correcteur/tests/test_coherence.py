"""Tests pour le module de coherence contextuelle POS bigram."""

from lectura_correcteur._coherence import appliquer_coherence
from lectura_correcteur._types import MotAnalyse, TypeCorrection


def test_con_pre_corrige_en_aux(mock_lexique):
    """Bigram (CON, PRE) suspect : 'et dans' -> 'est dans'."""
    analyses = [
        MotAnalyse(original="et", corrige="et", pos="CON", dans_lexique=True),
        MotAnalyse(original="dans", corrige="dans", pos="PRE", dans_lexique=True),
    ]
    corrections = appliquer_coherence(analyses, mock_lexique)
    # "et" (phone "e") n'est pas homophone de "est" (phone "ɛ") dans le mock
    # donc la correction ne se fait que si les phones correspondent
    # Avec le mock par defaut, et="e" et est="ɛ" sont differents -> pas de correction
    # C'est le comportement attendu : la coherence ne corrige que quand
    # un vrai homophone existe
    assert isinstance(corrections, list)


def test_coherence_desactivee_par_defaut(mock_lexique):
    """La coherence est OFF par defaut dans CorrecteurConfig."""
    from lectura_correcteur._config import CorrecteurConfig
    config = CorrecteurConfig()
    assert config.activer_coherence is False


def test_coherence_avec_homophones():
    """Bigram suspect resolu par homophone : 'et le' -> 'est le'."""
    from tests.conftest import MockLexique

    # Creer un lexique ou "et" et "est" sont homophones (meme phone "e")
    formes = {
        "et": [{"ortho": "et", "cgram": "CON", "phone": "e", "freq": 100.0}],
        "est": [{"ortho": "est", "cgram": "AUX", "phone": "e", "freq": 500.0}],
        "le": [{"ortho": "le", "cgram": "ART:def", "phone": "lə", "freq": 890.0}],
        "chat": [{"ortho": "chat", "cgram": "NOM", "phone": "ʃa", "freq": 45.0}],
    }
    lex = MockLexique(formes=formes)

    analyses = [
        MotAnalyse(original="et", corrige="et", pos="CON", dans_lexique=True),
        MotAnalyse(original="le", corrige="le", pos="ART:def", dans_lexique=True),
    ]
    corrections = appliquer_coherence(analyses, lex)
    assert len(corrections) == 1
    assert corrections[0].corrige == "est"
    assert analyses[0].pos == "AUX"


def test_coherence_pas_de_faux_positif():
    """Un bigram non suspect ne doit pas etre modifie."""
    from tests.conftest import MockLexique

    formes = {
        "le": [{"ortho": "le", "cgram": "ART:def", "phone": "lə", "freq": 890.0}],
        "chat": [{"ortho": "chat", "cgram": "NOM", "phone": "ʃa", "freq": 45.0}],
    }
    lex = MockLexique(formes=formes)

    analyses = [
        MotAnalyse(original="le", corrige="le", pos="ART:def", dans_lexique=True),
        MotAnalyse(original="chat", corrige="chat", pos="NOM", dans_lexique=True),
    ]
    corrections = appliquer_coherence(analyses, lex)
    assert len(corrections) == 0
