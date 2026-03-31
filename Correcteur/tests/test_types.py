"""Tests pour les types publics."""

from lectura_correcteur._types import (
    Correction,
    MotAnalyse,
    ResultatCorrection,
    TypeCorrection,
)


def test_type_correction_enum():
    assert TypeCorrection.AUCUNE.value == "aucune"
    assert TypeCorrection.GRAMMAIRE.value == "grammaire"
    assert TypeCorrection.HORS_LEXIQUE.value == "hors_lexique"


def test_mot_analyse_defaults():
    m = MotAnalyse(original="chat", corrige="chat")
    assert m.pos == ""
    assert m.dans_lexique is False
    assert m.type_correction == TypeCorrection.AUCUNE


def test_resultat_correction_n_corrections():
    r = ResultatCorrection(
        phrase_originale="Les enfant",
        phrase_corrigee="Les enfants",
        mots=[
            MotAnalyse(original="Les", corrige="Les"),
            MotAnalyse(original="enfant", corrige="enfants"),
        ],
    )
    assert r.n_corrections == 1


def test_correction_dataclass():
    c = Correction(
        index=0,
        original="et",
        corrige="est",
        type_correction=TypeCorrection.GRAMMAIRE,
        explication="'et' -> 'est' (verbe attendu)",
    )
    assert c.index == 0
    assert c.explication != ""


def test_no_g2p_fields():
    """Les types ne doivent pas avoir de champs G2P/P2G."""
    m = MotAnalyse(original="test", corrige="test")
    assert not hasattr(m, "ipa")
    assert not hasattr(m, "confiance")
    assert not hasattr(m, "alternatives")
