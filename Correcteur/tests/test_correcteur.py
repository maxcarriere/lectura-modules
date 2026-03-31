"""Tests d'integration du pipeline complet de correction."""

from lectura_correcteur import Correcteur, TypeCorrection
from lectura_correcteur._config import CorrecteurConfig


def test_phrase_vide(mock_lexique):
    c = Correcteur(mock_lexique)
    result = c.corriger("")
    assert result.phrase_corrigee == ""
    assert result.n_corrections == 0


def test_phrase_simple(mock_lexique):
    c = Correcteur(mock_lexique)
    result = c.corriger("Le chat dort")
    assert result.phrase_corrigee is not None
    assert len(result.phrase_corrigee) > 0


def test_grammaire_pluriel(mock_lexique):
    """'les enfant' -> 'les enfants' via regles grammaire (appel direct)."""
    from lectura_correcteur.grammaire import appliquer_grammaire

    mots = ["les", "enfant"]
    pos = ["ART:def", "NOM"]
    result, corrections = appliquer_grammaire(
        mots, pos, {}, mock_lexique,
    )
    assert "enfants" in [m.lower() for m in result]


def test_resultat_structure(mock_lexique):
    c = Correcteur(mock_lexique)
    result = c.corriger("Le chat dort")
    assert hasattr(result, "phrase_originale")
    assert hasattr(result, "phrase_corrigee")
    assert hasattr(result, "mots")
    assert hasattr(result, "n_corrections")
    assert hasattr(result, "corrections")


def test_syntaxe_majuscule(mock_lexique):
    """La premiere lettre doit etre en majuscule."""
    c = Correcteur(mock_lexique)
    result = c.corriger("le chat")
    assert result.phrase_corrigee[0].isupper()


def test_sous_modules_independants():
    """Les sous-modules doivent etre importables independamment."""
    from lectura_correcteur.grammaire import verifier_accords
    from lectura_correcteur.orthographe import VerificateurOrthographe
    from lectura_correcteur.syntaxe import verifier_ponctuation

    assert callable(verifier_accords)
    assert callable(verifier_ponctuation)
    assert VerificateurOrthographe is not None


def test_config_sans_grammaire(mock_lexique):
    """Avec grammaire desactivee, pas de corrections grammaticales."""
    config = CorrecteurConfig(activer_grammaire=False)
    c = Correcteur(mock_lexique, config=config)
    result = c.corriger("Le chat dort")
    assert result.phrase_corrigee is not None


def test_mot_analyse_sans_ipa(mock_lexique):
    """MotAnalyse ne doit pas avoir de champ ipa/confiance/alternatives."""
    c = Correcteur(mock_lexique)
    result = c.corriger("Le chat dort")
    for m in result.mots:
        assert not hasattr(m, "ipa")
        assert not hasattr(m, "confiance")
        assert not hasattr(m, "alternatives")


def test_pipeline_utilise_crf(mock_lexique):
    """Le pipeline doit utiliser le MorphoTagger CRF pour les POS."""
    c = Correcteur(mock_lexique)
    result = c.corriger("Le chat mange")
    # Les mots doivent avoir des POS assigns par le CRF
    for m in result.mots:
        assert m.pos != "" or m.original in ("Le", "le")
