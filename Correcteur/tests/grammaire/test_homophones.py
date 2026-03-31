"""Tests pour la desambiguation des homophones grammaticaux."""

from lectura_correcteur.grammaire._homophones import verifier_homophones


def test_est_comme_conjonction(mock_lexique):
    """'est' etiquete CON -> devrait devenir 'et'."""
    mots = ["chat", "est", "chien"]
    pos = ["NOM", "CON", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "et"
    assert len(corrections) == 1


def test_et_devant_adjectif_avec_sujet(mock_lexique):
    """'elle et belle' -> 'elle est belle' (sujet + CON + ADJ)."""
    mots = ["elle", "et", "belle"]
    pos = ["PRO:per", "CON", "ADJ"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "est"
    assert len(corrections) == 1


def test_pas_de_faux_positif_et(mock_lexique):
    """'chat et chien' -> pas de correction (CON correct)."""
    mots = ["chat", "et", "chien"]
    pos = ["NOM", "CON", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "et"
    assert len(corrections) == 0
