"""Tests pour l'accord de l'attribut en genre apres copule."""

from lectura_correcteur.grammaire._accord import verifier_accords


def test_elle_est_petit_petite(mock_lexique):
    """'elle est petit' -> 'elle est petite'."""
    mots = ["elle", "est", "petit"]
    pos = ["PRO:per", "AUX", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "petite"
    assert any("genre" in c.explication for c in corrections)


def test_elle_est_grand_grande(mock_lexique):
    """'elle est grand' -> 'elle est grande'."""
    mots = ["elle", "est", "grand"]
    pos = ["PRO:per", "AUX", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "grande"


def test_il_est_belle_beau(mock_lexique):
    """'il est belle' -> 'il est beau'."""
    mots = ["il", "est", "belle"]
    pos = ["PRO:per", "AUX", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    # belle -> bel (via generer_candidats_masculin lle->l)
    # MockLexique has "beau" as m, not "bel"
    # The function should find a valid masc form
    assert any(e.get("genre") == "m" for e in mock_lexique.info(result[2]))


def test_elle_semble_content_contente(mock_lexique):
    """'elle semble content' -> 'elle semble contente' (copule semble)."""
    mots = ["elle", "semble", "content"]
    pos = ["PRO:per", "VER", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "contente"


def test_elle_devient_grand_grande(mock_lexique):
    """'elle devient grand' -> 'elle devient grande' (copule devient)."""
    mots = ["elle", "devient", "grand"]
    pos = ["PRO:per", "VER", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "grande"


def test_elle_reste_petit_petite(mock_lexique):
    """'elle reste petit' -> 'elle reste petite' (copule reste)."""
    mots = ["elle", "reste", "petit"]
    pos = ["PRO:per", "VER", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "petite"


def test_pas_de_correction_si_correct(mock_lexique):
    """'elle est petite' -> pas de correction."""
    mots = ["elle", "est", "petite"]
    pos = ["PRO:per", "AUX", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "petite"
    assert not any("genre" in c.explication for c in corrections)


def test_il_est_petit_pas_de_correction(mock_lexique):
    """'il est petit' -> pas de correction (masc correct)."""
    mots = ["il", "est", "petit"]
    pos = ["PRO:per", "AUX", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "petit"


def test_elles_sont_petit_petites(mock_lexique):
    """'elles sont petit' -> 'elles sont petites' (genre + nombre)."""
    mots = ["elles", "sont", "petit"]
    pos = ["PRO:per", "AUX", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    # Regle 9 (genre) : petit -> petite, puis Regle 5 (nombre) : petite -> petites
    assert result[2] == "petites"


def test_nom_fem_sujet_via_lexique(mock_lexique):
    """'la maison est grand' -> 'la maison est grande' (sujet NOM fem)."""
    mots = ["la", "maison", "est", "grand"]
    pos = ["ART:def", "NOM", "AUX", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[3] == "grande"
