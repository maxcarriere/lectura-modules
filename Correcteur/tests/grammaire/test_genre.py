"""Tests pour les regles d'accord en genre (Rule 7 dans _accord.py)."""

from lectura_correcteur.grammaire._accord import verifier_accords


# --- Féminisation -oux -> -ousse/-ouce ---

def test_homme_roux_pas_correction(mock_lexique):
    """'homme roux' -> pas de correction (deja masculin)."""
    mots = ["homme", "roux"]
    pos = ["NOM", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "roux"
    assert not any(c.index == 1 for c in corrections)


def test_note_fausse_pas_correction(mock_lexique):
    """'note fausse' -> pas de correction (deja feminin)."""
    mots = ["note", "fausse"]
    pos = ["NOM", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "fausse"
    assert not any(c.index == 1 for c in corrections)


def test_femme_doux_douce(mock_lexique):
    """'femme doux' -> 'femme douce' (feminin -oux -> -ouce)."""
    mots = ["femme", "doux"]
    pos = ["NOM", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "douce"
    assert any(c.corrige == "douce" for c in corrections)


def test_femme_roux_rousse(mock_lexique):
    """'femme roux' -> 'femme rousse' (feminin -oux -> -ousse)."""
    mots = ["femme", "roux"]
    pos = ["NOM", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "rousse"
    assert any(c.corrige == "rousse" for c in corrections)


def test_note_faux_fausse(mock_lexique):
    """'note faux' -> 'note fausse' (feminin -aux -> -ausse)."""
    mots = ["note", "faux"]
    pos = ["NOM", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "fausse"
    assert any(c.corrige == "fausse" for c in corrections)


# --- Féminisation -eux -> -euse ---

def test_femme_heureux_heureuse(mock_lexique):
    """'femme heureux' -> 'femme heureuse' (feminin -eux -> -euse)."""
    mots = ["femme", "heureux"]
    pos = ["NOM", "ADJ"]
    result, corrections = verifier_accords(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "heureuse"
    assert any(c.corrige == "heureuse" for c in corrections)
