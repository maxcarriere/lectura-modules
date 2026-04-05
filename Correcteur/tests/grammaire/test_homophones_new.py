"""Tests pour les nouveaux homophones : leur/leurs, ca/sa, -er/-e apres aller."""

from lectura_correcteur.grammaire._homophones import verifier_homophones


# --- leur / leurs ---

def test_leur_enfants_leurs(mock_lexique):
    """'leur enfants' -> 'leurs enfants'."""
    mots = ["leur", "enfants"]
    pos = ["DET", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "leurs"
    assert len(corrections) == 1


def test_leurs_maison_leur(mock_lexique):
    """'leurs maison' -> 'leur maison'."""
    mots = ["leurs", "maison"]
    pos = ["DET", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "leur"
    assert len(corrections) == 1


def test_leurs_enfants_pas_correction(mock_lexique):
    """'leurs enfants' -> pas de correction (correct)."""
    mots = ["leurs", "enfants"]
    pos = ["DET", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "leurs"
    assert len(corrections) == 0


def test_leur_maison_pas_correction(mock_lexique):
    """'leur maison' -> pas de correction (correct)."""
    mots = ["leur", "maison"]
    pos = ["DET", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "leur"
    assert len(corrections) == 0


# --- ça / sa ---

def test_sa_mange_ca(mock_lexique):
    """'sa mange' -> 'ça mange'."""
    mots = ["sa", "mange"]
    pos = ["ADJ:pos", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "ça"
    assert len(corrections) == 1


def test_ca_maison_sa(mock_lexique):
    """'ça maison' -> 'sa maison'."""
    mots = ["ça", "maison"]
    pos = ["PRO:dem", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "sa"
    assert len(corrections) == 1


def test_sa_maison_pas_correction(mock_lexique):
    """'sa maison' -> pas de correction (correct)."""
    mots = ["sa", "maison"]
    pos = ["ADJ:pos", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "sa"
    assert len(corrections) == 0


# --- -er/-é apres aller ---

def test_vais_mange_manger(mock_lexique):
    """'vais mangé' -> 'vais manger' (infinitif apres aller)."""
    mots = ["vais", "mangé"]
    pos = ["VER", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "manger"
    assert len(corrections) == 1


def test_va_chanté_chanter(mock_lexique):
    """'va chanté' -> 'va chanter'."""
    mots = ["va", "chanté"]
    pos = ["VER", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "chanter"
    assert len(corrections) == 1


def test_va_manger_pas_correction(mock_lexique):
    """'va manger' -> pas de correction (deja infinitif)."""
    mots = ["va", "manger"]
    pos = ["VER", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "manger"
    assert len(corrections) == 0
