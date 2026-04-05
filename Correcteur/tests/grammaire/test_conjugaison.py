"""Tests pour les regles de conjugaison."""

from lectura_correcteur.grammaire._conjugaison import verifier_conjugaisons


def test_ils_mange_ent(mock_lexique):
    """'ils mange' -> 'ils mangent'."""
    mots = ["ils", "mange"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "mange"],
    )
    assert result[1] == "mangent"
    assert len(corrections) >= 1


def test_elles_mange_ent(mock_lexique):
    """'elles mange' -> 'elles mangent'."""
    mots = ["elles", "mange"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["elles", "mange"],
    )
    assert result[1] == "mangent"


def test_pas_de_correction_si_deja_correct(mock_lexique):
    """'ils mangent' -> pas de correction."""
    mots = ["ils", "mangent"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "mangent"],
    )
    assert result == ["ils", "mangent"]


def test_ils_dort_dorment(mock_lexique):
    """'ils dort' -> 'ils dorment' (3e groupe, -rt -> -rment)."""
    mots = ["ils", "dort"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "dort"],
    )
    assert result[1] == "dorment"
    assert len(corrections) >= 1


def test_ils_finit_finissent(mock_lexique):
    """'ils finit' -> 'ils finissent' (2e groupe, -it -> -issent)."""
    mots = ["ils", "finit"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "finit"],
    )
    assert result[1] == "finissent"
    assert len(corrections) >= 1


def test_ils_prend_prennent(mock_lexique):
    """'ils prend' -> 'ils prennent' (3e groupe, -nd -> -nnent)."""
    mots = ["ils", "prend"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "prend"],
    )
    assert result[1] == "prennent"
    assert len(corrections) >= 1


# --- Axe 3 : Verbes irreguliers 3pl ---

def test_ils_va_vont(mock_lexique):
    """'ils va' -> 'ils vont' (irregulier)."""
    mots = ["ils", "va"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "va"],
    )
    assert result[1] == "vont"
    assert len(corrections) >= 1


def test_ils_fait_font(mock_lexique):
    """'ils fait' -> 'ils font' (irregulier)."""
    mots = ["ils", "fait"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "fait"],
    )
    assert result[1] == "font"
    assert len(corrections) >= 1


def test_ils_dit_disent(mock_lexique):
    """'ils dit' -> 'ils disent' (irregulier)."""
    mots = ["ils", "dit"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["ils", "dit"],
    )
    assert result[1] == "disent"
    assert len(corrections) >= 1


# --- Axe conjugaison : nous/vous/je/tu ---

def test_nous_mange_mangeons(mock_lexique):
    """'nous mange' -> 'nous mangeons'."""
    mots = ["nous", "mange"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["nous", "mange"],
    )
    assert result[1] == "mangeons"
    assert len(corrections) >= 1


def test_vous_mange_mangez(mock_lexique):
    """'vous mange' -> 'vous mangez'."""
    mots = ["vous", "mange"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["vous", "mange"],
    )
    assert result[1] == "mangez"
    assert len(corrections) >= 1


def test_je_mangent_mange(mock_lexique):
    """'je mangent' -> 'je mange'."""
    mots = ["je", "mangent"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["je", "mangent"],
    )
    assert result[1] == "mange"
    assert len(corrections) >= 1


def test_tu_mangent_manges(mock_lexique):
    """'tu mangent' -> 'tu manges'."""
    mots = ["tu", "mangent"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["tu", "mangent"],
    )
    assert result[1] == "manges"
    assert len(corrections) >= 1


def test_il_mangent_mange(mock_lexique):
    """'il mangent' -> 'il mange'."""
    mots = ["il", "mangent"]
    pos = ["PRO:per", "VER"]
    result, corrections = verifier_conjugaisons(
        mots, pos, {}, mock_lexique, originaux=["il", "mangent"],
    )
    assert result[1] == "mange"
    assert len(corrections) >= 1
