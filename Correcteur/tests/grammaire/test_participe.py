"""Tests pour les regles de participe passe."""

from lectura_correcteur.grammaire._participe import verifier_participes_passes


def test_ai_manger_mange(mock_lexique):
    """'ai manger' -> 'ai mangé'."""
    mots = ["ai", "manger"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "mangé"
    assert len(corrections) == 1


def test_a_chanter_chante(mock_lexique):
    """'il a chanter' -> 'il a chanté'."""
    mots = ["il", "a", "chanter"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "chanté"
    assert len(corrections) == 1


def test_ont_jouer_joue(mock_lexique):
    """'ont jouer' -> 'ont joué'."""
    mots = ["ont", "jouer"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "joué"
    assert len(corrections) == 1


def test_avons_parler_parle(mock_lexique):
    """'avons parler' -> 'avons parlé'."""
    mots = ["avons", "parler"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "parlé"
    assert len(corrections) == 1


def test_avez_donner_donne(mock_lexique):
    """'avez donner' -> 'avez donné'."""
    mots = ["avez", "donner"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "donné"
    assert len(corrections) == 1


def test_pas_correction_sans_auxiliaire(mock_lexique):
    """'va manger' -> pas de correction (pas un auxiliaire)."""
    mots = ["va", "manger"]
    pos = ["VER", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "manger"
    assert len(corrections) == 0


def test_pas_correction_si_deja_participe(mock_lexique):
    """'il a mangé' -> pas de correction (deja participe)."""
    mots = ["il", "a", "mangé"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "mangé"
    assert len(corrections) == 0


# --- Participes passes irreguliers ---

def test_a_faire_fait(mock_lexique):
    """'il a faire' -> 'il a fait' (participe passe irregulier)."""
    mots = ["il", "a", "faire"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["il", "a", "faire"],
    )
    assert result[2] == "fait"
    assert any(c.corrige == "fait" for c in corrections)


def test_a_prendre_pris(mock_lexique):
    """'il a prendre' -> 'il a pris' (participe passe irregulier)."""
    mots = ["il", "a", "prendre"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["il", "a", "prendre"],
    )
    assert result[2] == "pris"
    assert any(c.corrige == "pris" for c in corrections)


def test_a_voir_vu(mock_lexique):
    """'il a voir' -> 'il a vu' (participe passe irregulier)."""
    mots = ["il", "a", "voir"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["il", "a", "voir"],
    )
    assert result[2] == "vu"
    assert any(c.corrige == "vu" for c in corrections)


def test_a_mettre_mis(mock_lexique):
    """'il a mettre' -> 'il a mis' (participe passe irregulier)."""
    mots = ["il", "a", "mettre"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["il", "a", "mettre"],
    )
    assert result[2] == "mis"
    assert any(c.corrige == "mis" for c in corrections)


def test_a_ecrire_ecrit(mock_lexique):
    """'il a ecrire' -> 'il a ecrit' (participe passe irregulier)."""
    mots = ["il", "a", "écrire"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["il", "a", "écrire"],
    )
    assert result[2] == "écrit"
    assert any(c.corrige == "écrit" for c in corrections)


def test_a_manger_mange_regulier(mock_lexique):
    """'il a manger' -> 'il a mange' (participe passe regulier via PP)."""
    mots = ["il", "a", "manger"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["il", "a", "manger"],
    )
    assert result[2] == "mangé"
    assert any(c.corrige == "mangé" for c in corrections)
