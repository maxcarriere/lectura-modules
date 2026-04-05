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
    """'a chanter' -> 'a chanté'."""
    mots = ["a", "chanter"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "chanté"
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
    """'a mangé' -> pas de correction (deja participe)."""
    mots = ["a", "mangé"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "mangé"
    assert len(corrections) == 0


# --- Participes passes irreguliers ---

def test_a_faire_fait(mock_lexique):
    """'a faire' -> 'a fait' (participe passe irregulier)."""
    mots = ["a", "faire"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["a", "faire"],
    )
    assert result[1] == "fait"
    assert any(c.corrige == "fait" for c in corrections)


def test_a_prendre_pris(mock_lexique):
    """'a prendre' -> 'a pris' (participe passe irregulier)."""
    mots = ["a", "prendre"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["a", "prendre"],
    )
    assert result[1] == "pris"
    assert any(c.corrige == "pris" for c in corrections)


def test_a_voir_vu(mock_lexique):
    """'a voir' -> 'a vu' (participe passe irregulier)."""
    mots = ["a", "voir"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["a", "voir"],
    )
    assert result[1] == "vu"
    assert any(c.corrige == "vu" for c in corrections)


def test_a_mettre_mis(mock_lexique):
    """'a mettre' -> 'a mis' (participe passe irregulier)."""
    mots = ["a", "mettre"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["a", "mettre"],
    )
    assert result[1] == "mis"
    assert any(c.corrige == "mis" for c in corrections)


def test_a_ecrire_ecrit(mock_lexique):
    """'a ecrire' -> 'a ecrit' (participe passe irregulier)."""
    mots = ["a", "écrire"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["a", "écrire"],
    )
    assert result[1] == "écrit"
    assert any(c.corrige == "écrit" for c in corrections)


def test_a_manger_mange_regulier(mock_lexique):
    """'a manger' -> 'a mange' (participe passe regulier via PP)."""
    mots = ["a", "manger"]
    pos = ["AUX", "VER"]
    result, corrections = verifier_participes_passes(
        mots, pos, {}, mock_lexique, originaux=["a", "manger"],
    )
    assert result[1] == "mangé"
    assert any(c.corrige == "mangé" for c in corrections)
