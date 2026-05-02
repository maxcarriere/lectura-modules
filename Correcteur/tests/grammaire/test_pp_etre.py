"""Tests pour l'accord du PP avec le sujet quand l'auxiliaire est etre."""

from lectura_correcteur.grammaire._participe import verifier_pp_accord_etre


def test_elle_est_alle_allee(mock_lexique):
    """'elle est allé' -> 'elle est allée'."""
    mots = ["elle", "est", "allé"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "allée"
    assert len(corrections) == 1


def test_ils_sont_arrive_arrives(mock_lexique):
    """'ils sont arrivé' -> 'ils sont arrivés'."""
    mots = ["ils", "sont", "arrivé"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "arrivés"
    assert len(corrections) == 1


def test_elles_sont_parti_parties(mock_lexique):
    """'elles sont parti' -> 'elles sont parties'."""
    mots = ["elles", "sont", "parti"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "parties"
    assert len(corrections) == 1


def test_elle_est_tombe_tombee(mock_lexique):
    """'elle est tombé' -> 'elle est tombée'."""
    mots = ["elle", "est", "tombé"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "tombée"
    assert len(corrections) == 1


def test_nous_sommes_alle_alles(mock_lexique):
    """'nous sommes allé' -> 'nous sommes allés' (masc plur)."""
    mots = ["nous", "sommes", "allé"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "allés"
    assert len(corrections) == 1


def test_pas_de_correction_deja_correct(mock_lexique):
    """'elle est allée' -> pas de correction."""
    mots = ["elle", "est", "allée"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "allée"
    assert len(corrections) == 0


def test_pas_de_correction_avec_avoir(mock_lexique):
    """'elle a mangé' -> pas de correction (avoir, pas etre)."""
    mots = ["elle", "a", "mangé"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "mangé"
    assert len(corrections) == 0


def test_pas_de_correction_infinitif(mock_lexique):
    """'elle est aller' -> pas de correction (infinitif, pas PP)."""
    mots = ["elle", "est", "aller"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "aller"
    assert len(corrections) == 0


def test_il_est_alle_pas_de_correction(mock_lexique):
    """'il est allé' -> pas de correction (masc sing deja correct)."""
    mots = ["il", "est", "allé"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "allé"
    assert len(corrections) == 0


def test_elle_est_ne_pas_venue(mock_lexique):
    """'elle est pas venu' -> 'elle est pas venue' (saut ne/pas)."""
    mots = ["elle", "est", "pas", "venu"]
    pos = ["PRO:per", "AUX", "ADV", "VER"]
    result, corrections = verifier_pp_accord_etre(
        mots, pos, {}, mock_lexique,
    )
    assert result[3] == "venue"
    assert len(corrections) == 1
