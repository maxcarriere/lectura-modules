"""Tests pour les regles de negation."""

from lectura_correcteur.grammaire._negation import verifier_negation


def test_mange_pas_ne_mange_pas(mock_lexique):
    """'je mange pas' -> 'je ne mange pas'."""
    mots = ["je", "mange", "pas"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["je", "ne", "mange", "pas"]
    assert len(corrections) == 1


def test_dort_plus(mock_lexique):
    """'il dort plus' -> 'il ne dort plus' (fin de phrase, negation)."""
    mots = ["il", "dort", "plus"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "dort", "plus"]
    assert len(corrections) == 1


def test_mange_jamais(mock_lexique):
    """'il mange jamais' -> 'il ne mange jamais'."""
    mots = ["il", "mange", "jamais"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "mange", "jamais"]
    assert len(corrections) == 1


def test_mange_rien(mock_lexique):
    """'il mange rien' -> 'il ne mange rien'."""
    mots = ["il", "mange", "rien"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "mange", "rien"]
    assert len(corrections) == 1


def test_pas_correction_si_ne_present(mock_lexique):
    """'il ne mange pas' -> pas de correction."""
    mots = ["il", "ne", "mange", "pas"]
    pos = ["PRO:per", "ADV", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "mange", "pas"]
    assert len(corrections) == 0


# --- "plus" comparatif/superlatif ---

def test_plus_comparatif_pas_negation(mock_lexique):
    """'c'est plus difficile' -> pas de 'ne' (comparatif)."""
    mots = ["c'est", "plus", "difficile"]
    pos = ["VER", "ADV", "ADJ"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["c'est", "plus", "difficile"]
    assert len(corrections) == 0


def test_plus_superlatif_pas_negation(mock_lexique):
    """'le plus beau' -> pas de 'ne' (superlatif)."""
    mots = ["le", "plus", "beau"]
    pos = ["ART:def", "ADV", "ADJ"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["le", "plus", "beau"]
    assert len(corrections) == 0


def test_plus_negation_fin_phrase(mock_lexique):
    """'il mange plus' -> 'il ne mange plus' (negation en fin de phrase)."""
    mots = ["il", "mange", "plus"]
    pos = ["PRO:per", "VER", "ADV"]
    # "plus" en fin de phrase, pas suivi de ADJ/ADV -> negation
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "mange", "plus"]
    assert len(corrections) == 1


# --- Elision ne -> n' ---

def test_elision_ne_voyelle(mock_lexique):
    """'il aime pas' -> 'il n'aime pas' (elision devant voyelle)."""
    mots = ["il", "aime", "pas"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "n'aime", "pas"]
    assert len(corrections) == 1
    assert corrections[0].corrige == "n'aime"


def test_elision_ne_h(mock_lexique):
    """'il habite pas' -> 'il n'habite pas' (elision devant h)."""
    mots = ["il", "habite", "pas"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "n'habite", "pas"]
    assert len(corrections) == 1
    assert corrections[0].corrige == "n'habite"


def test_ne_consonne_normal(mock_lexique):
    """'il mange pas' -> 'il ne mange pas' (consonne, pas d'elision)."""
    mots = ["il", "mange", "pas"]
    pos = ["PRO:per", "VER", "ADV"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "mange", "pas"]
    assert len(corrections) == 1
    assert corrections[0].corrige == "ne"


# --- "personne" comme negatif ---

def test_mange_personne(mock_lexique):
    """'il voit personne' -> insert ne (negation)."""
    mots = ["il", "voit", "personne"]
    pos = ["PRO:per", "VER", "PRO:ind"]
    result, corrections = verifier_negation(
        mots, pos, {}, mock_lexique,
    )
    assert result == ["il", "ne", "voit", "personne"]
    assert len(corrections) == 1
