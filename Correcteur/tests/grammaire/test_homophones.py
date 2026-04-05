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


# --- est (VER) -> et (coordination) ---

def test_est_entre_noms(mock_lexique):
    """'chat est les enfants' -> 'chat et les enfants'."""
    mots = ["chat", "est", "les", "enfants"]
    pos = ["NOM", "AUX", "ART:def", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "et"
    assert len(corrections) >= 1


def test_est_avant_pronom(mock_lexique):
    """'maison est il mange' -> 'maison et il mange'."""
    mots = ["maison", "est", "il", "mange"]
    pos = ["NOM", "AUX", "PRO:per", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "et"
    assert len(corrections) >= 1


def test_pas_correction_copule(mock_lexique):
    """'il est grand' -> pas de correction (copule correcte)."""
    mots = ["il", "est", "grand"]
    pos = ["PRO:per", "AUX", "ADJ"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "est"
    assert not any(c.corrige == "et" for c in corrections)


# --- a / à ---

def test_a_preposition_devant_verbe(mock_lexique):
    """'a' (VER) sans sujet 3sg + suivi de VER -> 'à' (preposition)."""
    mots = ["mange", "a", "manger"]
    pos = ["VER", "VER", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "à"
    assert any(c.corrige == "à" for c in corrections)


def test_a_auxiliaire_correct(mock_lexique):
    """'il a mangé' -> pas de correction ('a' correct apres sujet)."""
    mots = ["il", "a", "mangé"]
    pos = ["PRO:per", "AUX", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "a"
    assert not any(c.corrige == "à" for c in corrections)


# --- ou / où ---

def test_ou_pronom_relatif(mock_lexique):
    """'ou' etiquete PRO:rel -> 'où'."""
    mots = ["la", "maison", "ou", "il", "mange"]
    pos = ["ART:def", "NOM", "PRO:rel", "PRO:per", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "où"
    assert any(c.corrige == "où" for c in corrections)


def test_ou_conjonction_correct(mock_lexique):
    """'chat ou chien' -> pas de correction (conjonction correcte)."""
    mots = ["chat", "ou", "chien"]
    pos = ["NOM", "CON", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "ou"
    assert not any(c.corrige == "où" for c in corrections)


# --- ce / se ---

def test_ce_devant_verbe_se(mock_lexique):
    """'ce' (DET) devant VER -> 'se' (pronom reflexif)."""
    mots = ["il", "ce", "mange"]
    pos = ["PRO:per", "DET:dem", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "se"
    assert any(c.corrige == "se" for c in corrections)


def test_se_devant_nom_ce(mock_lexique):
    """'se' (PRO:per) devant NOM -> 'ce' (determinant)."""
    mots = ["se", "chat", "mange"]
    pos = ["PRO:per", "NOM", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "ce"
    assert any(c.corrige == "ce" for c in corrections)


# --- la / là ---

def test_la_adverbe_la(mock_lexique):
    """'la' etiquete ADV -> 'là' (adverbe de lieu)."""
    mots = ["il", "est", "la"]
    pos = ["PRO:per", "AUX", "ADV"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[2] == "là"
    assert any(c.corrige == "là" for c in corrections)


def test_la_article_correct(mock_lexique):
    """'la maison' -> pas de correction (article correct)."""
    mots = ["la", "maison"]
    pos = ["ART:def", "NOM"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[0] == "la"
    assert not any(c.corrige == "là" for c in corrections)


# --- on / ont ---

def test_on_apres_nom_ont(mock_lexique):
    """'ils on mangé' -> 'ils ont mangé' (auxiliaire 3pl)."""
    mots = ["ils", "on", "mangé"]
    pos = ["PRO:per", "PRO:ind", "VER"]
    result, corrections = verifier_homophones(
        mots, pos, {}, mock_lexique,
    )
    assert result[1] == "ont"
    assert any(c.corrige == "ont" for c in corrections)
