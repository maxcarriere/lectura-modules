"""Tests du decodeur multext."""

from lectura_lexique._multext import decoder_multext, filtre_multext


# --- decoder_multext ---

def test_decoder_verb_indicatif_present():
    result = decoder_multext("Vmip3s")
    assert result["pos"] == "VER"
    assert result["mode"] == "indicatif"
    assert result["temps"] == "present"
    assert result["personne"] == "3"
    assert result["nombre"] == "singulier"


def test_decoder_verb_participe_passe():
    result = decoder_multext("Vmps-sm")
    assert result["pos"] == "VER"
    assert result["mode"] == "participe"
    assert result["temps"] == "passe_simple"


def test_decoder_verb_infinitif():
    result = decoder_multext("Vmn----")
    assert result["pos"] == "VER"
    assert result["mode"] == "infinitif"


def test_decoder_nom_commun():
    result = decoder_multext("Ncms")
    assert result["pos"] == "NOM"
    assert result["sous_type"] == "commun"
    assert result["genre"] == "masculin"
    assert result["nombre"] == "singulier"


def test_decoder_nom_propre():
    result = decoder_multext("Np")
    assert result["pos"] == "NOM"
    assert result["sous_type"] == "propre"


def test_decoder_adj():
    result = decoder_multext("Afpfs")
    assert result["pos"] == "ADJ"
    assert result["genre"] == "feminin"
    assert result["nombre"] == "singulier"


def test_decoder_vide():
    assert decoder_multext("") == {}


def test_decoder_inconnu():
    result = decoder_multext("Z")
    assert result["pos"] == "Z"


def test_decoder_verb_subjonctif():
    result = decoder_multext("Vmsp3s")
    assert result["pos"] == "VER"
    assert result["mode"] == "subjonctif"
    assert result["temps"] == "present"


def test_decoder_determinant():
    result = decoder_multext("Da-ms")
    assert result["pos"] == "DET"
    assert result["sous_type"] == "article"
    assert result["genre"] == "masculin"
    assert result["nombre"] == "singulier"


# --- filtre_multext ---

def test_filtre_verb_indicatif():
    pattern = filtre_multext(pos="VER", mode="indicatif")
    assert pattern.startswith("V")
    assert "i" in pattern


def test_filtre_nom_feminin():
    pattern = filtre_multext(pos="NOM", genre="feminin")
    assert pattern.startswith("N")
    assert "f" in pattern


def test_filtre_nombre_pluriel():
    pattern = filtre_multext(pos="NOM", nombre="pluriel")
    assert pattern.startswith("N")
    assert "p" in pattern


def test_filtre_sans_pos():
    pattern = filtre_multext(mode="indicatif")
    assert pattern.startswith("V")


def test_filtre_vide():
    pattern = filtre_multext()
    assert pattern == "%"
