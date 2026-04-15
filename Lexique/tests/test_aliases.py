"""Tests pour le module _aliases."""

from lectura_lexique._aliases import ALIAS_MAP, resoudre_colonnes


def test_alias_map_contient_ortho():
    assert "ortho" in ALIAS_MAP
    assert "graphie" in ALIAS_MAP["ortho"]


def test_alias_map_contient_phone():
    assert "phone" in ALIAS_MAP
    assert "phon" in ALIAS_MAP["phone"]
    assert "ipa" in ALIAS_MAP["phone"]


def test_resoudre_colonnes_canoniques():
    """Les noms deja canoniques sont preserves."""
    mapping = resoudre_colonnes(["ortho", "phone", "cgram"])
    assert mapping["ortho"] == "ortho"
    assert mapping["phone"] == "phone"
    assert mapping["cgram"] == "cgram"


def test_resoudre_colonnes_alias():
    """Les alias sont resolus vers les noms canoniques."""
    mapping = resoudre_colonnes(["graphie", "phon", "category"])
    assert mapping["graphie"] == "ortho"
    assert mapping["phon"] == "phone"
    assert mapping["category"] == "cgram"


def test_resoudre_colonnes_inconnues():
    """Les colonnes inconnues sont gardees en lower."""
    mapping = resoudre_colonnes(["ortho", "custom_field"])
    assert mapping["custom_field"] == "custom_field"


def test_resoudre_colonnes_freq_alias():
    """Les alias de frequence pointent vers 'freq'."""
    mapping = resoudre_colonnes(["freq_opensubs", "freq_frantext"])
    assert mapping["freq_opensubs"] == "freq"
    assert mapping["freq_frantext"] == "freq"
