"""Tests pour la suspicion par frequence des mots in-lexique rares."""

from __future__ import annotations

from lectura_correcteur._types import TypeCorrection
from lectura_correcteur.orthographe._verificateur import VerificateurOrthographe


# ---------------------------------------------------------------------------
# 1. Correction basique : mot rare -> mot tres frequent (seuils atteints)
# ---------------------------------------------------------------------------

def test_suspicion_correction_basique():
    """Mot rare (freq=0.1, len>3) -> candidat tres frequent (freq=1000, ratio>=500x)."""
    from tests.conftest import MockLexique
    formes = {
        "dand": [{"ortho": "dand", "cgram": "NOM", "phone": "dɑ̃", "freq": 0.1}],
        "dans": [{"ortho": "dans", "cgram": "PRE", "phone": "dɑ̃", "freq": 1000.0}],
        "le": [{"ortho": "le", "cgram": "ART:def", "phone": "lə", "freq": 890.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["le", "dand"])
    mot = results[1]
    assert mot.corrige == "dans"
    assert mot.type_correction == TypeCorrection.HORS_LEXIQUE


# ---------------------------------------------------------------------------
# 2. Pas de correction si freq suffisante (au-dessus du seuil suspect)
# ---------------------------------------------------------------------------

def test_pas_de_suspicion_si_freq_suffisante(mock_lexique):
    """'son' (freq=120) ne doit pas etre corrige meme si 'sont' (freq=200) existe."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["le", "son"])
    mot = results[1]
    assert mot.corrige == "son"


# ---------------------------------------------------------------------------
# 3. Pas de correction pour nom propre (capitalise, pas en debut)
# ---------------------------------------------------------------------------

def test_pas_de_suspicion_nom_propre():
    """'Dan' (capitalise, pas en debut de phrase) -> reste 'Dan'."""
    from tests.conftest import MockLexique
    formes = {
        "dan": [{"ortho": "dan", "cgram": "NOM", "phone": "dɑ̃", "freq": 0.1}],
        "dans": [{"ortho": "dans", "cgram": "PRE", "phone": "dɑ̃", "freq": 7000.0}],
        "le": [{"ortho": "le", "cgram": "ART:def", "phone": "lə", "freq": 890.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["le", "Dan"])
    mot = results[1]
    assert mot.corrige == "Dan"


# ---------------------------------------------------------------------------
# 4. Pas de correction en contexte etranger
# ---------------------------------------------------------------------------

def test_pas_de_suspicion_contexte_etranger():
    """Apres un nom propre OOV, pas de suspicion."""
    from tests.conftest import MockLexique
    formes = {
        "dan": [{"ortho": "dan", "cgram": "NOM", "phone": "dɑ̃", "freq": 0.1}],
        "dans": [{"ortho": "dans", "cgram": "PRE", "phone": "dɑ̃", "freq": 7000.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    # "McGregor" est OOV et capitalise -> contexte etranger pour "dan"
    results = v.verifier_phrase(["McGregor", "dan"])
    mot = results[1]
    assert mot.corrige == "dan"


# ---------------------------------------------------------------------------
# 5. Accent prioritaire : "tres" -> "très" par accent, pas par suspicion
# ---------------------------------------------------------------------------

def test_accent_prioritaire_sur_suspicion(mock_lexique):
    """'tres' (freq=1) -> deja corrige en accent, suspicion ne fire pas."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["tres"])
    mot = results[0]
    assert mot.corrige == "très"


# ---------------------------------------------------------------------------
# 6. Freq absolue insuffisante : ratio enorme mais candidat freq < seuil abs
# ---------------------------------------------------------------------------

def test_pas_de_suspicion_freq_abs_insuffisante():
    """Meme si le ratio est enorme, si le candidat a freq < seuil abs -> pas de correction."""
    from tests.conftest import MockLexique
    # len("abcde") = 5 > 3 -> seuil abs = 500, ratio = 500
    formes = {
        "abcde": [{"ortho": "abcde", "cgram": "NOM", "phone": "abc", "freq": 0.01}],
        "abcdf": [{"ortho": "abcdf", "cgram": "NOM", "phone": "abcd", "freq": 300.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["abcde"])
    mot = results[0]
    # "abcdf" a freq=300 < 500 (seuil abs pour len>3), pas de correction
    assert mot.corrige == "abcde"


# ---------------------------------------------------------------------------
# 7. Mots courts plus stricts : ratio 200x et freq 500+
# ---------------------------------------------------------------------------

def test_mots_courts_seuils_stricts():
    """Pour len<=3, le ratio doit etre >= 1000x et freq >= 2000."""
    from tests.conftest import MockLexique
    formes = {
        "pur": [{"ortho": "pur", "cgram": "ADJ", "phone": "pyʁ", "freq": 1.5}],
        "pour": [{"ortho": "pour", "cgram": "PRE", "phone": "puʁ", "freq": 1500.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["pur"])
    mot = results[0]
    # "pour" freq=1500 < 2000 (seuil abs pour len<=3), pas de correction
    assert mot.corrige == "pur"


def test_mots_courts_correction_quand_seuils_atteints():
    """Pour len<=3, correction si ratio >= 1000x ET freq >= 2000."""
    from tests.conftest import MockLexique
    formes = {
        "pur": [{"ortho": "pur", "cgram": "ADJ", "phone": "pyʁ", "freq": 0.5}],
        "pour": [{"ortho": "pour", "cgram": "PRE", "phone": "puʁ", "freq": 7000.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["pur"])
    mot = results[0]
    # "pour" freq=7000 >= 2000 et ratio=14000x >= 1000 -> correction
    assert mot.corrige == "pour"


# ---------------------------------------------------------------------------
# 8. Suspicion fonctionne sans SymSpell (generation inline)
# ---------------------------------------------------------------------------

def test_suspicion_fonctionne_sans_symspell(mock_lexique):
    """La suspicion utilise la generation inline de candidats d=1."""
    v = VerificateurOrthographe(mock_lexique)
    results = v.verifier_phrase(["le", "dan"])
    mot = results[1]
    assert mot.corrige == "dan"


# ---------------------------------------------------------------------------
# 9. Mot etranger pas corrige
# ---------------------------------------------------------------------------

def test_pas_de_suspicion_mot_etranger():
    """Les mots dans _MOTS_ETRANGERS ne sont pas touches par la suspicion."""
    from tests.conftest import MockLexique
    formes = {
        "set": [{"ortho": "set", "cgram": "NOM", "phone": "sɛt", "freq": 0.1}],
        "ses": [{"ortho": "ses", "cgram": "DET", "phone": "se", "freq": 800.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["set"])
    mot = results[0]
    # "set" is in _MOTS_ETRANGERS -> suspicion skipped
    assert mot.corrige == "set"


# ---------------------------------------------------------------------------
# 10. Variantes accent ignorees par suspicion (deja gerees)
# ---------------------------------------------------------------------------

def test_variante_accent_ignoree_par_suspicion():
    """Les candidats accent-only sont ignores par le bloc suspicion."""
    from tests.conftest import MockLexique
    formes = {
        "foret": [{"ortho": "foret", "cgram": "NOM", "phone": "fɔʁɛ", "freq": 1.0}],
        "forêt": [{"ortho": "forêt", "cgram": "NOM", "phone": "fɔʁɛ", "freq": 50.0}],
        "forte": [{"ortho": "forte", "cgram": "ADJ", "phone": "fɔʁt", "freq": 25.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["foret"])
    mot = results[0]
    # Accent block fires first: foret -> forêt
    assert mot.corrige == "forêt"


# ---------------------------------------------------------------------------
# 11. Garde NOM PROPRE : mot uniquement NOM PROPRE -> pas de correction
# ---------------------------------------------------------------------------

def test_pas_de_suspicion_nom_propre_cgram():
    """Si toutes les entrees du lexique sont NOM PROPRE, pas de suspicion freq,
    mais correction de casse (rome → Rome)."""
    from tests.conftest import MockLexique
    formes = {
        "rome": [{"ortho": "rome", "cgram": "NOM PROPRE", "phone": "ʁɔm", "freq": 0.0}],
        "robe": [{"ortho": "robe", "cgram": "NOM", "phone": "ʁɔb", "freq": 500.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["rome"])
    mot = results[0]
    # Pas de suspicion freq (pas robe), mais correction casse NOM PROPRE
    assert mot.corrige == "Rome"


def test_suspicion_mixte_np_et_nom():
    """Si le mot a NOM PROPRE + NOM, la suspicion peut s'appliquer."""
    from tests.conftest import MockLexique
    formes = {
        "dan": [
            {"ortho": "dan", "cgram": "NOM", "phone": "dɑ̃", "freq": 0.1},
            {"ortho": "dan", "cgram": "NOM PROPRE", "phone": "dɑ̃", "freq": 0.0},
        ],
        "dans": [{"ortho": "dans", "cgram": "PRE", "phone": "dɑ̃", "freq": 7000.0}],
        "le": [{"ortho": "le", "cgram": "ART:def", "phone": "lə", "freq": 890.0}],
    }
    lex = MockLexique(formes=formes)
    v = VerificateurOrthographe(lex)
    results = v.verifier_phrase(["le", "dan"])
    mot = results[1]
    # "dan" a aussi une entree NOM (pas seulement NOM PROPRE) -> suspicion active
    assert mot.corrige == "dans"
