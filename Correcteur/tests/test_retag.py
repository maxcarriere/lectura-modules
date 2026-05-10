"""Tests : re-tagging POS apres correction orthographique.

Verifie que les regles de grammaire recoivent le POS/morpho des
formes **corrigees** (decided_words), pas celui des formes originales.
"""

from __future__ import annotations

from lectura_correcteur import Correcteur
from lectura_correcteur._config import CorrecteurConfig
from tests.conftest import MockLexique


def _build_lexique_dan() -> MockLexique:
    """Lexique ou 'dand' (OOV) -> 'dans' (PRE, freq=7000).

    'dand' est hors-lexique, donc le verificateur le corrige en 'dans'
    via SymSpell (distance 1). Apres correction, le POS doit etre PRE.
    On ajoute aussi des mots contexte pour former une phrase valide.
    """
    return MockLexique(formes={
        "dans": [{"ortho": "dans", "cgram": "PRE", "phone": "dɑ̃",
                  "freq": 7000.0}],
        "le": [{"ortho": "le", "cgram": "ART:def", "phone": "lə",
                "freq": 890.0, "genre": "m", "nombre": "s"}],
        "la": [{"ortho": "la", "cgram": "ART:def", "phone": "la",
                "freq": 720.0, "genre": "f", "nombre": "s"}],
        "chat": [{"ortho": "chat", "cgram": "NOM", "phone": "ʃa",
                  "freq": 45.0, "genre": "m", "nombre": "s"}],
        "jardin": [{"ortho": "jardin", "cgram": "NOM", "phone": "ʒaʁdɛ̃",
                    "freq": 20.0, "genre": "m", "nombre": "s"}],
    })


def _build_lexique_ecol() -> MockLexique:
    """Lexique ou 'ecol' (OOV) -> 'ecole' (NOM, freq=200).

    'ecol' est hors-lexique, SymSpell propose 'ecole' a distance 1.
    """
    return MockLexique(formes={
        "ecole": [{"ortho": "ecole", "cgram": "NOM", "phone": "ekɔl",
                   "freq": 200.0, "genre": "f", "nombre": "s"}],
        "la": [{"ortho": "la", "cgram": "ART:def", "phone": "la",
                "freq": 720.0, "genre": "f", "nombre": "s"}],
        "le": [{"ortho": "le", "cgram": "ART:def", "phone": "lə",
                "freq": 890.0, "genre": "m", "nombre": "s"}],
    })


# ---------------------------------------------------------------
# 1. POS reflete la forme corrigee apres correction orthographique
# ---------------------------------------------------------------

def test_retag_apres_correction_orthographe():
    """'dand' (OOV) corrige en 'dans' -> POS doit etre PRE, pas vide."""
    lex = _build_lexique_dan()
    config = CorrecteurConfig(
        activer_grammaire=True,
        activer_syntaxe=False,
    )
    c = Correcteur(lex, config=config)
    result = c.corriger("le chat dand le jardin")

    # Trouver le mot corrige "dans"
    mot_dans = None
    for m in result.mots:
        if m.original.lower() == "dand":
            mot_dans = m
            break
    assert mot_dans is not None, "Le mot 'dand' doit etre dans les analyses"
    assert mot_dans.corrige == "dans", f"dand -> dans, got {mot_dans.corrige}"
    # Le POS doit refleter la forme corrigee (PRE), pas rester vide
    assert mot_dans.pos == "PRE", (
        f"POS de 'dans' devrait etre PRE, got '{mot_dans.pos}'"
    )


# ---------------------------------------------------------------
# 2. Mots non corriges gardent leur POS d'origine
# ---------------------------------------------------------------

def test_retag_mot_non_corrige():
    """'chat' (dans le lexique) garde son POS NOM inchange."""
    lex = _build_lexique_dan()
    config = CorrecteurConfig(
        activer_grammaire=True,
        activer_syntaxe=False,
    )
    c = Correcteur(lex, config=config)
    result = c.corriger("le chat dand le jardin")

    mot_chat = None
    for m in result.mots:
        if m.original.lower() == "chat":
            mot_chat = m
            break
    assert mot_chat is not None
    assert mot_chat.corrige == "chat"
    assert mot_chat.pos == "NOM", (
        f"POS de 'chat' devrait rester NOM, got '{mot_chat.pos}'"
    )


# ---------------------------------------------------------------
# 3. Mot OOV corrige recoit le POS de la forme corrigee
# ---------------------------------------------------------------

def test_retag_oov_corrige():
    """'ecol' (OOV) corrige en 'ecole' -> POS doit etre NOM."""
    lex = _build_lexique_ecol()
    config = CorrecteurConfig(
        activer_grammaire=True,
        activer_syntaxe=False,
    )
    c = Correcteur(lex, config=config)
    result = c.corriger("la ecol")

    mot_ecol = None
    for m in result.mots:
        if m.original.lower() == "ecol":
            mot_ecol = m
            break
    assert mot_ecol is not None, "Le mot 'ecol' doit etre dans les analyses"
    assert mot_ecol.corrige == "ecole", f"ecol -> ecole, got {mot_ecol.corrige}"
    assert mot_ecol.pos == "NOM", (
        f"POS de 'ecole' devrait etre NOM, got '{mot_ecol.pos}'"
    )
