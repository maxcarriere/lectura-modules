"""Tests pour la generation de candidats."""

from lectura_correcteur._candidats import generer_candidats, _edit_distance


def test_edit_distance_identique():
    assert _edit_distance("chat", "chat") == 0


def test_edit_distance_substitution():
    assert _edit_distance("chat", "chot") == 1


def test_edit_distance_insertion():
    assert _edit_distance("chat", "chats") == 1


def test_edit_distance_multi():
    assert _edit_distance("abc", "xyz") == 3


def test_tier1_identite_present(mock_lexique):
    """Le mot lui-meme est toujours candidat (source=identite)."""
    candidats = generer_candidats(
        "chat", True, "NOM", {}, mock_lexique,
    )
    identites = [c for c in candidats if c.source == "identite"]
    assert len(identites) == 1
    assert identites[0].forme == "chat"
    assert identites[0].freq > 0


def test_tier1_homophones(mock_lexique):
    """Les homophones sont generes (meme phone)."""
    # "chat" et "chats" ont le meme phone "ʃa"
    candidats = generer_candidats(
        "chat", True, "NOM", {}, mock_lexique,
    )
    formes = {c.forme for c in candidats}
    assert "chat" in formes
    assert "chats" in formes  # homophone (meme phone)


def test_tier2_hors_lexique(mock_lexique):
    """Les mots hors-lexique generent des candidats d=1."""
    candidats = generer_candidats(
        "chta", False, "NOM", {}, mock_lexique,
    )
    # "chat" est a d=1 de "chta" (transposition)
    formes = {c.forme for c in candidats}
    assert "chat" in formes or len(candidats) > 1  # au moins l'identite + d1


def test_tier3_morpho_in_lexique(mock_lexique):
    """Les mots in-lexique generent des variantes morphologiques."""
    candidats = generer_candidats(
        "mange", True, "VER", {}, mock_lexique,
    )
    sources = {c.source for c in candidats}
    # Doit contenir "identite" et potentiellement "homophone" et "morpho"
    assert "identite" in sources
    # "mange" a lemme "manger", donc formes_de("manger") retourne d'autres formes
    formes = {c.forme for c in candidats}
    # Les formes du meme lemme incluent mangé, manger, manges, mangent...
    assert "mangé" in formes or "manges" in formes or "mangent" in formes


def test_tier3_morpho_genere_pp():
    """Le tier 3 genere le participe passe quand le lemme est partage."""
    from tests.conftest import MockLexique

    formes = {
        "mange": [{"ortho": "mange", "cgram": "VER", "phone": "mɑ̃ʒ",
                    "freq": 15.0, "lemme": "manger", "personne": "3", "nombre": "s"}],
        "mangé": [{"ortho": "mangé", "cgram": "VER", "phone": "mɑ̃ʒe",
                   "freq": 8.0, "lemme": "manger", "mode": "participe"}],
        "manger": [{"ortho": "manger", "cgram": "VER", "phone": "mɑ̃ʒe",
                    "freq": 20.0, "lemme": "manger", "mode": "inf"}],
    }
    lex = MockLexique(formes=formes)
    candidats = generer_candidats("mange", True, "VER", {}, lex)
    formes_gen = {c.forme for c in candidats}
    assert "mangé" in formes_gen
    assert "manger" in formes_gen


def test_no_candidats_sans_lexique():
    """Sans methodes lexique, retourne au moins l'identite."""
    class MinimalLexique:
        def existe(self, mot): return True
        def info(self, mot): return []
        def frequence(self, mot): return 0.0

    candidats = generer_candidats(
        "chat", True, "NOM", {}, MinimalLexique(),
    )
    assert len(candidats) >= 1
    assert candidats[0].source == "identite"


# --- Tests suggestions injection (Tier 2 from verificateur) ---

def test_tier2_from_suggestions(mock_lexique):
    """Quand suggestions est fourni pour un OOV, utilise ces suggestions comme Tier 2."""
    candidats = generer_candidats(
        "chet", False, "NOM", {}, mock_lexique,
        suggestions=["chat"],
    )
    formes = {c.forme for c in candidats}
    sources = {c.source for c in candidats}
    # Identite toujours presente
    assert "chet" in formes
    # Suggestion injectee comme Tier 2
    assert "chat" in formes
    assert "ortho_suggestion" in sources
    # Verifie que le candidat "chat" est enrichi avec POS/morpho
    chat_cands = [c for c in candidats if c.forme == "chat"]
    assert len(chat_cands) >= 1
    chat_c = chat_cands[0]
    assert chat_c.pos == "NOM"
    assert chat_c.freq > 0


def test_tier2_suggestions_none_fallback(mock_lexique):
    """Quand suggestions=None, fallback sur Tier 2 classique (d=1/d=2)."""
    candidats = generer_candidats(
        "chta", False, "NOM", {}, mock_lexique,
        suggestions=None,
    )
    sources = {c.source for c in candidats}
    # Pas de source "ortho_suggestion" car suggestions=None
    assert "ortho_suggestion" not in sources


def test_tier2_suggestions_dedup(mock_lexique):
    """Les suggestions deja vues en Tier 1 (homophones) ne sont pas dupliquees."""
    # "chats" est un homophone de "chat" (meme phone "ʃa")
    candidats = generer_candidats(
        "chat", False, "NOM", {}, mock_lexique,
        suggestions=["chats", "chat"],
    )
    formes = [c.forme for c in candidats]
    # "chat" apparait exactement une fois (identite)
    assert formes.count("chat") == 1
    # "chats" apparait via homophones ou suggestions, mais pas en double
    assert formes.count("chats") == 1
