"""Tests pour l'index SymSpell."""

from __future__ import annotations

import pytest

from lectura_correcteur._symspell import SymSpellIndex, _deletions, _obtenir_formes


class TestDeletions:
    """Tests de la fonction _deletions."""

    def test_deletions_basique(self):
        result = _deletions("abc")
        assert sorted(result) == ["ab", "ac", "bc"]

    def test_deletions_mot_1_char(self):
        result = _deletions("a")
        assert result == [""]

    def test_deletions_vide(self):
        result = _deletions("")
        assert result == []


class TestSymSpellIndex:
    """Tests de la construction et du lookup."""

    @pytest.fixture
    def petit_index(self):
        formes = frozenset({"chat", "chats", "chant", "maison", "mange"})
        return SymSpellIndex(formes)

    def test_build(self, petit_index):
        """L'index se construit sans erreur."""
        assert petit_index._formes == frozenset({"chat", "chats", "chant", "maison", "mange"})

    def test_substitution_d1(self, petit_index):
        """Substitution d=1 : 'chzt' -> 'chat' (z->a)."""
        sugg = petit_index.suggestions("chzt")
        assert "chat" in sugg

    def test_deletion_d1(self, petit_index):
        """Deletion d=1 : 'cht' -> 'chat'."""
        sugg = petit_index.suggestions("cht")
        assert "chat" in sugg

    def test_insertion_d1(self, petit_index):
        """Insertion d=1 : 'chaat' -> 'chat'."""
        sugg = petit_index.suggestions("chaat")
        assert "chat" in sugg

    def test_transposition_d1(self, petit_index):
        """Transposition = 2 operations (del+ins), donc trouvee via d=2 deletes."""
        sugg = petit_index.suggestions("caht")
        # Transposition chat->caht est d=2 en deletions, devrait etre trouvee
        assert "chat" in sugg

    def test_d2_trouve(self, petit_index):
        """Distance 2 : 'mison' -> 'maison'."""
        sugg = petit_index.suggestions("mison")
        assert "maison" in sugg

    def test_mot_inexistant_loin(self, petit_index):
        """Un mot tres different ne retourne rien."""
        sugg = petit_index.suggestions("zzzzzzz")
        assert len(sugg) == 0

    def test_pas_auto_suggestion(self, petit_index):
        """Le mot lui-meme n'est pas dans les suggestions."""
        sugg = petit_index.suggestions("chat")
        assert "chat" not in sugg

    def test_max_n(self, petit_index):
        """Respecte la limite max_n."""
        sugg = petit_index.suggestions("chat", max_n=2)
        assert len(sugg) <= 2

    def test_suggestions_proches(self, petit_index):
        """'chant' est a d=1 de 'chat' et 'chats'."""
        sugg = petit_index.suggestions("chant")
        assert "chat" in sugg or "chats" in sugg


class TestObtenirFormes:
    """Tests de l'extraction duck-typed des formes."""

    def test_avec_dict_formes(self, mock_lexique):
        """MockLexique a _formes dict -> extraction OK."""
        formes = _obtenir_formes(mock_lexique)
        assert formes is not None
        assert "chat" in formes
        assert "maison" in formes

    def test_avec_frozenset_formes(self):
        """Objet avec _formes frozenset -> extraction OK."""
        class FakeLexique:
            _formes = frozenset({"a", "b"})
        formes = _obtenir_formes(FakeLexique())
        assert formes == frozenset({"a", "b"})

    def test_avec_toutes_formes_methode(self):
        """Objet avec toutes_formes() -> extraction OK."""
        class FakeLexique:
            def toutes_formes(self):
                return ["x", "y", "z"]
        formes = _obtenir_formes(FakeLexique())
        assert formes == frozenset({"x", "y", "z"})

    def test_sans_attribut(self):
        """Objet sans attribut connu -> None."""
        class FakeLexique:
            pass
        formes = _obtenir_formes(FakeLexique())
        assert formes is None

    def test_formes_set_fallback(self):
        """Objet avec _formes_set frozenset -> extraction OK."""
        class FakeLexique:
            _formes_set = frozenset({"p", "q"})
        formes = _obtenir_formes(FakeLexique())
        assert formes == frozenset({"p", "q"})
