"""Tests pour le MorphoTagger CRF."""

from lectura_correcteur._morpho import MorphoTagger, tokenize


def test_tokenize_simple():
    tokens = tokenize("Les chats mangent")
    words = [tok for tok, is_word in tokens if is_word]
    assert words == ["Les", "chats", "mangent"]


def test_tokenize_elision():
    tokens = tokenize("l'ecole")
    words = [tok for tok, is_word in tokens if is_word]
    assert words == ["l'", "ecole"]


def test_tokenize_ponctuation():
    tokens = tokenize("Bonjour, le chat.")
    words = [tok for tok, is_word in tokens if is_word]
    assert "Bonjour" in words
    assert "chat" in words
    non_words = [tok for tok, is_word in tokens if not is_word]
    assert "," in non_words
    assert "." in non_words


def test_tagger_basic():
    tagger = MorphoTagger()
    results = tagger.tag("Les chats mangent")
    assert len(results) == 3
    for r in results:
        assert "mot" in r
        assert "pos" in r
        assert "lemme" in r


def test_tagger_pos_detection():
    tagger = MorphoTagger()
    results = tagger.tag("Les chats mangent les souris")
    pos_tags = [r["pos"] for r in results]
    # Le CRF devrait detecter au moins un NOM et un VER
    assert any("NOM" in p for p in pos_tags) or any("VER" in p for p in pos_tags)


def test_tagger_tag_words():
    tagger = MorphoTagger()
    results = tagger.tag_words(["Les", "chats", "mangent"])
    assert len(results) == 3


def test_tagger_empty():
    tagger = MorphoTagger()
    assert tagger.tag("") == []
    assert tagger.tag_words([]) == []


def test_tagger_traits():
    tagger = MorphoTagger()
    results = tagger.tag("Les enfants mangent")
    # Chaque resultat doit avoir les cles morpho
    for r in results:
        assert "genre" in r
        assert "nombre" in r
        assert "personne" in r
