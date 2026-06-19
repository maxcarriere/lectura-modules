"""Tests unitaires pour le module phonemes."""

import pytest


def test_ipa_to_phones_simple():
    from lectura_tts_monospeaker.phonemes import ipa_to_phones

    phones = ipa_to_phones("bɔ̃ʒuʁ")
    assert phones == ["b", "ɔ̃", "ʒ", "u", "ʁ"]


def test_ipa_to_phones_multichar():
    from lectura_tts_monospeaker.phonemes import ipa_to_phones

    phones = ipa_to_phones("tʃ")
    assert phones == ["tʃ"]


def test_ipa_to_phones_nasals():
    from lectura_tts_monospeaker.phonemes import ipa_to_phones

    phones = ipa_to_phones("ɛ̃tɛliʒɑ̃")
    assert "ɛ̃" in phones
    assert "ɑ̃" in phones


def test_ipa_to_phone_ids():
    from lectura_tts_monospeaker.phonemes import ipa_to_phone_ids

    ids = ipa_to_phone_ids("bɔ̃ʒuʁ")
    # SIL=2, b=21, ɔ̃=17, ʒ=31, u=8, ʁ=36, SIL=2
    assert ids[0] == 2  # SIL
    assert ids[-1] == 2  # SIL
    assert len(ids) == 7  # SIL + 5 phones + SIL


def test_ipa_to_phone_ids_no_silence():
    from lectura_tts_monospeaker.phonemes import ipa_to_phone_ids

    ids = ipa_to_phone_ids("ba", add_silence=False)
    assert ids[0] != 2 or len(ids) == 2  # pas de SIL ajoute


def test_phones_to_ids():
    from lectura_tts_monospeaker.phonemes import phones_to_ids

    ids = phones_to_ids(["b", "ɔ̃", "ʒ", "u", "ʁ"])
    assert len(ids) == 5
    assert all(isinstance(i, int) for i in ids)


def test_get_vocab():
    from lectura_tts_monospeaker.phonemes import get_vocab

    vocab = get_vocab()
    assert len(vocab) == 51
    assert vocab[0] == "<PAD>"
    assert vocab[2] == "#"


def test_unknown_phone():
    from lectura_tts_monospeaker.phonemes import phones_to_ids

    ids = phones_to_ids(["INCONNU"])
    assert ids == [1]  # UNK_ID
