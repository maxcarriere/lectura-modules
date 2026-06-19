"""Tests unitaires pour le module phonemes."""

import pytest


def test_ipa_to_phones_simple():
    from lectura_tts_multispeaker.phonemes import ipa_to_phones

    phones = ipa_to_phones("b\u0254\u0303\u0292u\u0281")
    assert phones == ["b", "\u0254\u0303", "\u0292", "u", "\u0281"]


def test_ipa_to_phones_multichar():
    from lectura_tts_multispeaker.phonemes import ipa_to_phones

    phones = ipa_to_phones("t\u0283")
    assert phones == ["t\u0283"]


def test_ipa_to_phones_nasals():
    from lectura_tts_multispeaker.phonemes import ipa_to_phones

    phones = ipa_to_phones("\u025b\u0303t\u025bli\u0292\u0251\u0303")
    assert "\u025b\u0303" in phones
    assert "\u0251\u0303" in phones


def test_ipa_to_phone_ids():
    from lectura_tts_multispeaker.phonemes import ipa_to_phone_ids

    ids = ipa_to_phone_ids("b\u0254\u0303\u0292u\u0281")
    assert ids[0] == 2  # SIL
    assert ids[-1] == 2  # SIL
    assert len(ids) == 7  # SIL + 5 phones + SIL


def test_ipa_to_phone_ids_no_silence():
    from lectura_tts_multispeaker.phonemes import ipa_to_phone_ids

    ids = ipa_to_phone_ids("ba", add_silence=False)
    assert ids[0] != 2 or len(ids) == 2


def test_phones_to_ids():
    from lectura_tts_multispeaker.phonemes import phones_to_ids

    ids = phones_to_ids(["b", "\u0254\u0303", "\u0292", "u", "\u0281"])
    assert len(ids) == 5
    assert all(isinstance(i, int) for i in ids)


def test_get_vocab():
    from lectura_tts_multispeaker.phonemes import get_vocab

    vocab = get_vocab()
    assert len(vocab) == 51
    assert vocab[0] == "<PAD>"
    assert vocab[2] == "#"


def test_unknown_phone():
    from lectura_tts_multispeaker.phonemes import phones_to_ids

    ids = phones_to_ids(["INCONNU"])
    assert ids == [1]  # UNK_ID
