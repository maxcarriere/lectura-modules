"""Tests pour lectura_tts_diphone.phonemes."""

import pytest


def test_import():
    """Le module s'importe sans deps lourdes."""
    from lectura_tts_diphone.phonemes import ipa_to_phones, phone_to_id, syllables_to_phones
    assert callable(ipa_to_phones)
    assert callable(phone_to_id)
    assert callable(syllables_to_phones)


def test_ipa_to_phones_simple():
    from lectura_tts_diphone.phonemes import ipa_to_phones
    phones = ipa_to_phones("ba")
    assert phones == ["b", "a"]


def test_ipa_to_phones_nasals():
    from lectura_tts_diphone.phonemes import ipa_to_phones
    phones = ipa_to_phones("b\u0254\u0303")  # bɔ̃
    assert len(phones) == 2
    assert phones[0] == "b"
    assert phones[1] == "\u0254\u0303"  # ɔ̃


def test_ipa_to_phones_bonjour():
    from lectura_tts_diphone.phonemes import ipa_to_phones
    phones = ipa_to_phones("b\u0254\u0303\u0292u\u0281")
    assert phones == ["b", "\u0254\u0303", "\u0292", "u", "\u0281"]


def test_ipa_to_phones_affricate():
    from lectura_tts_diphone.phonemes import ipa_to_phones
    phones = ipa_to_phones("t\u0283a")
    assert "t\u0283" in phones
    assert "a" in phones


def test_ipa_to_phones_empty():
    from lectura_tts_diphone.phonemes import ipa_to_phones
    assert ipa_to_phones("") == []


def test_phone_to_id_known():
    from lectura_tts_diphone.phonemes import phone_to_id
    id_b = phone_to_id("b")
    assert isinstance(id_b, int)
    assert id_b > 1  # pas PAD ni UNK


def test_phone_to_id_unknown():
    from lectura_tts_diphone.phonemes import phone_to_id
    id_unk = phone_to_id("XXXYYY")
    assert id_unk == 1  # UNK


def test_phone_to_id_fallback():
    from lectura_tts_diphone.phonemes import phone_to_id
    # "r" → "ʁ"
    id_r = phone_to_id("r")
    id_R = phone_to_id("\u0281")
    assert id_r == id_R


def test_syllables_to_phones():
    from lectura_tts_diphone.phonemes import syllables_to_phones
    phones = syllables_to_phones(["b\u0254\u0303", "\u0292u\u0281"])
    assert phones == ["b", "\u0254\u0303", "\u0292", "u", "\u0281"]


def test_get_phone2id_loaded():
    from lectura_tts_diphone.phonemes import get_phone2id
    p2id = get_phone2id()
    assert isinstance(p2id, dict)
    assert "#" in p2id
    assert "a" in p2id
    assert len(p2id) >= 40
