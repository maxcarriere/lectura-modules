"""Tests unitaires pour tables.py — TablesStore."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_formules.tables import TablesStore, UniteDef, get_store, get_sound_path, get_sounds_dir


class TestTablesStoreLoad:
    def test_load_returns_instance(self):
        store = TablesStore.load()
        assert isinstance(store, TablesStore)

    def test_unite_by_label_non_empty(self):
        store = get_store()
        assert len(store.unite_by_label) >= 70

    def test_unite_zero(self):
        store = get_store()
        ud = store.unite_by_label["[0]"]
        assert ud.texte == "zero"
        assert ud.phone == "zeʁo"
        assert ud.sound == "num002"
        assert ud.value == 0

    def test_unite_one(self):
        store = get_store()
        ud = store.unite_by_label["[1]"]
        assert ud.texte == "un"
        assert ud.display_rom == "I"
        assert ud.value == 1

    def test_unite_mille(self):
        store = get_store()
        ud = store.unite_by_label["[1000]"]
        assert ud.texte == "mille"
        assert ud.display_rom == "M"

    def test_variantes_special(self):
        store = get_store()
        assert "[20t]" in store.unite_by_label
        assert "[100s]" in store.unite_by_label
        assert "[et1]" in store.unite_by_label
        assert "[1ne]" in store.unite_by_label


class TestNumeraux:
    def test_numeral_mapping(self):
        store = get_store()
        assert store.numeral_to_unit["zero"] == "[0]"
        assert store.numeral_to_unit["un"] == "[1]"
        assert store.numeral_to_unit["mille"] == "[1000]"


class TestSymboles:
    def test_roman_symbols(self):
        store = get_store()
        assert store.symbol_to_unit["I"] == "[1]"
        assert store.symbol_to_unit["V"] == "[5]"
        assert store.symbol_to_unit["X"] == "[10]"
        assert store.symbol_to_unit["M"] == "[1000]"


class TestRomainBlocks:
    def test_block_4(self):
        store = get_store()
        assert store.roman_block_map["[4]"] == ["[1]", "[5]"]

    def test_block_9(self):
        store = get_store()
        assert store.roman_block_map["[9]"] == ["[1]", "[10]"]


class TestOrdinaux:
    def test_ordinal_un(self):
        store = get_store()
        texte, phone, sound = store.ordinal_map["un"]
        assert texte == "unième"
        assert sound == "ord001"

    def test_ordinal_count(self):
        store = get_store()
        assert len(store.ordinal_map) >= 25


class TestLettres:
    def test_letter_A(self):
        store = get_store()
        texte, phone, sound = store.letter_map["A"]
        assert texte == "a"
        assert sound == "let001"

    def test_letter_count(self):
        store = get_store()
        assert len(store.letter_map) >= 52


class TestSymbolesSons:
    def test_euro(self):
        store = get_store()
        texte, phone, sound, cat = store.symbol_sound_map["€"]
        assert texte == "euro"
        assert cat == "currency"

    def test_percent(self):
        store = get_store()
        assert "%" in store.symbol_sound_map

    def test_count(self):
        store = get_store()
        assert len(store.symbol_sound_map) >= 100


class TestCalendrier:
    def test_month_janvier(self):
        store = get_store()
        texte, phone, sound = store.month_map[1]
        assert texte == "janvier"
        assert sound == "cal001"

    def test_month_decembre(self):
        store = get_store()
        texte, phone, sound = store.month_map[12]
        assert texte == "décembre"
        assert sound == "cal012"

    def test_all_months_present(self):
        store = get_store()
        for i in range(1, 13):
            assert i in store.month_map

    def test_days_present(self):
        store = get_store()
        assert len(store.day_map) == 7
        assert "lundi" in store.day_map


class TestGrec:
    def test_alpha(self):
        store = get_store()
        texte, phone, sound = store.greek_map["α"]
        assert texte == "alpha"
        assert sound == "grk001"

    def test_prime(self):
        store = get_store()
        assert "'" in store.greek_map or "′" in store.greek_map


class TestSoundPath:
    def test_existing_sound(self):
        path = get_sound_path("num010")
        assert path is not None
        assert path.exists()
        assert path.suffix == ".wav"

    def test_nonexistent_sound(self):
        path = get_sound_path("nonexistent999")
        assert path is None

    def test_sounds_dir_exists(self):
        d = get_sounds_dir()
        assert Path(d).exists()
