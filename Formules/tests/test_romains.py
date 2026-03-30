"""Tests unitaires pour romains.py — chiffres romains bidirectionnels."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_formules.romains import int_to_roman, roman_to_int


class TestIntToRoman:
    def test_basic(self):
        assert int_to_roman(1) == "I"
        assert int_to_roman(2) == "II"
        assert int_to_roman(3) == "III"

    def test_subtractive(self):
        assert int_to_roman(4) == "IV"
        assert int_to_roman(9) == "IX"
        assert int_to_roman(40) == "XL"
        assert int_to_roman(90) == "XC"
        assert int_to_roman(400) == "CD"
        assert int_to_roman(900) == "CM"

    def test_standard(self):
        assert int_to_roman(5) == "V"
        assert int_to_roman(10) == "X"
        assert int_to_roman(50) == "L"
        assert int_to_roman(100) == "C"
        assert int_to_roman(500) == "D"
        assert int_to_roman(1000) == "M"

    def test_combined(self):
        assert int_to_roman(14) == "XIV"
        assert int_to_roman(42) == "XLII"
        assert int_to_roman(99) == "XCIX"
        assert int_to_roman(342) == "CCCXLII"
        assert int_to_roman(1999) == "MCMXCIX"
        assert int_to_roman(2024) == "MMXXIV"
        assert int_to_roman(3999) == "MMMCMXCIX"

    def test_vinculum(self):
        assert int_to_roman(5000) == "V̅"
        assert int_to_roman(10000) == "X̅"
        assert int_to_roman(6000) == "V̅M"
        assert int_to_roman(39999) == "X̅X̅X̅MX̅CMXCIX"

    def test_error_zero(self):
        with pytest.raises(ValueError):
            int_to_roman(0)

    def test_error_negative(self):
        with pytest.raises(ValueError):
            int_to_roman(-1)

    def test_error_too_large(self):
        with pytest.raises(ValueError):
            int_to_roman(40000)


class TestRomanToInt:
    def test_basic(self):
        assert roman_to_int("I") == 1
        assert roman_to_int("V") == 5
        assert roman_to_int("X") == 10

    def test_subtractive(self):
        assert roman_to_int("IV") == 4
        assert roman_to_int("IX") == 9
        assert roman_to_int("XL") == 40
        assert roman_to_int("XC") == 90
        assert roman_to_int("CD") == 400
        assert roman_to_int("CM") == 900

    def test_combined(self):
        assert roman_to_int("XLII") == 42
        assert roman_to_int("XCIX") == 99
        assert roman_to_int("MCMXCIX") == 1999
        assert roman_to_int("MMXXIV") == 2024

    def test_vinculum(self):
        assert roman_to_int("V̅") == 5000
        assert roman_to_int("X̅") == 10000

    def test_error_empty(self):
        with pytest.raises(ValueError):
            roman_to_int("")

    def test_error_invalid(self):
        with pytest.raises(ValueError):
            roman_to_int("Z")


class TestRoundTrip:
    def test_round_trip_small(self):
        for n in range(1, 100):
            assert roman_to_int(int_to_roman(n)) == n

    def test_round_trip_large(self):
        for n in [100, 342, 999, 1000, 1999, 2024, 3999, 5000, 10000]:
            assert roman_to_int(int_to_roman(n)) == n
