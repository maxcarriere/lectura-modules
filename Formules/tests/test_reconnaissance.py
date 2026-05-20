"""Tests unitaires pour reconnaissance.py — reconnaissance IPA → formule."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_formules.reconnaissance import reconnaitre_ipa, _tokenize_ipa
from lectura_formules.lecture_formules import (
    lire_nombre,
    lire_date,
    lire_heure,
    lire_monnaie,
    lire_pourcentage,
    lire_ordinal,
    lire_sigle,
)


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaire : round-trip (forward → reverse)
# ══════════════════════════════════════════════════════════════════════════════


def _round_trip(forward_fn, formula: str, expected_display_num: str | None = None):
    """Genere l'IPA via forward puis reconnait via reverse."""
    result_fwd = forward_fn(formula)
    ipa = result_fwd.phone
    result_rev = reconnaitre_ipa(ipa)
    assert result_rev is not None, (
        f"Echec reconnaissance pour {formula!r} (IPA: {ipa!r})"
    )
    # Le display_num reconstruit doit correspondre
    if expected_display_num is not None:
        assert result_rev.display_num == expected_display_num, (
            f"display_num: {result_rev.display_num!r} != {expected_display_num!r}"
        )
    # L'IPA reconstruit doit etre identique
    assert result_rev.phone.replace(" ", "") == ipa.replace(" ", ""), (
        f"IPA mismatch: {result_rev.phone!r} != {ipa!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Nombres
# ══════════════════════════════════════════════════════════════════════════════


class TestNombres:
    """Tests de reconnaissance pour les nombres."""

    def test_zero(self):
        _round_trip(lire_nombre, "0", "0")

    @pytest.mark.parametrize("n", range(1, 17))
    def test_1_a_16(self, n):
        _round_trip(lire_nombre, str(n), str(n))

    @pytest.mark.parametrize("n", [17, 18, 19])
    def test_17_a_19(self, n):
        _round_trip(lire_nombre, str(n), str(n))

    @pytest.mark.parametrize("n", [20, 21, 22, 29, 30, 31, 39])
    def test_20_a_39(self, n):
        _round_trip(lire_nombre, str(n), str(n))

    @pytest.mark.parametrize("n", [40, 41, 50, 51, 60, 61, 69])
    def test_40_a_69(self, n):
        _round_trip(lire_nombre, str(n), str(n))

    @pytest.mark.parametrize("n", [70, 71, 72, 79])
    def test_70_a_79(self, n):
        _round_trip(lire_nombre, str(n), str(n))

    @pytest.mark.parametrize("n", [80, 81, 89, 90, 91, 99])
    def test_80_a_99(self, n):
        _round_trip(lire_nombre, str(n), str(n))

    @pytest.mark.parametrize("n", [100, 101, 200, 201, 300, 500, 999])
    def test_centaines(self, n):
        _round_trip(lire_nombre, str(n), str(n))

    @pytest.mark.parametrize("n", [1000, 1001, 2000, 2500, 9999])
    def test_milliers(self, n):
        _round_trip(lire_nombre, str(n), str(n))

    @pytest.mark.parametrize("n", [1_000_000, 2_000_000, 1_234_567])
    def test_millions(self, n):
        _round_trip(lire_nombre, str(n))

    def test_negatif(self):
        _round_trip(lire_nombre, "-42", "-42")

    def test_decimal(self):
        _round_trip(lire_nombre, "3,14")


# ══════════════════════════════════════════════════════════════════════════════
# Dates
# ══════════════════════════════════════════════════════════════════════════════


class TestDates:
    """Tests de reconnaissance pour les dates."""

    def test_date_standard(self):
        _round_trip(lire_date, "15/03/2024", "15/03/2024")

    def test_date_1er(self):
        _round_trip(lire_date, "01/01/2000", "01/01/2000")

    def test_date_fin_annee(self):
        _round_trip(lire_date, "31/12/1999", "31/12/1999")


# ══════════════════════════════════════════════════════════════════════════════
# Heures
# ══════════════════════════════════════════════════════════════════════════════


class TestHeures:
    """Tests de reconnaissance pour les heures."""

    def test_heure_standard(self):
        _round_trip(lire_heure, "14h30", "14h30")

    def test_heure_zero_minutes(self):
        _round_trip(lire_heure, "1h00", "1h00")

    def test_heure_minuit(self):
        _round_trip(lire_heure, "0h15", "0h15")


# ══════════════════════════════════════════════════════════════════════════════
# Monnaies
# ══════════════════════════════════════════════════════════════════════════════


class TestMonnaies:
    """Tests de reconnaissance pour les monnaies."""

    def test_euros_entier(self):
        _round_trip(lire_monnaie, "42€")

    def test_euros_centimes(self):
        _round_trip(lire_monnaie, "42,50€")

    def test_dollars(self):
        _round_trip(lire_monnaie, "100$")


# ══════════════════════════════════════════════════════════════════════════════
# Pourcentages
# ══════════════════════════════════════════════════════════════════════════════


class TestPourcentages:
    """Tests de reconnaissance pour les pourcentages."""

    def test_pourcentage_simple(self):
        _round_trip(lire_pourcentage, "45%")

    def test_pour_mille(self):
        _round_trip(lire_pourcentage, "3‰")


# ══════════════════════════════════════════════════════════════════════════════
# Ordinaux
# ══════════════════════════════════════════════════════════════════════════════


class TestOrdinaux:
    """Tests de reconnaissance pour les ordinaux."""

    def test_premier(self):
        _round_trip(lire_ordinal, "1er", "1er")

    def test_deuxieme(self):
        _round_trip(lire_ordinal, "2e", "2e")

    def test_quarante_deuxieme(self):
        _round_trip(lire_ordinal, "42e", "42e")


# ══════════════════════════════════════════════════════════════════════════════
# Sigles
# ══════════════════════════════════════════════════════════════════════════════


class TestSigles:
    """Tests de reconnaissance pour les sigles."""

    def test_sncf(self):
        _round_trip(lire_sigle, "SNCF", "SNCF")

    def test_b2b(self):
        _round_trip(lire_sigle, "B2B", "B2B")


# ══════════════════════════════════════════════════════════════════════════════
# Tolerance espaces
# ══════════════════════════════════════════════════════════════════════════════


class TestToleranceEspaces:
    """Tests de tolerance aux espaces dans l'IPA."""

    def test_espaces_normaux(self):
        """IPA avec espaces normaux."""
        result = reconnaitre_ipa("kaʁɑ̃t dø")
        assert result is not None
        assert result.display_num == "42"

    def test_sans_espaces(self):
        """IPA sans aucun espace."""
        result = reconnaitre_ipa("kaʁɑ̃tdø")
        assert result is not None
        assert result.display_num == "42"

    def test_espaces_supplementaires(self):
        """IPA avec espaces supplementaires."""
        result = reconnaitre_ipa("kaʁɑ̃t  dø")
        assert result is not None
        assert result.display_num == "42"


# ══════════════════════════════════════════════════════════════════════════════
# Round-trip complet par type
# ══════════════════════════════════════════════════════════════════════════════


class TestRoundTrip:
    """Tests round-trip : forward → reverse pour chaque type."""

    @pytest.mark.parametrize("n", [0, 1, 7, 13, 20, 42, 71, 80, 99, 100, 200,
                                   500, 999, 1000, 1234, 10_000, 1_000_000])
    def test_nombre_round_trip(self, n):
        _round_trip(lire_nombre, str(n))

    def test_date_round_trip(self):
        _round_trip(lire_date, "25/12/2023")

    def test_heure_round_trip(self):
        _round_trip(lire_heure, "8h45")

    def test_monnaie_round_trip(self):
        _round_trip(lire_monnaie, "15€")

    def test_pourcentage_round_trip(self):
        _round_trip(lire_pourcentage, "50%")

    def test_ordinal_round_trip(self):
        _round_trip(lire_ordinal, "10e")

    def test_sigle_round_trip(self):
        _round_trip(lire_sigle, "ABC", "ABC")


# ══════════════════════════════════════════════════════════════════════════════
# Cas d'echec
# ══════════════════════════════════════════════════════════════════════════════


class TestEchec:
    """Tests des cas ou la reconnaissance doit echouer."""

    def test_chaine_vide(self):
        assert reconnaitre_ipa("") is None

    def test_ipa_invalide(self):
        assert reconnaitre_ipa("xyz123") is None

    def test_none_safe(self):
        """Verifier que les caracteres non-IPA retournent None."""
        assert reconnaitre_ipa("!!!") is None
