"""Tests pour la correction des homophones POS-aware dans P2G."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_p2g.posttraitement import corriger_p2g, corriger_phrase_v2


class TestHomophones:
    """Tests de correction est/et, a/à via POS."""

    def test_est_con_donne_et(self):
        assert corriger_p2g("est", pos="CON") == "et"

    def test_et_aux_donne_est(self):
        assert corriger_p2g("et", pos="AUX") == "est"

    def test_et_ver_donne_est(self):
        assert corriger_p2g("et", pos="VER") == "est"

    def test_a_pre_donne_a_accent(self):
        assert corriger_p2g("a", pos="PRE") == "à"

    def test_a_accent_aux_donne_a(self):
        assert corriger_p2g("à", pos="AUX") == "a"

    def test_a_accent_ver_donne_a(self):
        assert corriger_p2g("à", pos="VER") == "a"

    def test_est_ver_inchange(self):
        """'est' avec POS=VER ne doit pas changer (cas normal)."""
        # Pas dans _HOMOPHONES_POS car ("est", "VER") n'y est pas
        assert corriger_p2g("est", pos="VER") == "est"

    def test_et_con_inchange(self):
        """'et' avec POS=CON ne doit pas changer (cas normal)."""
        assert corriger_p2g("et", pos="CON") == "et"

    def test_casse_insensible(self):
        """La comparaison doit être insensible à la casse."""
        assert corriger_p2g("Est", pos="CON") == "et"
        assert corriger_p2g("ET", pos="AUX") == "est"
        assert corriger_p2g("A", pos="PRE") == "à"

    def test_mot_vide(self):
        assert corriger_p2g("", pos="CON") == ""

    def test_sans_morpho(self):
        """Sans morpho, seuls les homophones POS sont corrigés."""
        assert corriger_p2g("est", pos="CON") == "et"
        assert corriger_p2g("chat", pos="NOM") == "chat"


class TestCorrectionsExistantes:
    """Vérifier que les corrections morpho existantes ne régressent pas."""

    def test_pluriel_nom(self):
        result = corriger_p2g(
            "chat", pos="NOM",
            morpho={"Number": "Plur", "Gender": "_", "Person": "_", "VerbForm": "_"},
        )
        assert result == "chats"

    def test_feminin_adj(self):
        result = corriger_p2g(
            "grand", pos="ADJ",
            morpho={"Number": "Sing", "Gender": "Fem", "Person": "_", "VerbForm": "_"},
        )
        assert result == "grande"

    def test_verbe_3pl(self):
        result = corriger_p2g(
            "parle", pos="VER",
            morpho={"Number": "Plur", "Gender": "_", "Person": "3", "VerbForm": "Fin"},
        )
        assert result == "parlent"
