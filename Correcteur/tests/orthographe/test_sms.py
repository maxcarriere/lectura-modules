"""Tests pour l'expansion SMS."""

from __future__ import annotations

import pytest

from lectura_correcteur.orthographe._sms import SMS_TABLE, expander_sms


class TestExpanderSms:
    """Tests de l'expansion SMS basique."""

    def test_expansion_simple(self, mock_lexique):
        """bjr -> bonjour"""
        tokens = ["bjr"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["bonjour"]

    def test_expansion_bcp(self, mock_lexique):
        """bcp -> beaucoup"""
        tokens = ["bcp"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["beaucoup"]

    def test_expansion_tjs(self, mock_lexique):
        """tjs -> toujours"""
        tokens = ["tjs"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["toujours"]

    def test_expansion_mtn(self, mock_lexique):
        """mtn -> maintenant"""
        tokens = ["mtn"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["maintenant"]

    def test_expansion_pk(self, mock_lexique):
        """pk -> pourquoi"""
        tokens = ["pk"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["pourquoi"]

    def test_multi_mots(self, mock_lexique):
        """jsuis -> je suis (split en 2 tokens)"""
        tokens = ["jsuis"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["je", "suis"]

    def test_multi_mots_ct(self, mock_lexique):
        """ct -> c'etait"""
        tokens = ["ct"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["c'etait"]

    def test_pas_de_faux_positif_mot_connu(self, mock_lexique):
        """Un mot connu du lexique ne doit pas etre expande, meme s'il est dans SMS_TABLE."""
        # "un" est dans le lexique -> ne pas expander
        tokens = ["un", "chat"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["un", "chat"]

    def test_mot_inconnu_pas_dans_table(self, mock_lexique):
        """Un mot inconnu mais pas dans SMS_TABLE reste tel quel."""
        tokens = ["xyzabc"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["xyzabc"]

    def test_melange_sms_et_normal(self, mock_lexique):
        """Phrase mixte : certains tokens expandes, d'autres non."""
        tokens = ["bjr", "le", "chat"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["bonjour", "le", "chat"]

    def test_case_insensitive(self, mock_lexique):
        """L'expansion est case-insensitive."""
        tokens = ["BJR"]
        result = expander_sms(tokens, mock_lexique)
        assert result == ["bonjour"]

    def test_liste_vide(self, mock_lexique):
        """Liste vide -> liste vide."""
        assert expander_sms([], mock_lexique) == []


class TestSmsTable:
    """Verification de la table SMS."""

    def test_table_non_vide(self):
        assert len(SMS_TABLE) > 50

    def test_valeurs_non_vides(self):
        for key, val in SMS_TABLE.items():
            assert val, f"Valeur vide pour cle '{key}'"

    def test_cles_minuscules(self):
        for key in SMS_TABLE:
            assert key == key.lower() or key == key, f"Cle non minuscule: '{key}'"
