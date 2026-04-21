"""Tests d'intégration : tokeniseur + post-traitement + pipeline complet."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_nlp.tokeniseur import tokeniser, phrase_vers_chars
from lectura_nlp.posttraitement import (
    appliquer_denasalisation,
    appliquer_liaison,
    appliquer_regles_g2p,
    charger_corrections,
    charger_homographes,
    corriger_g2p,
)
from lectura_nlp.utils.ipa import iter_phonemes, est_voyelle, est_consonne
from lectura_nlp.utils.g2p_labels import _CONT, labels_from_alignment, reconstruct_ipa


# ── Tokeniseur ─────────────────────────────────────────────────────

class TestTokeniseur:
    def test_simple(self):
        assert tokeniser("Le chat mange.") == ["Le", "chat", "mange", "."]

    def test_apostrophe(self):
        tokens = tokeniser("L'enfant joue.")
        assert tokens == ["L'", "enfant", "joue", "."]

    def test_aujourdhui(self):
        tokens = tokeniser("C'est aujourd'hui.")
        assert "aujourd'hui" in tokens

    def test_empty(self):
        assert tokeniser("") == []
        assert tokeniser("   ") == []

    def test_punctuation(self):
        tokens = tokeniser("Bonjour, comment allez-vous ?")
        assert "," in tokens
        assert "?" in tokens

    def test_phrase_vers_chars(self):
        tokens = ["le", "chat"]
        chars, word_ids = phrase_vers_chars(tokens)
        assert chars[0] == "<BOS>"
        assert chars[-1] == "<EOS>"
        assert "<SEP>" in chars
        assert word_ids[0] == -1  # <BOS>
        assert word_ids[-1] == -1  # <EOS>


# ── IPA ────────────────────────────────────────────────────────────

class TestIPA:
    def test_iter_phonemes_simple(self):
        assert iter_phonemes("ʃa") == ["ʃ", "a"]

    def test_iter_phonemes_nasal(self):
        # ɑ̃ = ɑ + combining tilde
        ph = iter_phonemes("ɑ̃")
        assert len(ph) == 1

    def test_iter_phonemes_empty(self):
        assert iter_phonemes("") == []

    def test_est_voyelle(self):
        assert est_voyelle("a")
        assert est_voyelle("ɑ̃")
        assert not est_voyelle("p")

    def test_est_consonne(self):
        assert est_consonne("p")
        assert est_consonne("ʁ")
        assert not est_consonne("a")


# ── G2P Labels ─────────────────────────────────────────────────────

class TestG2PLabels:
    def test_labels_from_alignment(self):
        # "chat" → ʃ (0-1) + a (2-3) → _CONT system
        labels = labels_from_alignment("chat", ["ʃ", "a"], [(0, 2), (2, 4)])
        assert labels[0] == "ʃ"
        assert labels[1] == _CONT
        assert labels[2] == "a"
        assert labels[3] == _CONT

    def test_reconstruct_ipa(self):
        labels = ["ʃ", _CONT, "a", _CONT]
        assert reconstruct_ipa(labels) == "ʃa"

    def test_roundtrip(self):
        labels = ["b", _CONT, "ɔ̃", _CONT, _CONT, "ʒ", _CONT, "u", "ʁ"]
        ipa = reconstruct_ipa(labels)
        assert "ɔ̃" in ipa or "b" in ipa  # sanity check


# ── Post-traitement ────────────────────────────────────────────────

class TestPosttraitement:
    def test_denasalisation(self):
        result = appliquer_denasalisation("bɔ̃", "ɔ̃>ɔ")
        assert result == "bɔ"

    def test_denasalisation_empty(self):
        assert appliquer_denasalisation("bɔ̃", "") == "bɔ̃"

    def test_liaison_simple(self):
        tokens = ["les", "enfants"]
        phones = ["le", "ɑ̃fɑ̃"]
        liaisons = ["Lz", "none"]
        result = appliquer_liaison(tokens, phones, liaisons)
        assert result[0] == "lez"
        assert result[1] == "ɑ̃fɑ̃"

    def test_liaison_no_vowel(self):
        """Pas de liaison si le mot suivant ne commence pas par voyelle."""
        tokens = ["les", "chats"]
        phones = ["le", "ʃa"]
        liaisons = ["Lz", "none"]
        result = appliquer_liaison(tokens, phones, liaisons)
        assert result[0] == "le"  # No liaison before consonant

    def test_liaison_with_denas(self):
        tokens = ["bon", "ami"]
        phones = ["bɔ̃", "ami"]
        liaisons = ["Ln", "none"]
        denas = ["ɔ̃>ɔ", ""]
        result = appliquer_liaison(tokens, phones, liaisons, denas)
        assert result[0] == "bɔn"

    def test_corriger_g2p_with_table(self):
        corrections_path = Path(__file__).resolve().parent.parent / "src" / "lectura_nlp" / "data" / "g2p_corrections_unifie.json"
        charger_corrections(corrections_path)
        assert corriger_g2p("monsieur", "mɔ̃sjø") == "məsjø"
        assert corriger_g2p("chat", "ʃa") == "ʃa"  # no correction

    def test_regles_ex_consonne(self):
        assert appliquer_regles_g2p("expression", "ɛkpʁɛsjɔ̃") == "ɛkspʁɛsjɔ̃"
        assert appliquer_regles_g2p("extrême", "ɛktʁɛm") == "ɛkstʁɛm"
        assert appliquer_regles_g2p("expression", "ɛkspʁɛsjɔ̃") == "ɛkspʁɛsjɔ̃"  # already correct

    def test_regles_ex_voyelle(self):
        assert appliquer_regles_g2p("exemple", "ɛkɑ̃pl") == "ɛɡzɑ̃pl"
        assert appliquer_regles_g2p("existence", "ezistɑ̃s") == "ɛɡzistɑ̃s"
        assert appliquer_regles_g2p("examen", "ɛzamɑ̃") == "ɛɡzamɑ̃"

    def test_regles_yod(self):
        assert appliquer_regles_g2p("oublier", "ublje") == "ublije"
        assert appliquer_regles_g2p("crier", "kʁje") == "kʁije"
        assert appliquer_regles_g2p("maison", "mɛzɔ̃") == "mɛzɔ̃"  # no change


# ── Homographes (POS-aware) ──────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "lectura_nlp" / "data"


class TestHomographes:
    @pytest.fixture(autouse=True)
    def _load(self):
        """Charge corrections + homographes avant chaque test."""
        charger_corrections(_DATA_DIR / "g2p_corrections_unifie.json")
        charger_homographes(_DATA_DIR / "homographes.json")

    def test_homographes_est(self):
        """'est' AUX → /ɛ/, NOM → /ɛst/."""
        assert corriger_g2p("est", "ɛ", pos="AUX") == "ɛ"
        assert corriger_g2p("est", "ɛ", pos="NOM") == "ɛst"
        assert corriger_g2p("est", "ɛ", pos="ADJ") == "ɛst"

    def test_homographes_plus(self):
        """'plus' ADV → /ply/, NOM → /plys/."""
        assert corriger_g2p("plus", "ply", pos="ADV") == "ply"
        assert corriger_g2p("plus", "ply", pos="NOM") == "plys"

    def test_homographes_as(self):
        """'as' AUX → /a/, NOM → /as/."""
        assert corriger_g2p("as", "a", pos="AUX") == "a"
        assert corriger_g2p("as", "a", pos="NOM") == "as"

    def test_corriger_g2p_sans_pos(self):
        """Rétrocompatibilité : pos=None ne casse rien."""
        # Sans POS, un mot qui n'est pas dans les corrections passe les règles
        result = corriger_g2p("chat", "ʃa")
        assert result == "ʃa"
        # Un mot dans les corrections plate reste corrigé
        result = corriger_g2p("monsieur", "mɔ̃sjø")
        assert result == "məsjø"

    def test_homographes_priorite(self):
        """Homographe avec POS est prioritaire sur correction plate."""
        # "est" apparaît dans homographes — la correction plate ne doit
        # pas écraser le résultat POS-aware
        assert corriger_g2p("est", "ɛ", pos="NOM") == "ɛst"
        assert corriger_g2p("est", "ɛ", pos="AUX") == "ɛ"

    def test_homographes_pos_inconnu(self):
        """POS inconnu pour un homographe → fallback corrections/règles."""
        # "est" avec un POS absent de la table → pas de match homographe
        result = corriger_g2p("est", "ɛ", pos="PONCT")
        # Doit passer aux règles (pas de correction plate car retiré)
        assert result == "ɛ"
