"""Tests pour le module AZERTY (adjacence clavier + scoring)."""

from __future__ import annotations

import pytest

from lectura_correcteur._azerty import (
    AZERTY_ADJACENCE,
    _sont_adjacentes,
    ratio_adjacence_azerty,
)


class TestAzertyAdjacence:
    """Tests de la matrice d'adjacence."""

    def test_symetrie(self):
        """Si a est voisin de b, alors b est voisin de a."""
        for key, voisins in AZERTY_ADJACENCE.items():
            for v in voisins:
                assert key in AZERTY_ADJACENCE[v], (
                    f"{key} est voisin de {v} mais pas l'inverse"
                )

    def test_26_lettres(self):
        """La matrice couvre les 26 lettres de l'alphabet."""
        assert len(AZERTY_ADJACENCE) == 26
        for c in "abcdefghijklmnopqrstuvwxyz":
            assert c in AZERTY_ADJACENCE, f"Lettre manquante: {c}"

    def test_pas_auto_voisin(self):
        """Aucune touche n'est son propre voisin."""
        for key, voisins in AZERTY_ADJACENCE.items():
            assert key not in voisins, f"{key} est son propre voisin"

    def test_a_et_z_adjacentes(self):
        assert _sont_adjacentes("a", "z")
        assert _sont_adjacentes("z", "a")

    def test_a_et_m_non_adjacentes(self):
        assert not _sont_adjacentes("a", "m")

    def test_case_insensitive(self):
        assert _sont_adjacentes("A", "z")
        assert _sont_adjacentes("Z", "A")


class TestRatioAdjacence:
    """Tests du ratio d'adjacence AZERTY."""

    def test_mots_identiques(self):
        """Mots identiques -> 0.5 (neutre)."""
        assert ratio_adjacence_azerty("chat", "chat") == 0.5

    def test_substitution_adjacente(self):
        """Une substitution adjacente -> ratio 1.0."""
        # a->z sont adjacentes sur AZERTY
        ratio = ratio_adjacence_azerty("cat", "czt")
        assert ratio == 1.0

    def test_substitution_non_adjacente(self):
        """Une substitution non adjacente -> ratio 0.0."""
        # a->m ne sont pas adjacentes
        ratio = ratio_adjacence_azerty("cat", "cmt")
        assert ratio == 0.0

    def test_mixte(self):
        """2 substitutions, 1 adjacente 1 non -> ratio 0.5."""
        # position 0: c->v (adjacentes), position 2: t->m (non adjacentes)
        ratio = ratio_adjacence_azerty("cat", "vam")
        assert ratio == pytest.approx(0.5)

    def test_longueurs_differentes(self):
        """Insertion/deletion comptee comme non-adjacente."""
        # "chat" vs "chats" : 1 char excedentaire = 1 sub non-adjacente
        ratio = ratio_adjacence_azerty("chat", "chats")
        assert ratio == 0.0  # 0 adjacentes / 1 diff

    def test_mot_vide(self):
        ratio = ratio_adjacence_azerty("", "abc")
        assert ratio == 0.0

    def test_deux_vides(self):
        ratio = ratio_adjacence_azerty("", "")
        assert ratio == 0.5  # identiques


class TestScoringIntegration:
    """Test d'integration AZERTY dans le scoring."""

    def test_poids_azerty_present(self):
        from lectura_correcteur._scoring import W_AZERTY
        assert W_AZERTY == 0.05

    def test_poids_total_egal_un(self):
        from lectura_correcteur._scoring import (
            W_IDENTITE, W_FREQ, W_EDIT, W_POS,
            W_MORPHO, W_PHONE, W_CTX, W_AZERTY,
        )
        total = W_IDENTITE + W_FREQ + W_EDIT + W_POS + W_MORPHO + W_PHONE + W_CTX + W_AZERTY
        assert total == pytest.approx(1.0)
