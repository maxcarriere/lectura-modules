"""Tests pour le tokenizer STT-Formules."""

from __future__ import annotations

import pytest

from lectura_formules import (
    lire_nombre,
    lire_sigle,
    lire_date,
    lire_telephone,
    lire_ordinal,
    lire_fraction,
    lire_heure,
    lire_monnaie,
    lire_pourcentage,
    OptionsLecture,
)
from lectura_stt_formules._tokenizer import events_to_token_sequence
from lectura_stt_formules._vocab import (
    SPACE,
    ZERO, UN, DEUX, TROIS, QUATRE, CINQ, SIX, SEPT, HUIT, NEUF,
    DIX, ONZE, DOUZE, TREIZE, QUATORZE, QUINZE, SEIZE, UNE,
    VINGT, TRENTE, QUARANTE, CINQUANTE, SOIXANTE,
    CENT, MILLE, MILLION, MILLIARD,
    ET, VIRGULE, MOINS, PLUS,
    JANVIER, FEVRIER, MARS, AVRIL, MAI, JUIN,
    JUILLET, AOUT, SEPTEMBRE, OCTOBRE, NOVEMBRE, DECEMBRE,
    HEURE, MINUTE, MIDI,
    EURO, DOLLAR, CENTIME, LIVRE,
    POURCENT, POURMILLE,
    PREMIER, IEME, SUR, DEMI, TIERS, QUART,
    S_LETTER, N_LETTER, C_LETTER, F_LETTER,
    token_ids_to_names,
)


class TestNombresSimples:
    """Nombres atomiques et petits nombres."""

    def test_zero(self):
        result = lire_nombre("0")
        tokens = events_to_token_sequence(result)
        assert tokens == [ZERO]

    def test_un(self):
        result = lire_nombre("1")
        tokens = events_to_token_sequence(result)
        assert tokens == [UN]

    def test_dix(self):
        result = lire_nombre("10")
        tokens = events_to_token_sequence(result)
        assert tokens == [DIX]

    def test_seize(self):
        result = lire_nombre("16")
        tokens = events_to_token_sequence(result)
        assert tokens == [SEIZE]

    def test_vingt(self):
        result = lire_nombre("20")
        tokens = events_to_token_sequence(result)
        assert tokens == [VINGT]

    def test_vingt_et_un(self):
        result = lire_nombre("21")
        tokens = events_to_token_sequence(result)
        assert tokens == [VINGT, ET, UN]

    def test_quarante_deux(self):
        result = lire_nombre("42")
        tokens = events_to_token_sequence(result)
        assert tokens == [QUARANTE, DEUX]

    def test_soixante_et_onze(self):
        result = lire_nombre("71")
        tokens = events_to_token_sequence(result)
        assert tokens == [SOIXANTE, ET, ONZE]

    def test_quatre_vingt_dix(self):
        result = lire_nombre("90")
        tokens = events_to_token_sequence(result)
        assert tokens == [QUATRE, VINGT, DIX]

    def test_quatre_vingt_dix_neuf(self):
        result = lire_nombre("99")
        tokens = events_to_token_sequence(result)
        assert tokens == [QUATRE, VINGT, DIX, NEUF]


class TestNombresGrands:
    """Nombres a 3+ chiffres."""

    def test_cent(self):
        result = lire_nombre("100")
        tokens = events_to_token_sequence(result)
        assert tokens == [CENT]

    def test_deux_cents(self):
        result = lire_nombre("200")
        tokens = events_to_token_sequence(result)
        assert tokens == [DEUX, CENT]

    def test_mille(self):
        result = lire_nombre("1000")
        tokens = events_to_token_sequence(result)
        assert tokens == [MILLE]

    def test_1789(self):
        result = lire_nombre("1789")
        tokens = events_to_token_sequence(result)
        assert tokens == [MILLE, SEPT, CENT, QUATRE, VINGT, NEUF]

    def test_deux_mille(self):
        result = lire_nombre("2000")
        tokens = events_to_token_sequence(result)
        assert tokens == [DEUX, MILLE]

    def test_un_million(self):
        result = lire_nombre("1000000")
        tokens = events_to_token_sequence(result)
        assert tokens == [UN, MILLION]


class TestNombresDecimaux:
    """Nombres decimaux avec virgule."""

    def test_trois_virgule_quatorze(self):
        result = lire_nombre("3.14")
        tokens = events_to_token_sequence(result)
        # "trois virgule quatorze" ou "trois virgule un quatre"
        # Depend du mode decimal (m2 par defaut)
        assert VIRGULE in tokens
        assert tokens[0] == TROIS


class TestNombresNegatifs:
    """Nombres negatifs."""

    def test_moins_cinq(self):
        result = lire_nombre("-5")
        tokens = events_to_token_sequence(result)
        assert tokens[0] == MOINS
        assert CINQ in tokens


class TestDates:
    """Dates en format JJ/MM/AAAA."""

    def test_14_juillet_1789(self):
        result = lire_date("14/07/1789")
        tokens = events_to_token_sequence(result)
        # "quatorze <sp> juillet <sp> mille sept cent quatre-vingt-neuf"
        assert QUATORZE in tokens
        assert SPACE in tokens
        assert JUILLET in tokens
        assert MILLE in tokens

    def test_01_janvier_2000(self):
        result = lire_date("01/01/2000")
        tokens = events_to_token_sequence(result)
        # "premier <sp> janvier <sp> deux mille"
        assert JANVIER in tokens
        assert SPACE in tokens


class TestHeures:
    """Heures."""

    def test_14h30(self):
        result = lire_heure("14h30")
        tokens = events_to_token_sequence(result)
        assert QUATORZE in tokens
        assert HEURE in tokens
        assert TRENTE in tokens

    def test_8h(self):
        result = lire_heure("8h")
        tokens = events_to_token_sequence(result)
        assert HUIT in tokens
        assert HEURE in tokens


class TestTelephones:
    """Numeros de telephone."""

    def test_telephone_standard(self):
        result = lire_telephone("05.46.90.20.13")
        tokens = events_to_token_sequence(result)
        # Doit contenir des SPACE entre les groupes
        assert SPACE in tokens
        # 05 → zero cinq
        assert ZERO in tokens
        assert CINQ in tokens
        # 90 → quatre-vingt-dix
        assert QUATRE in tokens
        assert VINGT in tokens
        assert DIX in tokens
        # 13 → treize
        assert TREIZE in tokens


class TestSigles:
    """Sigles epeles lettre par lettre.

    Chaque lettre d'un sigle a un composant different dans les events,
    donc un SPACE est insere entre chaque lettre.
    """

    def test_sncf(self):
        result = lire_sigle("SNCF")
        tokens = events_to_token_sequence(result)
        assert tokens == [S_LETTER, SPACE, N_LETTER, SPACE, C_LETTER, SPACE, F_LETTER]

    def test_two_letters(self):
        result = lire_sigle("AB")
        tokens = events_to_token_sequence(result)
        from lectura_stt_formules._vocab import A_LETTER, B_LETTER
        assert tokens == [A_LETTER, SPACE, B_LETTER]


class TestOrdinaux:
    """Nombres ordinaux."""

    def test_premier(self):
        result = lire_ordinal("1er")
        tokens = events_to_token_sequence(result)
        assert PREMIER in tokens

    def test_deuxieme(self):
        result = lire_ordinal("2e")
        tokens = events_to_token_sequence(result)
        assert DEUX in tokens
        assert IEME in tokens

    def test_quarante_deuxieme(self):
        result = lire_ordinal("42e")
        tokens = events_to_token_sequence(result)
        assert QUARANTE in tokens
        assert DEUX in tokens
        assert IEME in tokens


class TestFractions:
    """Fractions."""

    def test_un_demi(self):
        result = lire_fraction("1/2")
        tokens = events_to_token_sequence(result)
        # Depend du mode fraction (hybride par defaut)
        assert UN in tokens or DEMI in tokens

    def test_trois_quarts(self):
        result = lire_fraction("3/4")
        tokens = events_to_token_sequence(result)
        assert TROIS in tokens

    def test_trois_cinquiemes(self):
        result = lire_fraction("3/5")
        tokens = events_to_token_sequence(result)
        assert TROIS in tokens


class TestMonnaie:
    """Montants en devises."""

    def test_42_euros(self):
        result = lire_monnaie("42€")
        tokens = events_to_token_sequence(result)
        assert QUARANTE in tokens
        assert DEUX in tokens
        assert EURO in tokens

    def test_100_dollars(self):
        result = lire_monnaie("100$")
        tokens = events_to_token_sequence(result)
        assert CENT in tokens
        assert DOLLAR in tokens

    def test_3_livres_20(self):
        result = lire_monnaie("3.20£")
        tokens = events_to_token_sequence(result)
        assert TROIS in tokens
        assert LIVRE in tokens


class TestPourcentages:
    """Pourcentages et pour-mille."""

    def test_42_pourcent(self):
        result = lire_pourcentage("42%")
        tokens = events_to_token_sequence(result)
        assert QUARANTE in tokens
        assert DEUX in tokens
        assert POURCENT in tokens

    def test_5_pourmille(self):
        result = lire_pourcentage("5\u2030")
        tokens = events_to_token_sequence(result)
        assert CINQ in tokens
        assert POURMILLE in tokens


class TestErreurs:
    """Cas d'erreur."""

    def test_ortho_inconnu_raises(self):
        """Un event avec un ortho non reconnu doit lever ValueError."""
        from lectura_formules.lecture_formules import (
            EventFormuleLecture,
            LectureFormuleResult,
        )
        fake_result = LectureFormuleResult(
            display_fr="inconnu",
            phone="foo",
            events=[EventFormuleLecture(ortho="XXXXXX", phone="foo")],
        )
        with pytest.raises(ValueError, match="ortho inconnu"):
            events_to_token_sequence(fake_result)

    def test_empty_events(self):
        """Un resultat sans events doit retourner une liste vide."""
        from lectura_formules.lecture_formules import LectureFormuleResult
        empty_result = LectureFormuleResult(display_fr="", phone="")
        tokens = events_to_token_sequence(empty_result)
        assert tokens == []


class TestEchantillonRepresentatif:
    """Verification que la tokenisation ne leve pas d'erreur
    sur un echantillon large de formules."""

    def _try_tokenize(self, func, text: str) -> bool:
        """Essaie de tokeniser, retourne True si OK."""
        try:
            result = func(text)
            if result and result.events:
                tokens = events_to_token_sequence(result)
                assert len(tokens) > 0
                return True
        except ValueError:
            return False
        return True

    def test_sample_nombres(self):
        ok = 0
        for n in [0, 1, 5, 10, 17, 21, 42, 71, 80, 90, 99, 100, 200,
                   500, 999, 1000, 1789, 2024, 10000, 100000, 1000000]:
            if self._try_tokenize(lire_nombre, str(n)):
                ok += 1
        assert ok >= 18, f"Seulement {ok}/21 nombres tokenises correctement"

    def test_sample_dates(self):
        ok = 0
        for d in ["01/01/2000", "14/07/1789", "25/12/2024", "11/11/1918"]:
            if self._try_tokenize(lire_date, d):
                ok += 1
        assert ok >= 3, f"Seulement {ok}/4 dates tokenisees correctement"

    def test_sample_heures(self):
        ok = 0
        for h in ["8h", "14h30", "23h59", "0h00", "12h"]:
            if self._try_tokenize(lire_heure, h):
                ok += 1
        assert ok >= 4, f"Seulement {ok}/5 heures tokenisees correctement"

    def test_sample_sigles(self):
        ok = 0
        for s in ["SNCF", "TGV", "ADN", "API", "HTML"]:
            if self._try_tokenize(lire_sigle, s):
                ok += 1
        assert ok >= 5, f"Seulement {ok}/5 sigles tokenises correctement"

    def test_sample_pourcentages(self):
        ok = 0
        for p in ["42%", "100%", "3.5%", "50%"]:
            if self._try_tokenize(lire_pourcentage, p):
                ok += 1
        assert ok >= 3, f"Seulement {ok}/4 pourcentages tokenises correctement"
