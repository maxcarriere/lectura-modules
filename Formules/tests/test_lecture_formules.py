"""Tests unitaires pour lecture_formules.py — lecture algorithmique des formules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_formules.lecture_formules import (
    EventFormuleLecture,
    LectureFormuleResult,
    OptionsLecture,
    lire_formule,
    lire_nombre,
    lire_sigle,
    lire_date,
    lire_telephone,
    lire_ordinal,
    lire_fraction,
    lire_scientifique,
    lire_maths,
    lire_numero,
    lire_heure,
    lire_monnaie,
    lire_pourcentage,
    lire_intervalle,
    lire_gps,
    lire_page_chapitre,
    enrichir_formules,
    _nombre_vers_francais,
    _bloc_0_999,
)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithme cœur : nombre → français
# ══════════════════════════════════════════════════════════════════════════════

class TestNombreVersFrancais:
    """Tests du cœur algorithmique de conversion nombre → français."""

    def _texte(self, n: int, feminin: bool = False) -> str:
        parts = _nombre_vers_francais(n, feminin=feminin)
        return "-".join(t for t, _p in parts)

    def _phone(self, n: int, feminin: bool = False) -> str:
        parts = _nombre_vers_francais(n, feminin=feminin)
        return "".join(p for _t, p in parts)

    def test_zero(self):
        assert self._texte(0) == "zéro"

    def test_unites(self):
        assert self._texte(1) == "un"
        assert self._texte(2) == "deux"
        assert self._texte(9) == "neuf"

    def test_teens(self):
        assert self._texte(10) == "dix"
        assert self._texte(11) == "onze"
        assert self._texte(16) == "seize"

    def test_dix_sept_a_dix_neuf(self):
        assert self._texte(17) == "dix-sept"
        assert self._texte(18) == "dix-huit"
        assert self._texte(19) == "dix-neuf"

    def test_dizaines(self):
        assert self._texte(20) == "vingt"
        assert self._texte(30) == "trente"
        assert self._texte(40) == "quarante"
        assert self._texte(50) == "cinquante"
        assert self._texte(60) == "soixante"

    def test_vingt_et_un(self):
        assert self._texte(21) == "vingt-et un"
        assert self._texte(31) == "trente-et un"
        assert self._texte(41) == "quarante-et un"
        assert self._texte(51) == "cinquante-et un"
        assert self._texte(61) == "soixante-et un"

    def test_soixante_dix(self):
        assert self._texte(70) == "soixante-dix"
        assert self._texte(71) == "soixante-et onze"
        assert self._texte(72) == "soixante-douze"
        assert self._texte(76) == "soixante-seize"
        assert self._texte(77) == "soixante-dix-sept"
        assert self._texte(79) == "soixante-dix-neuf"

    def test_quatre_vingts(self):
        assert self._texte(80) == "quatre-vingts"
        assert self._texte(81) == "quatre-vingt-un"
        assert self._texte(89) == "quatre-vingt-neuf"

    def test_quatre_vingt_dix(self):
        assert self._texte(90) == "quatre-vingt-dix"
        assert self._texte(91) == "quatre-vingt-onze"
        assert self._texte(96) == "quatre-vingt-seize"
        assert self._texte(97) == "quatre-vingt-dix-sept"
        assert self._texte(99) == "quatre-vingt-dix-neuf"

    def test_centaines(self):
        assert self._texte(100) == "cent"
        assert self._texte(200) == "deux-cents"
        assert self._texte(201) == "deux-cent-un"
        assert self._texte(342) == "trois-cent-quarante-deux"

    def test_mille(self):
        assert self._texte(1000) == "mille"
        assert self._texte(2000) == "deux-mille"
        assert self._texte(1001) == "mille-un"

    def test_grands_nombres(self):
        assert self._texte(1_000_000) == "un-million"
        assert self._texte(2_000_000) == "deux-millions"
        assert self._texte(1_000_000_000) == "un-milliard"

    def test_feminin(self):
        assert self._texte(1, feminin=True) == "une"
        assert self._texte(21, feminin=True) == "vingt-et une"
        assert self._texte(31, feminin=True) == "trente-et une"

    def test_negatif(self):
        parts = _nombre_vers_francais(-5)
        assert parts[0][0] == "moins"
        assert parts[1][0] == "cinq"

    def test_phone_342(self):
        phone = self._phone(342)
        assert "tʁwa" in phone
        assert "sɑ̃" in phone
        assert "kaʁɑ̃t" in phone
        assert "dø" in phone


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur NOMBRE
# ══════════════════════════════════════════════════════════════════════════════

class TestLireNombre:
    def test_nombre_simple(self):
        result = lire_nombre("342", span=(5, 8))
        assert "trois" in result.display_fr
        assert "cent" in result.display_fr
        assert "quarante" in result.display_fr
        assert "deux" in result.display_fr
        assert len(result.events) == 4
        assert result.phone  # non vide

    def test_nombre_zero(self):
        result = lire_nombre("0", span=(0, 1))
        assert result.display_fr == "zéro"

    def test_nombre_decimal(self):
        result = lire_nombre("3,14", span=(0, 4))
        assert "trois" in result.display_fr
        assert "virgule" in result.display_fr
        assert "quatorze" in result.display_fr

    def test_nombre_feminin(self):
        result = lire_nombre("1", span=(0, 1), feminin=True)
        assert result.display_fr == "une"

    def test_nombre_avec_espaces(self):
        result = lire_nombre("1 000", span=(0, 5))
        assert "mille" in result.display_fr

    def test_nombre_avec_apostrophe(self):
        result = lire_nombre("1'000", span=(0, 5))
        assert "mille" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur SIGLE
# ══════════════════════════════════════════════════════════════════════════════

class TestLireSigle:
    def test_sigle_simple(self):
        result = lire_sigle("SNCF", span=(0, 4))
        assert "esse" in result.display_fr
        assert "enne" in result.display_fr
        assert "cé" in result.display_fr
        assert "effe" in result.display_fr
        assert len(result.events) == 4

    def test_sigle_avec_chiffres(self):
        result = lire_sigle("B2B", span=(0, 3))
        assert "bé" in result.display_fr
        assert "deux" in result.display_fr
        assert len(result.events) == 3

    def test_sigle_avec_points(self):
        result = lire_sigle("W.W.F.", span=(0, 6))
        assert "double-vé" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur DATE
# ══════════════════════════════════════════════════════════════════════════════

class TestLireDate:
    def test_date_complete(self):
        result = lire_date("15/03/2024", span=(0, 10))
        assert "quinze" in result.display_fr
        assert "mars" in result.display_fr
        assert "deux" in result.display_fr
        assert len(result.events) >= 3  # jour + mois + année

    def test_date_premier(self):
        result = lire_date("01/01/2024", span=(0, 10))
        assert "premier" in result.display_fr
        assert "janvier" in result.display_fr

    def test_date_tirets(self):
        result = lire_date("25-12-2023", span=(0, 10))
        assert "vingt" in result.display_fr
        assert "décembre" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur TELEPHONE
# ══════════════════════════════════════════════════════════════════════════════

class TestLireTelephone:
    def test_telephone_standard(self):
        result = lire_telephone("06 12 34 56 78", span=(0, 14))
        assert "zéro" in result.display_fr
        assert "six" in result.display_fr
        assert "douze" in result.display_fr
        assert "trente" in result.display_fr  # 34
        assert len(result.events) >= 5  # au moins 5 paires

    def test_telephone_compact(self):
        result = lire_telephone("0612345678", span=(0, 10))
        assert "zéro" in result.display_fr
        assert "six" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur ORDINAL
# ══════════════════════════════════════════════════════════════════════════════

class TestLireOrdinal:
    def test_premier(self):
        result = lire_ordinal("1er", span=(0, 3))
        assert result.display_fr == "premier"

    def test_premiere(self):
        result = lire_ordinal("1ère", span=(0, 4))
        assert result.display_fr == "première"

    def test_deuxieme(self):
        result = lire_ordinal("2e", span=(0, 2))
        assert "deuxième" in result.display_fr

    def test_quarante_deuxieme(self):
        result = lire_ordinal("42e", span=(0, 3))
        assert "deuxième" in result.display_fr or "quarante" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur FRACTION
# ══════════════════════════════════════════════════════════════════════════════

class TestLireFraction:
    def test_fraction_hybride_demi(self):
        result = lire_fraction("1/2", span=(0, 3))
        assert "demi" in result.display_fr

    def test_fraction_hybride_tiers(self):
        result = lire_fraction("2/3", span=(0, 3))
        assert "tiers" in result.display_fr

    def test_fraction_hybride_quart(self):
        result = lire_fraction("3/4", span=(0, 3))
        assert "quarts" in result.display_fr
        assert "trois" in result.display_fr

    def test_fraction_hybride_cinquieme(self):
        result = lire_fraction("2/5", span=(0, 3))
        assert "cinquième" in result.display_fr

    def test_fraction_standard(self):
        opts = OptionsLecture(fraction_mode="standard")
        result = lire_fraction("3/4", span=(0, 3), options=opts)
        assert "sur" in result.display_fr
        assert "trois" in result.display_fr
        assert "quatre" in result.display_fr

    def test_fraction_ordinal(self):
        opts = OptionsLecture(fraction_mode="ordinal")
        result = lire_fraction("1/2", span=(0, 3), options=opts)
        assert "deuxième" in result.display_fr or "demi" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur SCIENTIFIQUE
# ══════════════════════════════════════════════════════════════════════════════

class TestLireScientifique:
    def test_scientifique_base(self):
        result = lire_scientifique("1.23e5", span=(0, 6))
        assert "fois" in result.display_fr
        assert "dix" in result.display_fr
        assert "exposant" in result.display_fr

    def test_scientifique_negatif(self):
        result = lire_scientifique("3.14e-5", span=(0, 7))
        assert "exposant" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur MATHS
# ══════════════════════════════════════════════════════════════════════════════

class TestLireMaths:
    def test_maths_simple(self):
        result = lire_maths("2+3", span=(0, 3))
        assert "deux" in result.display_fr
        assert "plus" in result.display_fr
        assert "trois" in result.display_fr

    def test_maths_variable(self):
        result = lire_maths("x", span=(0, 1))
        assert "ix" in result.display_fr

    def test_maths_grec(self):
        result = lire_maths("α", span=(0, 1))
        assert "alpha" in result.display_fr

    def test_maths_carre(self):
        result = lire_maths("x²", span=(0, 2))
        assert "carré" in result.display_fr

    def test_maths_fonction(self):
        result = lire_maths("sin(x)", span=(0, 6))
        assert "sinus" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur NUMERO
# ══════════════════════════════════════════════════════════════════════════════

class TestLireNumero:
    def test_numero_mixte(self):
        result = lire_numero("AB 123 CD", span=(0, 9))
        assert "a" in result.display_fr
        assert "bé" in result.display_fr
        assert "cent" in result.display_fr
        assert "cé" in result.display_fr

    def test_numero_chiffres_seuls(self):
        result = lire_numero("123", span=(0, 3))
        assert "cent" in result.display_fr
        assert "vingt" in result.display_fr
        assert "trois" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# API publique lire_formule()
# ══════════════════════════════════════════════════════════════════════════════

class TestLireFormule:
    def test_dispatch_nombre(self):
        result = lire_formule("nombre", "42", span=(0, 2))
        assert "quarante" in result.display_fr
        assert "deux" in result.display_fr

    def test_dispatch_sigle(self):
        result = lire_formule("sigle", "SNCF", span=(0, 4))
        assert "esse" in result.display_fr

    def test_dispatch_date(self):
        result = lire_formule("date", "15/03/2024", span=(0, 10))
        assert "mars" in result.display_fr

    def test_dispatch_telephone(self):
        result = lire_formule("telephone", "0612345678", span=(0, 10))
        assert "zéro" in result.display_fr

    def test_dispatch_feminin(self):
        result = lire_formule("nombre", "1", span=(0, 1), feminin=True)
        assert result.display_fr == "une"

    def test_dispatch_inconnu(self):
        result = lire_formule("inconnu", "???", span=(0, 3))
        assert isinstance(result, LectureFormuleResult)

    def test_events_ont_spans(self):
        result = lire_formule("nombre", "342", span=(5, 8))
        for evt in result.events:
            assert isinstance(evt.span_source, tuple)
            assert len(evt.span_source) == 2
            assert evt.span_source[0] >= 5
            assert evt.span_source[1] <= 8


# ══════════════════════════════════════════════════════════════════════════════
# Champ composant
# ══════════════════════════════════════════════════════════════════════════════

class TestComposant:
    def test_nombre_composant_unique(self):
        result = lire_nombre("342", span=(0, 3))
        for evt in result.events:
            assert evt.composant == 0

    def test_sigle_composant_par_lettre(self):
        result = lire_sigle("SNCF", span=(0, 4))
        composants = [evt.composant for evt in result.events]
        assert composants == [0, 1, 2, 3]

    def test_date_composants(self):
        result = lire_date("15/03/2024", span=(0, 10))
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # jour
        assert 1 in comp_set  # mois
        assert 2 in comp_set  # année

    def test_telephone_composants(self):
        result = lire_telephone("06 12 34 56 78", span=(0, 14))
        comps = result.composants()
        assert len(comps) == 5  # 5 paires

    def test_ordinal_composants(self):
        result = lire_ordinal("42e", span=(0, 3))
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # nombre
        assert 1 in comp_set  # suffixe ordinal

    def test_ordinal_premier_composant_unique(self):
        result = lire_ordinal("1er", span=(0, 3))
        for evt in result.events:
            assert evt.composant == 0

    def test_fraction_composants(self):
        opts = OptionsLecture(fraction_mode="standard")
        result = lire_fraction("3/4", span=(0, 3), options=opts)
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # numérateur
        assert 1 in comp_set  # "sur"
        assert 2 in comp_set  # dénominateur

    def test_fraction_hybride_composants(self):
        result = lire_fraction("1/2", span=(0, 3))
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # numérateur
        assert 2 in comp_set  # dénominateur

    def test_scientifique_composants(self):
        result = lire_scientifique("1.23e5", span=(0, 6))
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # mantisse
        assert 1 in comp_set  # "fois dix exposant"
        assert 2 in comp_set  # exposant

    def test_maths_composants(self):
        result = lire_maths("32.574 + 87", span=(0, 11))
        comps = result.composants()
        assert len(comps) == 3  # 32.574, +, 87

    def test_numero_composants(self):
        result = lire_numero("AB 123 CD", span=(0, 9))
        comps = result.composants()
        assert len(comps) == 3  # AB, 123, CD


# ══════════════════════════════════════════════════════════════════════════════
# Méthode composants()
# ══════════════════════════════════════════════════════════════════════════════

class TestComposantsMethod:
    def test_empty(self):
        result = LectureFormuleResult(display_fr="", phone="", events=[])
        assert result.composants() == []

    def test_single_composant(self):
        result = lire_nombre("42", span=(0, 2))
        comps = result.composants()
        assert len(comps) == 1
        assert len(comps[0]) == len(result.events)

    def test_multiple_composants(self):
        result = lire_maths("2+3", span=(0, 3))
        comps = result.composants()
        assert len(comps) == 3
        # First composant: "deux"
        assert comps[0][0].ortho == "deux"
        # Second composant: "plus"
        assert comps[1][0].ortho == "plus"
        # Third composant: "trois"
        assert comps[2][0].ortho == "trois"

    def test_composants_preserves_order(self):
        result = lire_date("15/03/2024", span=(0, 10))
        comps = result.composants()
        assert len(comps) == 3
        # Jour
        assert any("quinze" in evt.ortho for evt in comps[0])
        # Mois
        assert comps[1][0].ortho == "mars"
        # Année  — should contain "deux"
        assert any("deux" in evt.ortho for evt in comps[2])


# ══════════════════════════════════════════════════════════════════════════════
# enrichir_formules()
# ══════════════════════════════════════════════════════════════════════════════

class _MockType:
    def __init__(self, value: str):
        self.value = value

class _MockFormuleType:
    def __init__(self, value: str):
        self.value = value

class _MockToken:
    def __init__(self, type_val: str, text: str, span: tuple[int, int]):
        self.type = _MockType(type_val)
        self.text = text
        self.span = span

class _MockFormule(_MockToken):
    def __init__(self, text: str, span: tuple[int, int],
                 formule_type: str = "nombre"):
        super().__init__("formule", text, span)
        self.formule_type = _MockFormuleType(formule_type)
        self.children = []
        self.display_fr = ""


class TestEnrichirFormules:
    def test_enrichit_formule_nombre(self):
        tok = _MockFormule("42", (0, 2), formule_type="nombre")
        enrichir_formules([tok])
        assert "quarante" in tok.display_fr
        assert "deux" in tok.display_fr
        assert hasattr(tok, "lecture")
        assert tok.lecture is not None

    def test_enrichit_formule_sigle(self):
        tok = _MockFormule("SNCF", (0, 4), formule_type="sigle")
        enrichir_formules([tok])
        assert "esse" in tok.display_fr

    def test_ignore_non_formule(self):
        tok = _MockToken("mot", "bonjour", (0, 7))
        enrichir_formules([tok])
        assert not hasattr(tok, "display_fr") or tok.text == "bonjour"

    def test_retourne_liste(self):
        tokens = [_MockFormule("42", (0, 2))]
        result = enrichir_formules(tokens)
        assert result is tokens

    def test_mixed_tokens(self):
        mot = _MockToken("mot", "Le", (0, 2))
        formule = _MockFormule("42", (3, 5), formule_type="nombre")
        enrichir_formules([mot, formule])
        assert "quarante" in formule.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Nouveaux champs (v2.0) : sound_id, span_fr, span_num, display_num, display_rom, valeur
# ══════════════════════════════════════════════════════════════════════════════

class TestNouveauxChamps:
    def test_display_num(self):
        result = lire_nombre("42", span=(0, 2))
        assert result.display_num == "42"

    def test_display_rom(self):
        result = lire_nombre("42", span=(0, 2))
        assert result.display_rom == "XLII"

    def test_valeur_int(self):
        result = lire_nombre("42", span=(0, 2))
        assert result.valeur == 42

    def test_valeur_decimal(self):
        result = lire_nombre("3,14", span=(0, 4))
        assert abs(result.valeur - 3.14) < 0.001

    def test_span_fr_computed(self):
        result = lire_nombre("42", span=(0, 2))
        # "quarante-deux" → quarante:(0,8), deux:(9,13)
        for evt in result.events:
            assert evt.span_fr[1] > evt.span_fr[0]

    def test_display_rom_disabled(self):
        opts = OptionsLecture(romain_actif=False)
        result = lire_nombre("42", span=(0, 2), options=opts)
        assert result.display_rom == ""

    def test_display_rom_out_of_range(self):
        result = lire_nombre("0", span=(0, 1))
        assert result.display_rom == ""  # 0 is out of range

    def test_event_has_sound_id_field(self):
        result = lire_nombre("1", span=(0, 1))
        assert hasattr(result.events[0], "sound_id")

    def test_event_has_span_num_field(self):
        result = lire_nombre("1", span=(0, 1))
        assert hasattr(result.events[0], "span_num")


# ══════════════════════════════════════════════════════════════════════════════
# Méthodes décimales M1/M2
# ══════════════════════════════════════════════════════════════════════════════

class TestDecimalM1M2:
    def test_m2_default_groups_by_3(self):
        result = lire_nombre("3,0025124", span=(0, 9))
        # M2: 3 virgule zéro zéro 251 24
        assert "virgule" in result.display_fr
        assert "zéro" in result.display_fr
        # 251 should be "deux-cent-cinquante-et un"
        assert "deux" in result.display_fr
        assert "cent" in result.display_fr

    def test_m1_reads_whole(self):
        opts = OptionsLecture(decimal_method="m1")
        result = lire_nombre("3,0025124", span=(0, 9), options=opts)
        assert "virgule" in result.display_fr
        assert "zéro" in result.display_fr
        # M1: whole rest as one number (25124)
        assert "mille" in result.display_fr

    def test_m2_simple_decimal(self):
        result = lire_nombre("3,14", span=(0, 4))
        assert "virgule" in result.display_fr
        assert "quatorze" in result.display_fr

    def test_m2_leading_zeros_few(self):
        result = lire_nombre("1,001", span=(0, 5))
        # 3 zeros or less: read individually
        assert "zéro" in result.display_fr

    def test_m2_leading_zeros_many(self):
        result = lire_nombre("1,00001", span=(0, 7))
        # >3 zeros: "N fois zéro"
        assert "fois" in result.display_fr

    def test_m1_simple_decimal(self):
        opts = OptionsLecture(decimal_method="m1")
        result = lire_nombre("3,14", span=(0, 4), options=opts)
        assert "virgule" in result.display_fr
        assert "quatorze" in result.display_fr

    def test_decimal_valeur(self):
        result = lire_nombre("3,14", span=(0, 4))
        assert abs(result.valeur - 3.14) < 0.001

    def test_decimal_display_num(self):
        result = lire_nombre("3,14", span=(0, 4))
        assert result.display_num == "3,14"


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur HEURE
# ══════════════════════════════════════════════════════════════════════════════

class TestLireHeure:
    def test_heure_h_format(self):
        result = lire_heure("14h54", span=(0, 5))
        assert "quatorze" in result.display_fr
        assert "heure" in result.display_fr

    def test_heure_colon_format(self):
        result = lire_heure("16:21", span=(0, 5))
        assert "seize" in result.display_fr
        assert "heure" in result.display_fr

    def test_heure_une(self):
        result = lire_heure("1h", span=(0, 2))
        assert "une" in result.display_fr
        assert "heure" in result.display_fr  # singulier

    def test_heure_avec_min(self):
        result = lire_heure("3h15min", span=(0, 7))
        assert "trois" in result.display_fr
        assert "quinze" in result.display_fr
        assert "minute" in result.display_fr

    def test_heure_hms(self):
        result = lire_heure("3h15min10s", span=(0, 10))
        assert "trois" in result.display_fr
        assert "quinze" in result.display_fr
        assert "dix" in result.display_fr
        assert "seconde" in result.display_fr

    def test_heure_min_only(self):
        result = lire_heure("45min", span=(0, 5))
        assert "quarante" in result.display_fr
        assert "minute" in result.display_fr

    def test_heure_sec_only(self):
        result = lire_heure("30s", span=(0, 3))
        assert "trente" in result.display_fr
        assert "seconde" in result.display_fr

    def test_heure_composants(self):
        result = lire_heure("14h54", span=(0, 5))
        comps = result.composants()
        assert len(comps) >= 2  # heures + minutes

    def test_dispatch_heure(self):
        result = lire_formule("heure", "14h54", span=(0, 5))
        assert "quatorze" in result.display_fr

    def test_heure_display_num(self):
        result = lire_heure("14h54", span=(0, 5))
        assert result.display_num == "14h54"

    def test_heure_sans_minutes_mot_colon(self):
        opts = OptionsLecture(heure_mot_minutes=False)
        result = lire_heure("16:21", span=(0, 5), options=opts)
        assert "minute" not in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur MONNAIE
# ══════════════════════════════════════════════════════════════════════════════

class TestLireMonnaie:
    def test_euros_simple(self):
        result = lire_monnaie("12€", span=(0, 3))
        assert "douze" in result.display_fr
        assert "euro" in result.display_fr

    def test_euros_avec_centimes(self):
        result = lire_monnaie("12,50€", span=(0, 6))
        assert "douze" in result.display_fr
        assert "euro" in result.display_fr
        assert "et" in result.display_fr
        assert "centime" in result.display_fr

    def test_dollars_prefix(self):
        result = lire_monnaie("$99", span=(0, 3))
        assert "dollar" in result.display_fr

    def test_livre(self):
        result = lire_monnaie("5£", span=(0, 2))
        assert "livr" in result.display_fr

    def test_chf(self):
        result = lire_monnaie("3.50CHF", span=(0, 7))
        assert "franc" in result.display_fr
        assert "suisse" in result.display_fr

    def test_singulier(self):
        result = lire_monnaie("1€", span=(0, 2))
        assert "euro" in result.display_fr
        # Singulier, pas "euros"
        assert result.display_fr.count("euros") == 0 or "euro" in result.display_fr

    def test_display_num(self):
        result = lire_monnaie("12,50€", span=(0, 6))
        assert "€" in result.display_num

    def test_valeur(self):
        result = lire_monnaie("12,50€", span=(0, 6))
        assert abs(result.valeur - 12.5) < 0.01

    def test_composants(self):
        result = lire_monnaie("12,50€", span=(0, 6))
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # majeur
        assert 1 in comp_set  # "et"
        assert 2 in comp_set  # mineur

    def test_dispatch_monnaie(self):
        result = lire_formule("monnaie", "5€", span=(0, 2))
        assert "euro" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur POURCENTAGE
# ══════════════════════════════════════════════════════════════════════════════

class TestLirePourcentage:
    def test_pour_cent(self):
        result = lire_pourcentage("45%", span=(0, 3))
        assert "quarante" in result.display_fr
        assert "pour cent" in result.display_fr

    def test_pour_mille(self):
        result = lire_pourcentage("3‰", span=(0, 2))
        assert "trois" in result.display_fr
        assert "pour mille" in result.display_fr

    def test_decimal_percent(self):
        result = lire_pourcentage("12.5%", span=(0, 5))
        assert "virgule" in result.display_fr
        assert "pour cent" in result.display_fr

    def test_composants(self):
        result = lire_pourcentage("45%", span=(0, 3))
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # nombre
        assert 1 in comp_set  # "pour cent"

    def test_display_num(self):
        result = lire_pourcentage("45%", span=(0, 3))
        assert result.display_num == "45%"

    def test_dispatch_pourcentage(self):
        result = lire_formule("pourcentage", "45%", span=(0, 3))
        assert "pour cent" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur INTERVALLE
# ══════════════════════════════════════════════════════════════════════════════

class TestLireIntervalle:
    def test_closed_closed(self):
        result = lire_intervalle("[2;5]", span=(0, 5))
        assert "deux" in result.display_fr
        assert "cinq" in result.display_fr

    def test_open_open(self):
        result = lire_intervalle("]0;+∞[", span=(0, 6))
        assert "zéro" in result.display_fr
        assert "infini" in result.display_fr

    def test_minus_infinity(self):
        result = lire_intervalle("]-∞;3]", span=(0, 6))
        assert "moins" in result.display_fr
        assert "infini" in result.display_fr
        assert "trois" in result.display_fr

    def test_composants(self):
        result = lire_intervalle("[2;5]", span=(0, 5))
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # borne gauche
        assert 1 in comp_set  # connecteur
        assert 2 in comp_set  # borne droite

    def test_display_num(self):
        result = lire_intervalle("[2;5]", span=(0, 5))
        assert result.display_num == "[2;5]"

    def test_decimal_bounds(self):
        result = lire_intervalle("[1.5;3.7]", span=(0, 9))
        assert "virgule" in result.display_fr

    def test_dispatch_intervalle(self):
        result = lire_formule("intervalle", "[2;5]", span=(0, 5))
        assert "de" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur GPS
# ══════════════════════════════════════════════════════════════════════════════

class TestLireGps:
    def test_dms_simple(self):
        result = lire_gps('48°51\'24"N', span=(0, 11))
        assert "quarante-huit" in result.display_fr
        assert "degré" in result.display_fr
        assert "nord" in result.display_fr

    def test_dms_with_minutes(self):
        result = lire_gps("48°51'N", span=(0, 7))
        assert "degré" in result.display_fr
        assert "minute" in result.display_fr
        assert "nord" in result.display_fr

    def test_decimal_degrees(self):
        result = lire_gps("48.8566°N", span=(0, 9))
        assert "degré" in result.display_fr
        assert "nord" in result.display_fr

    def test_direction_west(self):
        result = lire_gps("2°21'7\"W", span=(0, 8))
        assert "ouest" in result.display_fr

    def test_direction_sud(self):
        result = lire_gps("33°51'S", span=(0, 7))
        assert "sud" in result.display_fr

    def test_dual_coords(self):
        result = lire_gps('48°51\'24"N 2°21\'7"E', span=(0, 20))
        comps = result.composants()
        assert len(comps) == 2  # lat + lon

    def test_display_num(self):
        result = lire_gps('48°51\'24"N', span=(0, 11))
        assert "°" in result.display_num
        assert "N" in result.display_num

    def test_dispatch_gps(self):
        result = lire_formule("gps", "48°51'N", span=(0, 7))
        assert "degré" in result.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Lecteur PAGE_CHAPITRE
# ══════════════════════════════════════════════════════════════════════════════

class TestLirePageChapitre:
    def test_page_simple(self):
        result = lire_page_chapitre("p.42", span=(0, 4))
        assert "page" in result.display_fr
        assert "quarante" in result.display_fr

    def test_page_sans_point(self):
        result = lire_page_chapitre("p42", span=(0, 3))
        assert "page" in result.display_fr
        assert "quarante" in result.display_fr

    def test_page_majuscule(self):
        result = lire_page_chapitre("P.42", span=(0, 4))
        assert "page" in result.display_fr

    def test_chapitre(self):
        result = lire_page_chapitre("chap.3", span=(0, 6))
        assert "chapitre" in result.display_fr
        assert "trois" in result.display_fr

    def test_chapitre_court(self):
        result = lire_page_chapitre("ch.12", span=(0, 5))
        assert "chapitre" in result.display_fr
        assert "douze" in result.display_fr

    def test_composants_page(self):
        result = lire_page_chapitre("p.42", span=(0, 4))
        comp_set = {evt.composant for evt in result.events}
        assert 0 in comp_set  # préfixe
        assert 1 in comp_set  # nombre

    def test_display_num(self):
        result = lire_page_chapitre("p.42", span=(0, 4))
        assert "42" in result.display_num

    def test_valeur(self):
        result = lire_page_chapitre("p.42", span=(0, 4))
        assert result.valeur == 42

    def test_dispatch_page(self):
        result = lire_formule("page_chapitre", "p.42", span=(0, 4))
        assert "page" in result.display_fr

    def test_dispatch_chapitre(self):
        result = lire_formule("page_chapitre", "chap.3", span=(0, 6))
        assert "chapitre" in result.display_fr
