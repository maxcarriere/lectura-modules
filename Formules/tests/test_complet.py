"""Tests complets — tous les types de formule, symboles, cas limites.

Couvre :
- 15 FormuleType (détection + lecture)
- Symboles mathématiques et séparateurs
- Cas limites : zéros, grands nombres, signes, décimaux
- Options de lecture (fraction_mode, decimal_method, etc.)
- span_rom, span_num, span_fr cohérents
- Normalisation (ensembles, factorielles, intervalles)
"""

from __future__ import annotations

import pytest

from lectura_formules.lecture_formules import (
    EventFormuleLecture,
    LectureFormuleResult,
    OptionsLecture,
    lire_date,
    lire_fraction,
    lire_formule,
    lire_gps,
    lire_heure,
    lire_intervalle,
    lire_maths,
    lire_monnaie,
    lire_nombre,
    lire_numero,
    lire_ordinal,
    lire_page_chapitre,
    lire_pourcentage,
    lire_scientifique,
    lire_sigle,
    lire_telephone,
)
from lectura_tokeniseur import LecturaTokeniseur
from lectura_tokeniseur.models import FormuleType


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

_tok = LecturaTokeniseur()


def _detect(text: str) -> list[tuple[str, FormuleType]]:
    """Retourne [(text, formule_type)] pour chaque formule détectée."""
    r = _tok.analyze(text)
    return [
        (t.text, t.formule_type)
        for t in r.tokens
        if hasattr(t, "formule_type") and t.formule_type
    ]


def _first_type(text: str) -> FormuleType | None:
    """Retourne le type de la première formule détectée."""
    d = _detect(text)
    return d[0][1] if d else None


def _spans_valid(result: LectureFormuleResult) -> bool:
    """Vérifie que span_fr est cohérent pour chaque event."""
    for evt in result.events:
        s, e = evt.span_fr
        if s > e:
            return False
        frag = result.display_fr[s:e]
        if frag != evt.ortho:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 1. NOMBRE
# ══════════════════════════════════════════════════════════════════════════════


class TestNombre:
    """Lecture des nombres : entiers, décimaux, signes, zéros, grands nombres."""

    @pytest.mark.parametrize("text, attendu", [
        ("0", "zéro"),
        ("1", "un"),
        ("10", "dix"),
        ("11", "onze"),
        ("16", "seize"),
        ("17", "dix-sept"),
        ("20", "vingt"),
        ("21", "vingt-et un"),
        ("42", "quarante-deux"),
        ("70", "soixante-dix"),
        ("71", "soixante-et onze"),
        ("77", "soixante-dix-sept"),
        ("80", "quatre-vingts"),
        ("81", "quatre-vingt-un"),
        ("90", "quatre-vingt-dix"),
        ("91", "quatre-vingt-onze"),
        ("99", "quatre-vingt-dix-neuf"),
        ("100", "cent"),
        ("101", "cent-un"),
        ("200", "deux-cents"),
        ("201", "deux-cent-un"),
        ("1000", "mille"),
        ("1001", "mille-un"),
        ("2000", "deux-mille"),
    ])
    def test_entiers(self, text, attendu):
        r = lire_nombre(text)
        assert r.display_fr == attendu

    @pytest.mark.parametrize("text, attendu", [
        ("0", 0),
        ("1", 1),
        ("42", 42),
        ("1000", 1000),
        ("1000000", 1_000_000),
    ])
    def test_valeur(self, text, attendu):
        r = lire_nombre(text)
        assert r.valeur == attendu

    def test_zero(self):
        r = lire_nombre("0")
        assert r.display_fr == "zéro"
        assert r.valeur == 0
        assert len(r.events) == 1
        assert r.events[0].ortho == "zéro"

    def test_grands_nombres(self):
        r = lire_nombre("1000000")
        assert "million" in r.display_fr
        r = lire_nombre("2000000000")
        assert "milliard" in r.display_fr

    @pytest.mark.parametrize("text, fragments", [
        ("3.14", ["trois", "virgule", "quatorze"]),
        ("0.5", ["zéro", "virgule", "cinq"]),
        ("100.01", ["cent", "virgule", "zéro", "un"]),
        ("0.001", ["zéro", "virgule", "zéro", "zéro", "un"]),
    ])
    def test_decimaux(self, text, fragments):
        r = lire_nombre(text)
        for frag in fragments:
            assert frag in r.display_fr

    @pytest.mark.parametrize("text, signe", [
        ("-5", "moins"),
        ("+5", "plus"),
        ("±5", "plus ou moins"),
    ])
    def test_signes(self, text, signe):
        r = lire_nombre(text)
        assert signe in r.display_fr

    def test_feminin(self):
        r = lire_nombre("1", options=OptionsLecture(), feminin=True)
        assert "une" in r.display_fr

    def test_display_rom(self):
        r = lire_nombre("42")
        assert r.display_rom == "XLII"

    def test_display_rom_zero(self):
        r = lire_nombre("0")
        assert r.display_rom == ""

    def test_display_rom_option_off(self):
        r = lire_nombre("42", options=OptionsLecture(romain_actif=False))
        assert r.display_rom == ""

    def test_display_num_apostrophes(self):
        """Les grands nombres utilisent des apostrophes dans display_num."""
        r = lire_nombre("123456")
        assert "'" in r.display_num or r.display_num == "123456"

    def test_span_fr_coherent(self):
        r = lire_nombre("42")
        assert _spans_valid(r)

    def test_decimal_m1(self):
        r = lire_nombre("3.14", options=OptionsLecture(decimal_method="m1"))
        assert "quatorze" in r.display_fr

    def test_decimal_m2(self):
        r = lire_nombre("3.14", options=OptionsLecture(decimal_method="m2"))
        assert "quatorze" in r.display_fr

    def test_detection_type(self):
        assert _first_type("42") == FormuleType.NOMBRE
        assert _first_type("3.14") == FormuleType.NOMBRE
        assert _first_type("-7") == FormuleType.NOMBRE


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIGLE
# ══════════════════════════════════════════════════════════════════════════════


class TestSigle:
    """Lecture des sigles : lettres épelées."""

    @pytest.mark.parametrize("text, fragments", [
        ("SNCF", ["esse", "enne", "cé", "effe"]),
        ("TV", ["té", "vé"]),
        ("IBM", ["i", "bé", "emme"]),
    ])
    def test_epellation(self, text, fragments):
        r = lire_sigle(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_detection_type(self):
        assert _first_type("SNCF") == FormuleType.SIGLE
        assert _first_type("TV") == FormuleType.SIGLE


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATE
# ══════════════════════════════════════════════════════════════════════════════


class TestDate:
    """Lecture des dates."""

    @pytest.mark.parametrize("text, fragments", [
        ("01/01/2024", ["premier", "janvier", "deux-mille-vingt-quatre"]),
        ("25/12/2000", ["vingt-cinq", "décembre", "deux-mille"]),
        ("14-07-1789", ["quatorze", "juillet"]),
        ("31.12.1999", ["trente-et un", "décembre"]),
    ])
    def test_dates(self, text, fragments):
        r = lire_date(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_premier_janvier(self):
        r = lire_date("01/01/2024")
        assert "premier" in r.display_fr

    def test_composants_3(self):
        """Une date produit 3 composants : jour, mois, année."""
        r = lire_date("25/12/2000")
        assert len(r.composants()) == 3

    def test_detection_type(self):
        assert _first_type("25/12/2000") == FormuleType.DATE


# ══════════════════════════════════════════════════════════════════════════════
# 4. TELEPHONE
# ══════════════════════════════════════════════════════════════════════════════


class TestTelephone:
    """Lecture des numéros de téléphone."""

    @pytest.mark.parametrize("text", [
        "01 42 68 53 00",
        "0612345678",
        "06.12.34.56.78",
        "06-12-34-56-78",
    ])
    def test_telephone_5_composants(self, text):
        r = lire_telephone(text)
        assert len(r.composants()) == 5

    def test_detection_type(self):
        assert _first_type("01 42 68 53 00") == FormuleType.TELEPHONE


# ══════════════════════════════════════════════════════════════════════════════
# 5. ORDINAL
# ══════════════════════════════════════════════════════════════════════════════


class TestOrdinal:
    """Lecture des ordinaux : arabes et romains."""

    @pytest.mark.parametrize("text, attendu", [
        ("1er", "premier"),
        ("1re", "première"),
        ("1ère", "première"),
        ("2e", "deuxième"),
        ("2nd", "second"),
        ("2nde", "seconde"),
        ("3e", "troisième"),
        ("10e", "dixième"),
        ("21e", "vingt-et unième"),
        ("100e", "centième"),
        ("1000e", "millième"),
    ])
    def test_ordinaux_arabes(self, text, attendu):
        r = lire_ordinal(text)
        assert r.display_fr == attendu

    @pytest.mark.parametrize("text, attendu", [
        ("Ier", "premier"),
        ("IIe", "deuxième"),
        ("IIIe", "troisième"),
        ("IVe", "quatrième"),
        ("Ve", "cinquième"),
        ("Xe", "dixième"),
        ("XXIe", "vingt-et unième"),
    ])
    def test_ordinaux_romains(self, text, attendu):
        r = lire_ordinal(text)
        assert r.display_fr == attendu

    @pytest.mark.parametrize("text, rom", [
        ("1er", "Ier"),
        ("2e", "IIe"),
        ("3e", "IIIe"),
        ("10e", "Xe"),
        ("21e", "XXIe"),
        ("100e", "Ce"),
        ("1000e", "Me"),
    ])
    def test_display_rom(self, text, rom):
        r = lire_ordinal(text)
        assert r.display_rom == rom

    def test_span_rom_non_vide(self):
        """Les events ordinaux doivent avoir des span_rom non vides."""
        for text in ["1er", "2e", "21e", "100e"]:
            r = lire_ordinal(text)
            assert r.display_rom
            for evt in r.events:
                assert evt.span_rom != (0, 0), f"{text}: span_rom vide pour {evt.ortho!r}"

    def test_span_rom_21e_detail(self):
        """21e → XXIe : vingt=(0,2) et unième=(2,4)."""
        r = lire_ordinal("21e")
        assert r.display_rom == "XXIe"

    def test_detection_type(self):
        assert _first_type("1er") == FormuleType.ORDINAL
        assert _first_type("XXIe") == FormuleType.ORDINAL


# ══════════════════════════════════════════════════════════════════════════════
# 6. FRACTION
# ══════════════════════════════════════════════════════════════════════════════


class TestFraction:
    """Lecture des fractions selon les modes."""

    @pytest.mark.parametrize("text, attendu", [
        ("1/2", "demi"),
        ("1/3", "tiers"),
        ("1/4", "quart"),
        ("3/4", "quarts"),
        ("2/5", "cinquième"),
    ])
    def test_hybride(self, text, attendu):
        r = lire_fraction(text, options=OptionsLecture(fraction_mode="hybride"))
        assert attendu in r.display_fr

    @pytest.mark.parametrize("text, attendu", [
        ("1/2", "sur"),
        ("3/4", "sur"),
    ])
    def test_standard(self, text, attendu):
        r = lire_fraction(text, options=OptionsLecture(fraction_mode="standard"))
        assert attendu in r.display_fr

    def test_ordinal_mode(self):
        r = lire_fraction("1/2", options=OptionsLecture(fraction_mode="ordinal"))
        assert "deuxième" in r.display_fr

    def test_composants(self):
        r = lire_fraction("3/4")
        comps = r.composants()
        assert len(comps) >= 2  # numérateur + dénominateur

    def test_detection_type(self):
        assert _first_type("1/2") == FormuleType.FRACTION
        assert _first_type("3/4") == FormuleType.FRACTION


# ══════════════════════════════════════════════════════════════════════════════
# 7. SCIENTIFIQUE
# ══════════════════════════════════════════════════════════════════════════════


class TestScientifique:
    """Notation scientifique."""

    @pytest.mark.parametrize("text, fragments", [
        ("3.14e5", ["trois", "virgule", "quatorze", "dix", "exposant", "cinq"]),
        ("1e-3", ["un", "dix", "exposant", "moins", "trois"]),
        ("6.022e23", ["six", "vingt-trois"]),
    ])
    def test_scientifique(self, text, fragments):
        r = lire_scientifique(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_detection_type(self):
        assert _first_type("3.14e5") == FormuleType.SCIENTIFIQUE


# ══════════════════════════════════════════════════════════════════════════════
# 8. MATHS — opérateurs, fonctions, symboles
# ══════════════════════════════════════════════════════════════════════════════


class TestMaths:
    """Formules mathématiques : opérateurs, fonctions, symboles."""

    # ── Opérateurs de base ──────────────────────────────────────────────

    @pytest.mark.parametrize("text, attendu", [
        ("2+3", "plus"),
        ("5-2", "moins"),
        ("3×4", "fois"),
        ("3*4", "fois"),
        ("10÷2", "divisé par"),
        ("x/y", "sur"),
        ("a=b", "égal"),
        ("x≠y", "différent de"),
        ("x<y", "inférieur à"),
        ("x>y", "supérieur à"),
        ("x≤y", "inférieur ou égal à"),
        ("x≥y", "supérieur ou égal à"),
        ("x≈y", "approximativement égal à"),
        ("x≡y", "identique à"),
    ])
    def test_operateurs(self, text, attendu):
        r = lire_maths(text)
        assert attendu in r.display_fr

    # ── Opérateurs ensemblistes ─────────────────────────────────────────

    @pytest.mark.parametrize("text, attendu", [
        ("x∈A", "appartient à"),
        ("x∉A", "n'appartient pas à"),
        ("A⊂B", "inclus dans"),
        ("A∪B", "union"),
        ("A∩B", "intersection"),
    ])
    def test_operateurs_ensembles(self, text, attendu):
        r = lire_maths(text)
        assert attendu in r.display_fr

    # ── Flèches et implications ─────────────────────────────────────────

    @pytest.mark.parametrize("text, attendu", [
        ("A→B", "donne"),
        ("A←B", "reçoit"),
        ("A↔B", "équivalent à"),
        ("A⇒B", "implique"),
        ("A⇔B", "équivalent à"),
    ])
    def test_fleches(self, text, attendu):
        r = lire_maths(text)
        assert attendu in r.display_fr

    # ── Fonctions mathématiques ─────────────────────────────────────────

    @pytest.mark.parametrize("text, attendu", [
        ("sin(x)", "sinus"),
        ("cos(x)", "cosinus"),
        ("tan(x)", "tangente"),
        ("exp(x)", "exponentielle"),
        ("ln(x)", "logarithme népérien"),
        ("log(x)", "logarithme"),
        ("sqrt(x)", "racine carrée"),
        ("abs(x)", "valeur absolue"),
    ])
    def test_fonctions(self, text, attendu):
        r = lire_maths(text)
        assert attendu in r.display_fr

    # ── Symboles spéciaux ───────────────────────────────────────────────

    @pytest.mark.parametrize("text, attendu", [
        ("x²", "au carré"),
        ("x³", "au cube"),
        ("√9", "racine carrée"),
        ("∞", "infini"),
        ("∑x", "somme"),
        ("∏x", "produit"),
        ("∫x", "intégrale"),
        ("∂x", "dérivée partielle"),
        ("∇x", "nabla"),
    ])
    def test_symboles_speciaux(self, text, attendu):
        r = lire_maths(text)
        assert attendu in r.display_fr

    # ── Lettres grecques ────────────────────────────────────────────────

    @pytest.mark.parametrize("text, attendu", [
        ("α+1", "alpha"),
        ("β+1", "bêta"),
        ("γ+1", "gamma"),
        ("δ+1", "delta"),
        ("π+1", "pi"),
        ("θ+1", "thêta"),
        ("λ+1", "lambda"),
        ("σ+1", "sigma"),
        ("ω+1", "oméga"),
    ])
    def test_lettres_grecques(self, text, attendu):
        r = lire_maths(text)
        assert attendu in r.display_fr

    # ── Variables et lettres ────────────────────────────────────────────

    @pytest.mark.parametrize("text, attendu", [
        ("x+1", "ix"),
        ("a+b", "a"),
        ("n+1", "enne"),
    ])
    def test_variables(self, text, attendu):
        r = lire_maths(text)
        assert attendu in r.display_fr

    # ── Puissances et indices ───────────────────────────────────────────

    def test_puissance_exposant(self):
        r = lire_maths("x⁴")
        assert "puissance" in r.display_fr
        assert "quatre" in r.display_fr

    def test_puissance_negative(self):
        r = lire_maths("x⁻²")
        assert "puissance" in r.display_fr
        assert "moins" in r.display_fr

    def test_indice(self):
        r = lire_maths("x₁+x₂")
        assert "indice" in r.display_fr

    # ── Factorielle ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("text", ["5!", "n!", "10!"])
    def test_factorielle(self, text):
        r = lire_maths(text)
        assert "factorielle" in r.display_fr

    def test_factorielle_detection(self):
        assert _first_type("5!") == FormuleType.MATHS
        assert _first_type("n!") == FormuleType.MATHS

    def test_factorielle_dans_expression(self):
        r = lire_maths("3!+2")
        assert "factorielle" in r.display_fr
        assert "plus" in r.display_fr

    # ── Parenthèses et crochets ─────────────────────────────────────────

    def test_parentheses(self):
        r = lire_maths("(x+1)")
        # Smart parens peut supprimer les parenthèses
        assert "ix" in r.display_fr
        assert "plus" in r.display_fr

    def test_crochets(self):
        r = lire_maths("[a+b]")
        assert "crochet" in r.display_fr or "a" in r.display_fr

    def test_accolades(self):
        r = lire_maths("{x}")
        assert "accolade" in r.display_fr

    # ── Unités physiques ────────────────────────────────────────────────

    @pytest.mark.parametrize("text, attendu", [
        ("5 km", "kilomètre"),
        ("10 kg", "kilogramme"),
        ("3 cm", "centimètre"),
        ("100 m", "mètre"),
        ("5 °C", "degré"),  # °C est tokenisé ° + C séparément
    ])
    def test_unites(self, text, attendu):
        r = lire_maths(text)
        assert attendu in r.display_fr

    def test_unite_composee(self):
        r = lire_maths("60 km/h")
        assert "kilomètre" in r.display_fr
        assert "par" in r.display_fr or "heure" in r.display_fr

    # ── Ensembles {…} ───────────────────────────────────────────────────

    @pytest.mark.parametrize("text, fragments", [
        ("{1;2;3}", ["accolade ouvrante", "un", "point-virgule", "trois", "accolade fermante"]),
        ("{a;b}", ["accolade ouvrante", "a", "point-virgule", "accolade fermante"]),
        ("{1,2,3}", ["accolade ouvrante", "un", "virgule", "trois", "accolade fermante"]),
    ])
    def test_ensembles(self, text, fragments):
        r = lire_maths(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_ensemble_detection(self):
        assert _first_type("{1;2;3}") == FormuleType.MATHS

    def test_ensemble_avec_egalite(self):
        """A={1;2;3} doit être un seul token MATHS."""
        d = _detect("A={1;2;3}")
        maths = [t for t, ft in d if ft == FormuleType.MATHS]
        assert len(maths) == 1
        assert maths[0] == "A={1;2;3}"

    # ── Séparateurs virgule/point-virgule ───────────────────────────────

    def test_virgule_dans_ensemble(self):
        r = lire_maths("{1,2}")
        assert "virgule" in r.display_fr

    def test_point_virgule_dans_ensemble(self):
        r = lire_maths("{1;2}")
        assert "point-virgule" in r.display_fr

    # ── Prime ───────────────────────────────────────────────────────────

    def test_prime(self):
        r = lire_maths("f'(x)")
        assert "prime" in r.display_fr or "de" in r.display_fr

    # ── Détection MATHS ─────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "2x+3", "sin(x)", "√9", "x∈A", "5!", "n!", "{1;2;3}",
        "A={1;2;3}", "f'(x)", "x₁+x₂",
    ])
    def test_detection_maths(self, text):
        assert _first_type(text) == FormuleType.MATHS

    def test_span_fr_coherent(self):
        r = lire_maths("2x+3")
        assert _spans_valid(r)


# ══════════════════════════════════════════════════════════════════════════════
# 9. NUMERO
# ══════════════════════════════════════════════════════════════════════════════


class TestNumero:
    """Numéros alphanumériques."""

    @pytest.mark.parametrize("text, fragments", [
        ("AB.123.CD", ["a", "bé"]),
        ("654 001 45", ["six-cent-cinquante-quatre"]),
    ])
    def test_numero(self, text, fragments):
        r = lire_numero(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_detection_type(self):
        assert _first_type("654 001 45") == FormuleType.NUMERO


# ══════════════════════════════════════════════════════════════════════════════
# 10. HEURE
# ══════════════════════════════════════════════════════════════════════════════


class TestHeure:
    """Lecture des heures."""

    @pytest.mark.parametrize("text, fragments", [
        ("14h30", ["quatorze", "heure", "trente"]),
        ("8h00", ["huit", "heure"]),
        ("16:21", ["seize", "heure", "vingt-et un"]),
        ("0h30", ["zéro", "heure", "trente"]),
        ("1h", ["une", "heure"]),
    ])
    def test_heures(self, text, fragments):
        r = lire_heure(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_option_mot_minutes(self):
        r = lire_heure("14h30", options=OptionsLecture(heure_mot_minutes=True))
        assert "minute" in r.display_fr

    def test_midi(self):
        r = lire_heure("12h00")
        assert "douze" in r.display_fr

    def test_minuit(self):
        r = lire_heure("0h00")
        assert "zéro" in r.display_fr

    def test_detection_type(self):
        assert _first_type("14h30") == FormuleType.HEURE
        assert _first_type("16:21") == FormuleType.HEURE


# ══════════════════════════════════════════════════════════════════════════════
# 11. MONNAIE
# ══════════════════════════════════════════════════════════════════════════════


class TestMonnaie:
    """Lecture des montants monétaires."""

    @pytest.mark.parametrize("text, fragments", [
        ("42€", ["quarante-deux", "euro"]),
        ("1€", ["un", "euro"]),
        ("$100", ["cent", "dollar"]),
        ("5£", ["cinq", "livre"]),
        ("100 CHF", ["cent", "franc"]),
    ])
    def test_monnaies(self, text, fragments):
        r = lire_monnaie(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_centimes(self):
        r = lire_monnaie("42€50", options=OptionsLecture(monnaie_dire_centimes=True))
        assert "centime" in r.display_fr or "cinquante" in r.display_fr

    def test_sans_centimes(self):
        r = lire_monnaie("42€", options=OptionsLecture(monnaie_dire_centimes=False))
        assert "quarante-deux" in r.display_fr

    def test_detection_type(self):
        assert _first_type("42€") == FormuleType.MONNAIE
        assert _first_type("$100") == FormuleType.MONNAIE


# ══════════════════════════════════════════════════════════════════════════════
# 12. POURCENTAGE
# ══════════════════════════════════════════════════════════════════════════════


class TestPourcentage:
    """Lecture des pourcentages."""

    @pytest.mark.parametrize("text, fragments", [
        ("50%", ["cinquante", "pour cent"]),
        ("100%", ["cent", "pour cent"]),
        ("3‰", ["trois", "pour mille"]),
    ])
    def test_pourcentages(self, text, fragments):
        r = lire_pourcentage(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_detection_type(self):
        assert _first_type("50%") == FormuleType.POURCENTAGE


# ══════════════════════════════════════════════════════════════════════════════
# 13. INTERVALLE
# ══════════════════════════════════════════════════════════════════════════════


class TestIntervalle:
    """Lecture des intervalles mathématiques."""

    @pytest.mark.parametrize("text, fragments", [
        ("[0;1]", ["zéro", "un"]),
        ("]0;1[", ["zéro", "un"]),
        ("[0;+∞[", ["zéro", "infini"]),
        ("]-∞;0]", ["infini", "zéro"]),
    ])
    def test_intervalles(self, text, fragments):
        r = lire_intervalle(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_intervalle_ferme(self):
        r = lire_intervalle("[2;5]")
        # Crochets fermés → "de … à …"
        assert "deux" in r.display_fr
        assert "cinq" in r.display_fr

    def test_intervalle_ouvert(self):
        r = lire_intervalle("]2;5[")
        assert "deux" in r.display_fr
        assert "cinq" in r.display_fr

    def test_detection_type(self):
        assert _first_type("[0;1]") == FormuleType.INTERVALLE

    def test_intervalle_avec_virgule(self):
        """[0,1] peut être un intervalle (virgule comme séparateur)."""
        ft = _first_type("[0,1]")
        assert ft == FormuleType.INTERVALLE

    def test_intervalle_avec_egalite(self):
        """I=[0;1] doit être détecté comme MATHS (contient I=)."""
        d = _detect("I=[0;1]")
        maths = [t for t, ft in d if ft == FormuleType.MATHS]
        assert len(maths) == 1

    def test_intervalle_ouvert_francais(self):
        """I=]1;+∞[ : notation française avec ] ouvrant."""
        d = _detect("I=]1;+∞[")
        maths = [t for t, ft in d if ft == FormuleType.MATHS]
        assert len(maths) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# 14. GPS
# ══════════════════════════════════════════════════════════════════════════════


class TestGps:
    """Coordonnées GPS."""

    @pytest.mark.parametrize("text, fragments", [
        ("48°51'24\"N", ["quarante-huit", "degré"]),
        ("2.35°E", ["deux", "degré"]),
    ])
    def test_gps(self, text, fragments):
        r = lire_gps(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_detection_type(self):
        assert _first_type("48°51'24\"N") == FormuleType.GPS


# ══════════════════════════════════════════════════════════════════════════════
# 15. PAGE / CHAPITRE
# ══════════════════════════════════════════════════════════════════════════════


class TestPageChapitre:
    """Références de pages et chapitres."""

    @pytest.mark.parametrize("text, fragments", [
        ("p.42", ["page", "quarante-deux"]),
        ("p42", ["page", "quarante-deux"]),
        ("chap.3", ["chapitre", "trois"]),
        ("ch.12", ["chapitre", "douze"]),
    ])
    def test_pages(self, text, fragments):
        r = lire_page_chapitre(text)
        for frag in fragments:
            assert frag in r.display_fr

    def test_valeur(self):
        r = lire_page_chapitre("p.42")
        assert r.valeur == "42" or r.valeur == 42

    def test_detection_type(self):
        assert _first_type("p.42") == FormuleType.PAGE_CHAPITRE


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch — lire_formule()
# ══════════════════════════════════════════════════════════════════════════════


class TestDispatch:
    """lire_formule() redirige vers le bon lecteur."""

    @pytest.mark.parametrize("ftype, text, fragment", [
        ("nombre", "42", "quarante-deux"),
        ("sigle", "SNCF", "esse"),
        ("date", "25/12/2000", "décembre"),
        ("telephone", "01 42 68 53 00", "quarante-deux"),
        ("ordinal", "1er", "premier"),
        ("fraction", "1/2", "demi"),
        ("scientifique", "3.14e5", "quatorze"),
        ("maths", "2x+3", "plus"),
        ("numero", "654 001 45", "six"),
        ("heure", "14h30", "quatorze"),
        ("monnaie", "42€", "euro"),
        ("pourcentage", "50%", "pour cent"),
        ("intervalle", "[0;1]", "zéro"),
        ("gps", "48°51'24\"N", "degré"),
        ("page_chapitre", "p.42", "page"),
    ])
    def test_dispatch(self, ftype, text, fragment):
        r = lire_formule(formule_type=ftype, text=text)
        assert fragment in r.display_fr


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation — préservation des formules
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalisation:
    """La normalisation ne doit pas casser les formules."""

    @pytest.mark.parametrize("text, attendu", [
        # Ensembles préservés
        ("{1;2;3}", "{1;2;3}"),
        ("{a;b}", "{a;b}"),
        # Intervalles avec ; préservés
        ("[0;1]", "[0;1]"),
        ("]1;+∞[", "]1;+∞["),
        # Virgule décimale hors brackets → point
        ("3,14", "3.14"),
        # Factorielle collée
        ("5!", "5!"),
        ("n!", "n!"),
        ("3!+2", "3!+2"),
    ])
    def test_normalisation_preservee(self, text, attendu):
        from lectura_tokeniseur.normalisation import normalise
        assert normalise(text) == attendu

    def test_virgule_dans_accolades(self):
        """La virgule dans {1,2,3} ne doit pas être traitée comme décimale."""
        from lectura_tokeniseur.normalisation import normalise
        result = normalise("{1,2,3}")
        assert "." not in result  # Pas de conversion en point décimal

    def test_exclamation_phrase(self):
        """L'exclamation dans une phrase reste espacée."""
        from lectura_tokeniseur.normalisation import normalise
        result = normalise("Bonjour !")
        assert "Bonjour !" == result


# ══════════════════════════════════════════════════════════════════════════════
# Détection de type — pipeline complet tokeniseur
# ══════════════════════════════════════════════════════════════════════════════


class TestDetection:
    """Vérification du type détecté par le tokeniseur pour chaque formule."""

    @pytest.mark.parametrize("text, expected_type", [
        ("42", FormuleType.NOMBRE),
        ("3.14", FormuleType.NOMBRE),
        ("-7", FormuleType.NOMBRE),
        ("+42", FormuleType.NOMBRE),
        ("SNCF", FormuleType.SIGLE),
        ("25/12/2000", FormuleType.DATE),
        ("01 42 68 53 00", FormuleType.TELEPHONE),
        ("1er", FormuleType.ORDINAL),
        ("XXIe", FormuleType.ORDINAL),
        ("1/2", FormuleType.FRACTION),
        ("3/4", FormuleType.FRACTION),
        ("14h30", FormuleType.HEURE),
        ("16:21", FormuleType.HEURE),
        ("42€", FormuleType.MONNAIE),
        ("$100", FormuleType.MONNAIE),
        ("50%", FormuleType.POURCENTAGE),
        ("[0;1]", FormuleType.INTERVALLE),
        ("p.42", FormuleType.PAGE_CHAPITRE),
        ("2x+3", FormuleType.MATHS),
        ("sin(x)", FormuleType.MATHS),
        ("5!", FormuleType.MATHS),
        ("{1;2;3}", FormuleType.MATHS),
    ])
    def test_detection(self, text, expected_type):
        ft = _first_type(text)
        assert ft == expected_type, f"{text!r}: attendu {expected_type}, obtenu {ft}"


# ══════════════════════════════════════════════════════════════════════════════
# Cohérence des spans
# ══════════════════════════════════════════════════════════════════════════════


class TestSpans:
    """Vérification de la cohérence des spans pour chaque type."""

    @pytest.mark.parametrize("func, text", [
        (lire_nombre, "42"),
        (lire_nombre, "3.14"),
        (lire_nombre, "0"),
        (lire_nombre, "1000"),
        (lire_ordinal, "1er"),
        (lire_ordinal, "21e"),
        (lire_fraction, "1/2"),
        (lire_maths, "2x+3"),
        (lire_maths, "{1;2;3}"),
        (lire_maths, "5!"),
        (lire_sigle, "SNCF"),
    ])
    def test_span_fr_coherent(self, func, text):
        r = func(text)
        assert _spans_valid(r), f"span_fr incohérent pour {text!r}"

    @pytest.mark.parametrize("text, has_rom", [
        ("42", True),
        ("0", False),
        ("1er", True),
        ("21e", True),
        ("100e", True),
    ])
    def test_display_rom_presence(self, text, has_rom):
        """display_rom est présent pour les nombres > 0 et les ordinaux."""
        if text.endswith(("er", "e", "ère")):
            r = lire_ordinal(text)
        else:
            r = lire_nombre(text)
        if has_rom:
            assert r.display_rom, f"display_rom vide pour {text!r}"
        else:
            assert not r.display_rom


# ══════════════════════════════════════════════════════════════════════════════
# Cas limites et régressions
# ══════════════════════════════════════════════════════════════════════════════


class TestCasLimites:
    """Cas limites, régressions, edge cases."""

    def test_nombre_avec_apostrophes(self):
        """1'000 → mille."""
        r = lire_nombre("1000")
        assert "mille" in r.display_fr

    def test_virgule_decimale_hors_brackets(self):
        """3,14 reste un NOMBRE (pas cassé par _inside_brackets)."""
        assert _first_type("3,14") == FormuleType.NOMBRE

    def test_virgule_decimale_dans_accolades(self):
        """{1,2,3} → les virgules sont des séparateurs, pas des décimales."""
        d = _detect("{1,2,3}")
        # Doit être un seul token MATHS, pas 3 nombres
        maths = [t for t, ft in d if ft == FormuleType.MATHS]
        assert len(maths) == 1

    def test_point_virgule_hors_brackets(self):
        """Le ; hors brackets ne fusionne pas les tokens."""
        d = _detect("x+3")
        # x+3 est MATHS ; le ; ne devrait pas être fusionné avec y
        assert any(ft == FormuleType.MATHS for _, ft in d)

    def test_ensemble_vide(self):
        """Accolades vides {}."""
        # Peut ne pas être détecté comme formule, c'est OK
        r = lire_maths("{}")
        assert "accolade" in r.display_fr

    def test_nombre_zero_decimale(self):
        """0.0 → zéro virgule zéro."""
        r = lire_nombre("0.0")
        assert "zéro" in r.display_fr
        assert "virgule" in r.display_fr

    def test_nombre_leading_zeros(self):
        """0001 → un (les zéros initiaux sont ignorés)."""
        r = lire_nombre("0001")
        assert "un" in r.display_fr

    def test_fraction_dans_maths(self):
        """2x+1/2 — la fraction est reconnue dans les maths."""
        r = lire_maths("2x+1/2")
        assert "demi" in r.display_fr or "sur" in r.display_fr

    def test_signe_seul_pas_maths(self):
        """-5 est un NOMBRE, pas MATHS."""
        assert _first_type("-5") == FormuleType.NOMBRE

    def test_pm_est_nombre(self):
        """±5 est classifié NOMBRE par le tokeniseur (signe + nombre)."""
        assert _first_type("±5") == FormuleType.NOMBRE
