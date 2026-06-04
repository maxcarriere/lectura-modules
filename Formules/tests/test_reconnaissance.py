"""Tests unitaires pour reconnaissance.py — reconnaissance IPA → formule."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lectura_formules.reconnaissance import (
    reconnaitre_ipa,
    _tokenize_ipa,
    reconnaitre_maths_ipa,
    reconnaitre_maths_ipa_stt,
    detect_formula_spans,
    detect_formula_spans_stt,
    _tokenize_ipa_math,
    _is_valid_math_sequence,
    _reconstruct_maths,
    _is_math_token,
)
from lectura_formules.lecture_formules import (
    lire_nombre,
    lire_date,
    lire_heure,
    lire_monnaie,
    lire_pourcentage,
    lire_ordinal,
    lire_sigle,
    lire_maths,
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


# ══════════════════════════════════════════════════════════════════════════════
# Formules mathematiques — round-trip
# ══════════════════════════════════════════════════════════════════════════════


def _math_round_trip(formula: str, expected_display_num: str | None = None):
    """Forward via lire_maths, puis reverse via reconnaitre_maths_ipa."""
    result_fwd = lire_maths(formula)
    ipa = result_fwd.phone
    result_rev = reconnaitre_maths_ipa(ipa)
    assert result_rev is not None, (
        f"Echec reconnaissance math pour {formula!r} (IPA: {ipa!r})"
    )
    if expected_display_num is not None:
        assert result_rev.display_num == expected_display_num, (
            f"display_num: {result_rev.display_num!r} != {expected_display_num!r}"
        )
    assert result_rev.phone.replace(" ", "") == ipa.replace(" ", ""), (
        f"IPA mismatch: {result_rev.phone!r} != {ipa!r}"
    )


class TestMathsRoundTrip:
    """Tests round-trip pour les formules mathematiques."""

    @pytest.mark.parametrize("formula", [
        "2+3=5",
        "10-7=3",
        "3×4=12",
        "10÷2=5",
        "x+1=0",
        "a=b",
    ])
    def test_arithmetique_simple(self, formula):
        _math_round_trip(formula, formula)

    @pytest.mark.parametrize("formula", [
        "x=1",
        "x+y",
        "a≤b",
        "a≥b",
    ])
    def test_variables(self, formula):
        _math_round_trip(formula, formula)

    def test_equation_quadratique(self):
        _math_round_trip("f(x)=2x²+5x-3", "f(x)=2x²+5x-3")

    def test_pythagore(self):
        _math_round_trip("x²+y²=r²", "x²+y²=r²")

    def test_cube(self):
        _math_round_trip("x³", "x³")

    def test_factorielle(self):
        _math_round_trip("n!", "n!")

    def test_signe_negatif(self):
        _math_round_trip("-5+3", "-5+3")

    def test_decimale(self):
        _math_round_trip("π=3,14", "π=3,14")

    @pytest.mark.parametrize("formula", [
        "α+β=γ",
        "cos(π)",
    ])
    def test_grec(self, formula):
        _math_round_trip(formula, formula)

    @pytest.mark.parametrize("formula", [
        "sin(x)",
        "cos(π)",
    ])
    def test_fonctions(self, formula):
        _math_round_trip(formula, formula)

    def test_racine_carree(self):
        _math_round_trip("√9", "√9")

    def test_emc2(self):
        """E=mc² — la casse E→e est acceptable (meme IPA)."""
        result_fwd = lire_maths("E=mc²")
        ipa = result_fwd.phone
        result_rev = reconnaitre_maths_ipa(ipa)
        assert result_rev is not None
        # e ou E, les deux sont valides (meme IPA)
        assert result_rev.display_num in ("E=mc²", "e=mc²")


# ══════════════════════════════════════════════════════════════════════════════
# Formules mathematiques — faux positifs
# ══════════════════════════════════════════════════════════════════════════════


class TestMathsFauxPositifs:
    """Verifie que les mots courants ne sont PAS reconnus comme formules."""

    @pytest.mark.parametrize("ipa", [
        "eɡal",         # "egal" seul
        "plys",         # "plus" seul
        "mwɛ̃",         # "moins" seul
        "fwa",          # "fois" seul
        "syʁ",          # "sur" seul
        "a",            # lettre seule
        "be",           # lettre seule
        "dø",           # nombre seul (2)
        "sɛ̃k",         # nombre seul (5)
    ])
    def test_mot_isole_rejete(self, ipa):
        assert reconnaitre_maths_ipa(ipa) is None

    @pytest.mark.parametrize("ipa", [
        "dø ɑ̃fɑ̃",              # "deux enfants" — pas une formule
        "il ɛ plys ɡʁɑ̃",       # "il est plus grand"
    ])
    def test_phrase_rejetee(self, ipa):
        assert reconnaitre_maths_ipa(ipa) is None

    def test_chaine_vide(self):
        assert reconnaitre_maths_ipa("") is None

    def test_ipa_invalide(self):
        assert reconnaitre_maths_ipa("xyz!!!") is None


# ══════════════════════════════════════════════════════════════════════════════
# Formules mathematiques — tokenisation
# ══════════════════════════════════════════════════════════════════════════════


class TestMathsTokenisation:
    """Tests de la tokenisation IPA math."""

    def test_tokenise_addition(self):
        tokens = _tokenize_ipa_math("dø plys tʁwa")
        assert tokens is not None
        cats = [t.category for t in tokens]
        assert "nombre" in cats
        assert "symbole" in cats

    def test_tokenise_grec(self):
        tokens = _tokenize_ipa_math("alfa plys bɛta")
        assert tokens is not None
        cats = [t.category for t in tokens]
        assert "grec" in cats
        assert "symbole" in cats

    def test_tokenise_fonction(self):
        tokens = _tokenize_ipa_math("sinys də iks")
        assert tokens is not None
        cats = [t.category for t in tokens]
        assert "fonction" in cats
        assert "connecteur" in cats
        assert "lettre" in cats

    def test_tokenise_echoue_sur_bruit(self):
        assert _tokenize_ipa_math("xyzabc") is None


# ══════════════════════════════════════════════════════════════════════════════
# Formules mathematiques — garde de validation
# ══════════════════════════════════════════════════════════════════════════════


class TestMathsGarde:
    """Tests de la garde _is_valid_math_sequence."""

    def test_sequence_valide(self):
        tokens = _tokenize_ipa_math("dø plys tʁwa eɡal sɛ̃k")
        assert tokens is not None
        assert _is_valid_math_sequence(tokens)

    def test_trop_court_rejete(self):
        """Un seul token ne forme pas une formule."""
        tokens = _tokenize_ipa_math("dø")
        assert tokens is not None
        assert not _is_valid_math_sequence(tokens)

    def test_sans_symbole_math_rejete(self):
        """Nombres seuls sans operateur → pas une formule math."""
        tokens = _tokenize_ipa_math("dø tʁwa")
        assert tokens is not None
        assert not _is_valid_math_sequence(tokens)

    def test_prefix_sqrt_valide(self):
        """√9 : 2 tokens avec operateur prefix → valide."""
        tokens = _tokenize_ipa_math("ʁasin kaʁe də nœf")
        assert tokens is not None
        assert _is_valid_math_sequence(tokens)

    def test_postfix_carre_valide(self):
        """x² : 2 tokens avec operateur postfix → valide."""
        tokens = _tokenize_ipa_math("iks o kaʁe")
        assert tokens is not None
        assert _is_valid_math_sequence(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# Formules mathematiques — reconstruction
# ══════════════════════════════════════════════════════════════════════════════


class TestMathsReconstruction:
    """Tests de _reconstruct_maths."""

    def test_addition(self):
        tokens = _tokenize_ipa_math("dø plys tʁwa")
        assert tokens is not None
        formula = _reconstruct_maths(tokens)
        assert formula == "2+3"

    def test_egalite(self):
        tokens = _tokenize_ipa_math("dø plys tʁwa eɡal sɛ̃k")
        assert tokens is not None
        formula = _reconstruct_maths(tokens)
        assert formula == "2+3=5"

    def test_grec(self):
        tokens = _tokenize_ipa_math("alfa plys bɛta eɡal ɡama")
        assert tokens is not None
        formula = _reconstruct_maths(tokens)
        assert formula == "α+β=γ"

    def test_smart_paren(self):
        """f(x) via 'ɛf de iks'."""
        tokens = _tokenize_ipa_math("ɛf də iks")
        assert tokens is not None
        formula = _reconstruct_maths(tokens)
        assert formula == "f(x)"

    def test_racine(self):
        tokens = _tokenize_ipa_math("ʁasin kaʁe də nœf")
        assert tokens is not None
        formula = _reconstruct_maths(tokens)
        assert formula == "√9"


# ══════════════════════════════════════════════════════════════════════════════
# Formules mathematiques — detection de spans
# ══════════════════════════════════════════════════════════════════════════════


class TestMathsSpans:
    """Tests de detect_formula_spans."""

    def test_formule_dans_phrase(self):
        words = ["dø", "plys", "tʁwa", "eɡal", "sɛ̃k"]
        spans = detect_formula_spans(words, min_span=3, max_span=20)
        assert len(spans) == 1
        start, end, result = spans[0]
        assert result.display_num == "2+3=5"
        assert start == 0
        assert end == 5

    def test_formule_entouree(self):
        words = ["il", "kalkyl", "dø", "plys", "tʁwa", "eɡal", "sɛ̃k", "ʁapidmɑ̃"]
        spans = detect_formula_spans(words, min_span=3, max_span=20)
        assert len(spans) == 1
        assert spans[0][2].display_num == "2+3=5"
        assert spans[0][0] == 2
        assert spans[0][1] == 7

    def test_fx_complet(self):
        words = [
            "il", "kalkyl",
            "ɛf", "də", "iks", "eɡal", "dø", "iks", "o", "kaʁe",
            "plys", "sɛ̃k", "iks", "mwɛ̃", "tʁwa",
        ]
        spans = detect_formula_spans(words, min_span=2, max_span=20)
        assert len(spans) >= 1
        nums = [s[2].display_num for s in spans]
        assert "f(x)=2x²+5x-3" in nums

    def test_sin_x(self):
        words = ["sinys", "də", "iks"]
        spans = detect_formula_spans(words, min_span=2, max_span=20)
        assert len(spans) == 1
        assert spans[0][2].display_num == "sin(x)"

    def test_racine_9(self):
        words = ["ʁasin", "kaʁe", "də", "nœf"]
        spans = detect_formula_spans(words, min_span=2, max_span=20)
        assert len(spans) == 1
        assert spans[0][2].display_num == "√9"

    def test_aucune_formule(self):
        words = ["il", "paʁl", "fʁɑ̃sɛ"]
        spans = detect_formula_spans(words, min_span=3, max_span=20)
        assert spans == []

    def test_trop_court(self):
        words = ["dø"]
        spans = detect_formula_spans(words, min_span=3, max_span=20)
        assert spans == []


# ══════════════════════════════════════════════════════════════════════════════
# Formules mathematiques — tolerance STT
# ══════════════════════════════════════════════════════════════════════════════


class TestMathsSTT:
    """Tests de la reconnaissance math avec tolerance STT."""

    @pytest.mark.parametrize("formula", [
        "2+3=5",
        "x=1",
        "α+β=γ",
        "f(x)=2x²+5x-3",
        "sin(x)",
        "√9",
    ])
    def test_round_trip_exact(self, formula):
        """La version STT doit aussi fonctionner avec de l'IPA exact."""
        result_fwd = lire_maths(formula)
        result_rev = reconnaitre_maths_ipa_stt(result_fwd.phone)
        assert result_rev is not None, (
            f"Echec STT pour {formula!r} (IPA: {result_fwd.phone!r})"
        )

    def test_tolerance_open_e(self):
        """ɛ→e : bɛta → beta (normalisation STT)."""
        result = reconnaitre_maths_ipa_stt("alfa plys beta eɡal ɡama")
        assert result is not None
        assert result.display_num == "α+β=γ"

    def test_tolerance_ascii_g(self):
        """g ASCII → ɡ IPA : gama avec g ASCII."""
        result = reconnaitre_maths_ipa_stt("alfa plys bɛta eɡal gama")
        assert result is not None
        assert result.display_num == "α+β=γ"

    def test_tolerance_ctc_wit(self):
        """Variante CTC : wit pour huit."""
        result = reconnaitre_maths_ipa_stt("wit plys dø eɡal dis")
        assert result is not None
        assert result.display_num == "8+2=10"

    def test_faux_positif_stt(self):
        """Les mots isoles ne doivent pas matcher en STT non plus."""
        assert reconnaitre_maths_ipa_stt("eɡal") is None
        assert reconnaitre_maths_ipa_stt("plys") is None
        assert reconnaitre_maths_ipa_stt("") is None


class TestMathsSpansSTT:
    """Tests de detect_formula_spans_stt."""

    def test_spans_stt_basic(self):
        """Span detection STT avec IPA exact."""
        words = ["dø", "plys", "tʁwa", "eɡal", "sɛ̃k"]
        spans = detect_formula_spans_stt(words, min_span=3, max_span=20)
        assert len(spans) == 1
        assert spans[0][2].display_num == "2+3=5"

    def test_spans_stt_tolerance(self):
        """Span detection STT avec variante CTC."""
        words = ["wit", "plys", "dø", "eɡal", "dis"]
        spans = detect_formula_spans_stt(words, min_span=3, max_span=20)
        assert len(spans) == 1
        assert spans[0][2].display_num == "8+2=10"
