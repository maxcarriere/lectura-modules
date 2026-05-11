"""Tests pour _tags.py : round-trip appliquer_tag/detecter_tag."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Lexique" / "src"))

from lectura_correcteur._tags import (
    CONJ_1P,
    CONJ_1S,
    CONJ_2P,
    CONJ_2S,
    CONJ_3P,
    CONJ_3S,
    CONJ_INF,
    FEM,
    HOMO_A,
    HOMO_A_ACC,
    HOMO_CA,
    HOMO_CE,
    HOMO_CES,
    HOMO_EST,
    HOMO_ET,
    HOMO_LA,
    HOMO_LA_ACC,
    HOMO_LEUR,
    HOMO_LEURS,
    HOMO_ON,
    HOMO_ONT,
    HOMO_OU,
    HOMO_OU_ACC,
    HOMO_PEUT,
    HOMO_PEU,
    HOMO_SA,
    HOMO_SE,
    HOMO_SES,
    HOMO_SON,
    HOMO_SONT,
    KEEP,
    MASC,
    N_TAGS,
    PLUR,
    PP_E,
    PP_EE,
    PP_EES,
    PP_ER,
    PP_ES,
    SING,
    TAG2IDX,
    TAGS,
    appliquer_tag,
    detecter_tag,
)
from lectura_lexique import Lexique

LEXIQUE_DB = Path(__file__).resolve().parent.parent.parent / "Lexique" / "lexique_lectura.db"


@pytest.fixture(scope="module")
def lexique():
    if not LEXIQUE_DB.exists():
        pytest.skip("Lexique DB not found")
    lex = Lexique(str(LEXIQUE_DB))
    yield lex
    lex.close()


# -- Metadata ----------------------------------------------------------

def test_tags_unique():
    """Chaque tag doit etre unique dans la liste."""
    assert len(TAGS) == len(set(TAGS))


def test_tag2idx_coherent():
    assert len(TAG2IDX) == N_TAGS
    for i, tag in enumerate(TAGS):
        assert TAG2IDX[tag] == i


# -- KEEP ---------------------------------------------------------------

def test_keep(lexique):
    assert appliquer_tag("chat", KEEP, lexique) == "chat"
    assert detecter_tag("chat", "chat", "", lexique) == KEEP


# -- Homophones ---------------------------------------------------------

_HOMO_CASES = [
    # (original_fautif, forme_corrigee, tag_attendu)
    ("a", "\u00e0", HOMO_A_ACC),
    ("\u00e0", "a", HOMO_A),
    ("est", "et", HOMO_ET),
    ("et", "est", HOMO_EST),
    ("son", "sont", HOMO_SONT),
    ("sont", "son", HOMO_SON),
    ("on", "ont", HOMO_ONT),
    ("ont", "on", HOMO_ON),
    ("ou", "o\u00f9", HOMO_OU_ACC),
    ("o\u00f9", "ou", HOMO_OU),
    ("la", "l\u00e0", HOMO_LA_ACC),
    ("l\u00e0", "la", HOMO_LA),
    ("ce", "se", HOMO_SE),
    ("se", "ce", HOMO_CE),
    ("ces", "ses", HOMO_SES),
    ("ses", "ces", HOMO_CES),
    ("leur", "leurs", HOMO_LEURS),
    ("leurs", "leur", HOMO_LEUR),
    ("sa", "\u00e7a", HOMO_CA),
    ("\u00e7a", "sa", HOMO_SA),
    ("peu", "peut", HOMO_PEUT),
    ("peut", "peu", HOMO_PEU),
]


@pytest.mark.parametrize("orig,corr,tag", _HOMO_CASES)
def test_homo_detecter(orig, corr, tag, lexique):
    """detecter_tag doit identifier le bon tag homophone."""
    detected = detecter_tag(orig, corr, "HOMO", lexique)
    assert detected == tag, f"detecter_tag({orig!r}, {corr!r}) = {detected!r}, attendu {tag!r}"


@pytest.mark.parametrize("orig,corr,tag", _HOMO_CASES)
def test_homo_appliquer(orig, corr, tag, lexique):
    """appliquer_tag doit produire la forme corrigee."""
    result = appliquer_tag(orig, tag, lexique)
    assert result == corr, f"appliquer_tag({orig!r}, {tag!r}) = {result!r}, attendu {corr!r}"


@pytest.mark.parametrize("orig,corr,tag", _HOMO_CASES)
def test_homo_roundtrip(orig, corr, tag, lexique):
    """Round-trip : appliquer(orig, detecter(orig, corr)) == corr."""
    detected = detecter_tag(orig, corr, "HOMO", lexique)
    result = appliquer_tag(orig, detected, lexique)
    assert result == corr


# -- Accords (nombre) ---------------------------------------------------

_ACC_NOMBRE_CASES = [
    ("chien", "chiens", "ACC", PLUR),
    ("chiens", "chien", "ACC", SING),
    ("animal", "animaux", "ACC", PLUR),
    ("pomme", "pommes", "ACC", PLUR),
    ("pommes", "pomme", "ACC", SING),
]


@pytest.mark.parametrize("orig,corr,type_err,tag", _ACC_NOMBRE_CASES)
def test_accord_nombre_detecter(orig, corr, type_err, tag, lexique):
    detected = detecter_tag(orig, corr, type_err, lexique)
    assert detected == tag


@pytest.mark.parametrize("orig,corr,type_err,tag", _ACC_NOMBRE_CASES)
def test_accord_nombre_roundtrip(orig, corr, type_err, tag, lexique):
    detected = detecter_tag(orig, corr, type_err, lexique)
    result = appliquer_tag(orig, detected, lexique)
    assert result == corr, f"appliquer_tag({orig!r}, {detected!r}) = {result!r}, attendu {corr!r}"


# -- Conjugaison --------------------------------------------------------

_CONJ_CASES = [
    # "mange" est ambigu (1s/3s) → on accepte CONJ_1S ou CONJ_3S
    ("mangent", "mange", "CONJ", (CONJ_1S, CONJ_3S)),
    ("mange", "mangent", "CONJ", CONJ_3P),
    ("mange", "mangeons", "CONJ", CONJ_1P),
    ("mange", "mangez", "CONJ", CONJ_2P),
    ("mange", "manges", "CONJ", CONJ_2S),
]


@pytest.mark.parametrize("orig,corr,type_err,tag", _CONJ_CASES)
def test_conj_detecter(orig, corr, type_err, tag, lexique):
    detected = detecter_tag(orig, corr, type_err, lexique)
    if isinstance(tag, tuple):
        assert detected in tag, f"detecter_tag({orig!r}, {corr!r}) = {detected!r}, attendu un de {tag!r}"
    else:
        assert detected == tag


@pytest.mark.parametrize("orig,corr,type_err,tag", _CONJ_CASES)
def test_conj_roundtrip(orig, corr, type_err, tag, lexique):
    detected = detecter_tag(orig, corr, type_err, lexique)
    result = appliquer_tag(orig, detected, lexique)
    assert result == corr, f"appliquer_tag({orig!r}, {detected!r}) = {result!r}, attendu {corr!r}"


# -- Participe passe ----------------------------------------------------

_PP_CASES = [
    ("manger", "mang\u00e9", "PP", PP_E),
    ("mang\u00e9", "manger", "PP", PP_ER),
    ("mang\u00e9", "mang\u00e9e", "PP", PP_EE),
    ("mang\u00e9", "mang\u00e9s", "PP", PP_ES),
    ("mang\u00e9", "mang\u00e9es", "PP", PP_EES),
    ("lancer", "lanc\u00e9", "PP", PP_E),
    ("lanc\u00e9", "lancer", "PP", PP_ER),
]


@pytest.mark.parametrize("orig,corr,type_err,tag", _PP_CASES)
def test_pp_detecter(orig, corr, type_err, tag, lexique):
    detected = detecter_tag(orig, corr, type_err, lexique)
    assert detected == tag


@pytest.mark.parametrize("orig,corr,type_err,tag", _PP_CASES)
def test_pp_roundtrip(orig, corr, type_err, tag, lexique):
    detected = detecter_tag(orig, corr, type_err, lexique)
    result = appliquer_tag(orig, detected, lexique)
    assert result == corr, f"appliquer_tag({orig!r}, {detected!r}) = {result!r}, attendu {corr!r}"


# -- PHON/ACCENT/TYPO retournent KEEP -----------------------------------

def test_phon_returns_keep(lexique):
    assert detecter_tag("construisent", "construit", "PHON", lexique) == KEEP


def test_accent_returns_keep(lexique):
    assert detecter_tag("caf\u00e8", "caf\u00e9", "ACCENT", lexique) == KEEP


def test_typo_returns_keep(lexique):
    assert detecter_tag("quk", "qui", "TYPO", lexique) == KEEP


# -- Coverage test sur le corpus ----------------------------------------

def test_corpus_coverage(lexique):
    """Verifie que detecter_tag couvre bien les erreurs du corpus 10k.

    Les types PHON/ACCENT/TYPO retournent KEEP (geres par OOV algorithmique).
    Les types ACC/CONJ/PP/HOMO doivent donner un tag non-KEEP.
    """
    import json

    corpus_path = Path(__file__).resolve().parent.parent / "data" / "corpus" / "corpus_10000.jsonl"
    if not corpus_path.exists():
        pytest.skip("Corpus 10k not found")

    stats = {"total": 0, "covered": 0, "keep_expected": 0, "uncovered": []}

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i >= 500:  # Limiter a 500 phrases pour le test
                break
            entry = json.loads(line)
            for err in entry["erreurs"]:
                stats["total"] += 1
                orig = err["perturbe"]
                corr = err["original"]
                type_err = err["type"]

                tag = detecter_tag(orig, corr, type_err, lexique)

                if type_err in ("PHON", "ACCENT", "TYPO"):
                    stats["keep_expected"] += 1
                    # Ces types doivent retourner KEEP
                    assert tag == KEEP, (
                        f"type={type_err}, orig={orig!r}, corr={corr!r} -> {tag}, attendu KEEP"
                    )
                    stats["covered"] += 1
                else:
                    if tag != KEEP:
                        stats["covered"] += 1
                    else:
                        stats["uncovered"].append(
                            f"type={type_err}, orig={orig!r}, corr={corr!r}"
                        )

    coverage = stats["covered"] / stats["total"] if stats["total"] else 0
    n_edit_errors = stats["total"] - stats["keep_expected"]
    n_covered_edit = stats["covered"] - stats["keep_expected"]
    edit_coverage = n_covered_edit / n_edit_errors if n_edit_errors else 0

    # Log stats
    print(f"\nCoverage: {stats['covered']}/{stats['total']} = {coverage:.1%}")
    print(f"Edit coverage (ACC/CONJ/PP/HOMO): {n_covered_edit}/{n_edit_errors} = {edit_coverage:.1%}")
    if stats["uncovered"]:
        print(f"Uncovered ({len(stats['uncovered'])}):")
        for case in stats["uncovered"][:20]:
            print(f"  {case}")

    # Au moins 80% de couverture sur les erreurs edit
    assert edit_coverage >= 0.80, (
        f"Edit coverage trop basse: {edit_coverage:.1%} "
        f"({len(stats['uncovered'])} non couverts)"
    )
