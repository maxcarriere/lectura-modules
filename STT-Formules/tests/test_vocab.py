"""Tests pour le vocabulaire STT-Formules."""

from __future__ import annotations

import pytest

from lectura_stt_formules._vocab import (
    BLANK, SPACE,
    ZERO, UN, DEUX, TROIS, QUATRE, CINQ, SIX, SEPT, HUIT, NEUF,
    DIX, ONZE, DOUZE, TREIZE, QUATORZE, QUINZE, SEIZE, UNE,
    VINGT, TRENTE, QUARANTE, CINQUANTE, SOIXANTE,
    CENT, MILLE, MILLION, MILLIARD,
    ET, VIRGULE, MOINS, PLUS,
    JANVIER, FEVRIER, MARS, AVRIL, MAI, JUIN,
    JUILLET, AOUT, SEPTEMBRE, OCTOBRE, NOVEMBRE, DECEMBRE,
    HEURE, MINUTE, SECONDE_T, MIDI,
    EURO, DOLLAR, CENTIME, LIVRE,
    POURCENT, POURMILLE,
    PREMIER, IEME, SUR, DEMI, TIERS, QUART,
    A_LETTER, B_LETTER, S_LETTER, N_LETTER, W_LETTER, Z_LETTER,
    NUM_TOKENS, VOCAB, ORTHO_TO_TOKENS,
    token_ids_to_names, vocab_to_json,
)


class TestVocabSize:
    """Verification de la taille du vocabulaire."""

    def test_num_tokens_is_87(self):
        assert NUM_TOKENS == 87

    def test_vocab_dict_has_87_entries(self):
        assert len(VOCAB) == 87

    def test_token_ids_are_contiguous_0_to_86(self):
        assert set(VOCAB.keys()) == set(range(87))


class TestTokenIdUniqueness:
    """Chaque token a un ID unique."""

    def test_all_ids_unique(self):
        all_ids = list(VOCAB.keys())
        assert len(all_ids) == len(set(all_ids))

    def test_all_names_unique(self):
        all_names = list(VOCAB.values())
        assert len(all_names) == len(set(all_names))


class TestTokenRanges:
    """Verification des plages de tokens."""

    def test_control_tokens(self):
        assert BLANK == 0
        assert SPACE == 1

    def test_numbers_range(self):
        assert ZERO == 2
        assert UNE == 19

    def test_dizaines_range(self):
        assert VINGT == 20
        assert SOIXANTE == 24

    def test_echelles_range(self):
        assert CENT == 25
        assert MILLIARD == 28

    def test_connecteurs_range(self):
        assert ET == 29
        assert PLUS == 32

    def test_mois_range(self):
        assert JANVIER == 33
        assert DECEMBRE == 44

    def test_heure_range(self):
        assert HEURE == 45
        assert MIDI == 48

    def test_devises_range(self):
        assert EURO == 49
        assert LIVRE == 52

    def test_pourcentage_range(self):
        assert POURCENT == 53
        assert POURMILLE == 54

    def test_ordinaux_range(self):
        assert PREMIER == 55
        assert QUART == 60

    def test_lettres_range(self):
        assert A_LETTER == 61
        assert Z_LETTER == 86


class TestOrthoMapping:
    """Verification du mapping ortho → tokens."""

    def test_nombres_atomiques(self):
        assert ORTHO_TO_TOKENS["zéro"] == [ZERO]
        assert ORTHO_TO_TOKENS["un"] == [UN]
        assert ORTHO_TO_TOKENS["neuf"] == [NEUF]
        assert ORTHO_TO_TOKENS["seize"] == [SEIZE]
        assert ORTHO_TO_TOKENS["une"] == [UNE]

    def test_dizaines(self):
        assert ORTHO_TO_TOKENS["vingt"] == [VINGT]
        assert ORTHO_TO_TOKENS["vingts"] == [VINGT]
        assert ORTHO_TO_TOKENS["soixante"] == [SOIXANTE]

    def test_echelles(self):
        assert ORTHO_TO_TOKENS["cent"] == [CENT]
        assert ORTHO_TO_TOKENS["cents"] == [CENT]
        assert ORTHO_TO_TOKENS["mille"] == [MILLE]
        assert ORTHO_TO_TOKENS["million"] == [MILLION]
        assert ORTHO_TO_TOKENS["millions"] == [MILLION]
        assert ORTHO_TO_TOKENS["milliard"] == [MILLIARD]
        assert ORTHO_TO_TOKENS["milliards"] == [MILLIARD]

    def test_connecteurs(self):
        assert ORTHO_TO_TOKENS["et"] == [ET]
        assert ORTHO_TO_TOKENS["virgule"] == [VIRGULE]
        assert ORTHO_TO_TOKENS["moins"] == [MOINS]
        assert ORTHO_TO_TOKENS["plus"] == [PLUS]

    def test_connecteurs_composes(self):
        assert ORTHO_TO_TOKENS["et un"] == [ET, UN]
        assert ORTHO_TO_TOKENS["et une"] == [ET, UNE]
        assert ORTHO_TO_TOKENS["et onze"] == [ET, ONZE]

    def test_mois(self):
        assert ORTHO_TO_TOKENS["janvier"] == [JANVIER]
        assert ORTHO_TO_TOKENS["décembre"] == [DECEMBRE]
        assert ORTHO_TO_TOKENS["août"] == [AOUT]

    def test_heure(self):
        assert ORTHO_TO_TOKENS["heure"] == [HEURE]
        assert ORTHO_TO_TOKENS["heures"] == [HEURE]
        assert ORTHO_TO_TOKENS["midi"] == [MIDI]

    def test_devises(self):
        assert ORTHO_TO_TOKENS["euro"] == [EURO]
        assert ORTHO_TO_TOKENS["euros"] == [EURO]
        assert ORTHO_TO_TOKENS["dollar"] == [DOLLAR]
        assert ORTHO_TO_TOKENS["dollars"] == [DOLLAR]
        assert ORTHO_TO_TOKENS["centime"] == [CENTIME]
        assert ORTHO_TO_TOKENS["centimes"] == [CENTIME]
        assert ORTHO_TO_TOKENS["livre"] == [LIVRE]
        assert ORTHO_TO_TOKENS["livres"] == [LIVRE]

    def test_pourcentage(self):
        assert ORTHO_TO_TOKENS["pour cent"] == [POURCENT]
        assert ORTHO_TO_TOKENS["pour mille"] == [POURMILLE]

    def test_ordinaux(self):
        assert ORTHO_TO_TOKENS["premier"] == [PREMIER]
        assert ORTHO_TO_TOKENS["première"] == [PREMIER]
        assert ORTHO_TO_TOKENS["deuxième"] == [DEUX, IEME]
        assert ORTHO_TO_TOKENS["dixième"] == [DIX, IEME]
        assert ORTHO_TO_TOKENS["centième"] == [CENT, IEME]

    def test_ordinaux_composes_et(self):
        assert ORTHO_TO_TOKENS["et unième"] == [ET, UN, IEME]
        assert ORTHO_TO_TOKENS["et onzième"] == [ET, ONZE, IEME]

    def test_fractions(self):
        assert ORTHO_TO_TOKENS["sur"] == [SUR]
        assert ORTHO_TO_TOKENS["demi"] == [DEMI]
        assert ORTHO_TO_TOKENS["tiers"] == [TIERS]
        assert ORTHO_TO_TOKENS["quart"] == [QUART]
        assert ORTHO_TO_TOKENS["quarts"] == [QUART]

    def test_lettres_sigles(self):
        assert ORTHO_TO_TOKENS["a"] == [A_LETTER]
        assert ORTHO_TO_TOKENS["bé"] == [B_LETTER]
        assert ORTHO_TO_TOKENS["zède"] == [Z_LETTER]
        assert ORTHO_TO_TOKENS["esse"] == [S_LETTER]
        assert ORTHO_TO_TOKENS["enne"] == [N_LETTER]
        assert ORTHO_TO_TOKENS["double-vé"] == [W_LETTER]

    def test_all_tokens_referenced(self):
        """Chaque token ID (sauf BLANK et SPACE) doit etre utilise dans au moins un mapping.

        BLANK est le token CTC blank, jamais dans ORTHO_TO_TOKENS.
        SPACE est insere dynamiquement entre composants par le tokenizer.
        """
        used_ids: set[int] = set()
        for token_list in ORTHO_TO_TOKENS.values():
            used_ids.update(token_list)
        # BLANK et SPACE ne sont jamais dans ORTHO_TO_TOKENS
        all_expected = set(range(NUM_TOKENS)) - {BLANK, SPACE}
        missing = all_expected - used_ids
        assert not missing, f"Tokens non references dans ORTHO_TO_TOKENS: {missing}"


class TestHelperFunctions:
    """Tests des fonctions utilitaires."""

    def test_token_ids_to_names(self):
        names = token_ids_to_names([QUARANTE, DEUX])
        assert names == ["QUARANTE", "DEUX"]

    def test_token_ids_to_names_with_space(self):
        names = token_ids_to_names([QUATORZE, SPACE, JUILLET])
        assert names == ["QUATORZE", "<space>", "JUILLET"]

    def test_token_ids_to_names_unknown(self):
        names = token_ids_to_names([999])
        assert names == ["?999"]

    def test_vocab_to_json(self):
        result = vocab_to_json()
        assert isinstance(result, dict)
        assert len(result) == 87
        assert result["<blank>"] == 0
        assert result["<space>"] == 1
        assert result["ZERO"] == 2
