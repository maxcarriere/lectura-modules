"""Correcteur V2 — Pipeline a 3 passes sequentielles.

Architecture :
  Phrase -> Tokenisation -> [MotV2 list]
      |
  Passe 1 : Orthographe pure (OOV -> lexique)
      |
  Passe 2 : POS Viterbi (homophones, G2P + n-gram POS)
      |
  Passe 3 : Morpho Viterbi (accords, n-gram PM)
      |
  Reconstruction -> Phrase corrigee

Le v1 (Correcteur) n'est PAS modifie — le v2 coexiste.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from lectura_correcteur._config import CorrecteurV2Config
from lectura_correcteur._passe1_orthographe import passe1_orthographe
from lectura_correcteur._passe2_pos import passe2_pos
from lectura_correcteur._passe3_morpho import passe3_morpho
from lectura_correcteur._types import (
    Correction,
    MotAnalyse,
    MotV2,
    ResultatCorrection,
    TypeCorrection,
)
from lectura_correcteur._utils import (
    LexiqueNormalise,
    transferer_casse,
)

logger = logging.getLogger(__name__)

# Regex tokenisation francaise (identique a _tagger_lexique._TOKEN_RE)
_TOKEN_RE = re.compile(
    r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu|[Jj]usqu|[Qq]uelqu)"
    r"['\u2019]"
    r"|[\w]+(?:-[\w]+)*"
    r"|[^\s\w]+",
)


class CorrecteurV2:
    """Pipeline V2 a 3 passes : Orthographe -> POS -> Morpho.

    Separe clairement les preoccupations : chaque passe a un contrat
    entree/sortie defini et n'interfere pas avec les autres.
    """

    def __init__(
        self,
        lexique: Any,
        *,
        config: CorrecteurV2Config | None = None,
        tokeniseur: Any | None = None,
    ) -> None:
        self._lexique = LexiqueNormalise(lexique)
        self._config = config or CorrecteurV2Config()
        self._tokeniseur = tokeniseur

        # G2P tagger (neural, avec tag_words_rich)
        self._g2p_tagger = self._init_g2p_tagger()

        # POS n-gram
        self._pos_ngram = self._init_pos_ngram()

        # LM homophones
        self._lm_homophones = self._init_lm_homophones()

    def _init_g2p_tagger(self):
        """Charge le tagger G2P unifie (hybride avec overrides mots-outils)."""
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie

        g2p_adapter = creer_adapter_g2p_unifie()
        if g2p_adapter is not None:
            from lectura_correcteur._tagger_hybride import TaggerHybride
            return TaggerHybride(
                g2p_adapter, self._lexique,
                lm_homophones=None,  # injecte apres init
            )

        # Fallback : tagger lexique
        logger.warning("G2P Unifie V2 indisponible — fallback LexiqueTagger")
        from lectura_correcteur._tagger_lexique import LexiqueTagger
        return LexiqueTagger(self._lexique)

    def _init_pos_ngram(self):
        """Charge le n-gram POS."""
        from lectura_correcteur._pos_ngram import PosNgram

        chemin = Path(__file__).resolve().parent / "data" / "pos_ngram.db"
        if chemin.exists():
            return PosNgram(chemin)
        logger.warning("pos_ngram.db absent — passe 2/3 degradees")
        return None

    def _init_lm_homophones(self):
        """Charge le LM trigramme specialise homophones."""
        from lectura_correcteur._lm_homophones import LMHomophones

        chemin = Path(__file__).resolve().parent / "data" / "homophones_trigrams.db"
        if chemin.exists():
            lm = LMHomophones(chemin, lexique=self._lexique)
            # Injecter dans le tagger hybride si applicable
            if hasattr(self._g2p_tagger, "_lm_homophones"):
                self._g2p_tagger._lm_homophones = lm
            return lm
        logger.warning("homophones_trigrams.db absent — LM homophones desactive")
        return None

    @property
    def lexique(self):
        return self._lexique

    def corriger(self, phrase: str) -> ResultatCorrection:
        """Pipeline V2 complet : tokenisation + 3 passes + reconstruction.

        Args:
            phrase: texte a corriger

        Returns:
            ResultatCorrection avec phrase_corrigee, mots, corrections
        """
        if not phrase.strip():
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        # Tokeniser avec spans pour reconstruction fidele
        token_spans = self._tokenize_with_spans(phrase)

        if not token_spans:
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        # Construire les MotV2 pour les word-tokens uniquement
        mots: list[MotV2] = []
        word_span_indices: list[int] = []  # index dans token_spans
        for idx, (tok, iw, _start, _end) in enumerate(token_spans):
            if iw:
                m = MotV2(
                    position=len(mots),
                    original=tok,
                    forme=tok.lower(),
                )
                mots.append(m)
                word_span_indices.append(idx)

        if not mots:
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        # G2P comme g2p (prononcer) pour passe 1
        g2p = self._g2p_tagger if hasattr(self._g2p_tagger, "prononcer") else None

        # --- Passe 1 : Orthographe ---
        passe1_orthographe(mots, self._lexique, g2p)

        # --- Passe 2 : POS Viterbi ---
        if self._pos_ngram is not None:
            passe2_pos(
                mots, self._g2p_tagger, self._lexique,
                self._pos_ngram, self._lm_homophones, self._config,
            )

        # --- Passe 3 : Morpho Viterbi ---
        if self._pos_ngram is not None:
            passe3_morpho(mots, self._lexique, self._pos_ngram, self._config)

        # --- Reconstruction ---
        return self._reconstruire(phrase, token_spans, word_span_indices, mots)

    def _tokenize_with_spans(
        self, phrase: str,
    ) -> list[tuple[str, bool, int, int]]:
        """Tokenise en preservant les spans (start, end) dans la phrase originale.

        Returns:
            list[(token, is_word, start, end)]
        """
        result: list[tuple[str, bool, int, int]] = []
        for m in _TOKEN_RE.finditer(phrase):
            tok = m.group()
            iw = tok[0].isalpha() or tok[0] == "_"
            result.append((tok, iw, m.start(), m.end()))
        return result

    def _reconstruire(
        self,
        phrase_originale: str,
        token_spans: list[tuple[str, bool, int, int]],
        word_span_indices: list[int],
        mots: list[MotV2],
    ) -> ResultatCorrection:
        """Reconstruit la phrase en remplacant dans la chaine originale.

        Preserve integralement l'espacement et la ponctuation d'origine ;
        seuls les word-tokens modifies sont substitues.
        """
        # Construire les remplacements (position decroissante pour stabilite)
        replacements: list[tuple[int, int, str]] = []  # (start, end, new_text)
        all_corrections: list[Correction] = []
        mot_analyses: list[MotAnalyse] = []

        for mi, m in enumerate(mots):
            span_idx = word_span_indices[mi]
            _tok, _iw, start, end = token_spans[span_idx]

            # Transferer la casse
            corrige = transferer_casse(m.original, m.forme)

            # Determiner le type de correction
            changed = m.original.lower() != m.forme
            type_corr = TypeCorrection.AUCUNE
            regle = ""
            explication = ""
            if changed and m.corrections:
                passe, r, expl = m.corrections[-1]
                if passe == 1:
                    type_corr = TypeCorrection.HORS_LEXIQUE
                else:
                    type_corr = TypeCorrection.GRAMMAIRE
                regle = r
                explication = expl

            mot_analyses.append(MotAnalyse(
                original=m.original,
                corrige=corrige,
                pos=m.pos,
                morpho={
                    "genre": m.genre,
                    "nombre": m.nombre,
                    "personne": m.personne,
                },
                dans_lexique=m.dans_lexique,
                type_correction=type_corr,
                confiance=1.0,
                confiance_pos=m.confiance_pos,
                pos_scores=m.pos_scores,
            ))

            if changed:
                replacements.append((start, end, corrige))
                all_corrections.append(Correction(
                    index=m.position,
                    original=m.original,
                    corrige=corrige,
                    type_correction=type_corr,
                    regle=regle,
                    explication=explication,
                ))

        # Appliquer les remplacements de droite a gauche
        phrase_corrigee = phrase_originale
        for start, end, new_text in sorted(replacements, reverse=True):
            phrase_corrigee = (
                phrase_corrigee[:start] + new_text + phrase_corrigee[end:]
            )

        return ResultatCorrection(
            phrase_originale=phrase_originale,
            phrase_corrigee=phrase_corrigee,
            mots=mot_analyses,
            corrections=all_corrections,
        )
