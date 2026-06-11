"""Correcteur V4 — Pipeline Orthographe + P2G sans ortho_words.

Architecture :
  Phrase -> Tokenisation -> [MotV2 list]
      |
  Passe 1 : Orthographe pure (OOV -> lexique)     [identique V2/V3]
      |
  Passe 2 : G2P -> P2G sans ortho_words           [NOUVEAU V4]
      |
  Reconstruction -> Phrase corrigee

Le V4 combine la passe 1 du V1/V3 (orthographe pure) avec un P2G sans
ortho_words pour la grammaire. Le P2G sans ortho_words fournit un
meilleur tagging POS/Morpho sur texte fautif et resout les homophones
grammaticaux que le V3 avec ortho_words echoue.

Si le moteur P2G n'est pas disponible, fallback vers les passes V2.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from lectura_correcteur._config import CorrecteurV4Config
from lectura_correcteur._passe1_orthographe import passe1_orthographe
from lectura_correcteur._passe2_p2g import passe2_p2g
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

# Regex tokenisation francaise (identique V2/V3)
_TOKEN_RE = re.compile(
    r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu|[Jj]usqu|[Qq]uelqu)"
    r"['\u2019]"
    r"|[\w]+(?:-[\w]+)*"
    r"|[^\s\w]+",
)


class CorrecteurV4:
    """Pipeline V4 : Orthographe + P2G sans ortho_words.

    Remplace les passes 2 (POS Viterbi) et 3 (Morpho Viterbi) du V2
    par un P2G sans ortho_words. La passe 1 est identique.

    Si le moteur P2G n'est pas disponible et ``fallback_v2=True``,
    le pipeline retombe sur les passes V2.
    """

    def __init__(
        self,
        lexique: Any,
        *,
        config: CorrecteurV4Config | None = None,
        tokeniseur: Any | None = None,
    ) -> None:
        self._lexique = LexiqueNormalise(lexique)
        self._config = config or CorrecteurV4Config()
        self._tokeniseur = tokeniseur

        # G2P tagger (neural, avec tag_words_rich + prononcer)
        self._g2p_tagger = self._init_g2p_tagger()

        # P2G adapter
        self._p2g_adapter = self._init_p2g_adapter()

        # POS n-gram (pour fallback V2 uniquement)
        self._pos_ngram = self._init_pos_ngram()

    def _init_g2p_tagger(self) -> Any:
        """Charge le tagger G2P unifie (identique V2/V3)."""
        from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie

        g2p_adapter = creer_adapter_g2p_unifie()
        if g2p_adapter is not None:
            from lectura_correcteur._tagger_hybride import TaggerHybride
            return TaggerHybride(
                g2p_adapter, self._lexique,
                lm_homophones=None,
            )

        logger.warning("G2P Unifie V2 indisponible — fallback LexiqueTagger")
        from lectura_correcteur._tagger_lexique import LexiqueTagger
        return LexiqueTagger(self._lexique)

    def _init_p2g_adapter(self) -> Any | None:
        """Charge l'adaptateur P2G."""
        from lectura_correcteur._adapter_p2g import creer_adapter_p2g

        adapter = creer_adapter_p2g()
        if adapter is None:
            logger.warning("P2G indisponible — fallback passes V2")
        return adapter

    def _init_pos_ngram(self) -> Any | None:
        """Charge le n-gram POS (pour fallback V2 uniquement)."""
        from lectura_correcteur._pos_ngram import PosNgram
        from pathlib import Path

        chemin = Path(__file__).resolve().parent / "data" / "pos_ngram.db"
        if chemin.exists():
            return PosNgram(chemin)
        logger.warning("pos_ngram.db absent")
        return None

    @property
    def lexique(self) -> Any:
        return self._lexique

    def corriger(self, phrase: str) -> ResultatCorrection:
        """Pipeline V4 complet : tokenisation + 2 passes + reconstruction.

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
        word_span_indices: list[int] = []
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

        # --- Passe 1 : Orthographe (identique V2/V3) ---
        passe1_orthographe(mots, self._lexique, g2p)

        # --- Passe 2 : P2G sans ortho_words (V4) ou fallback V2 ---
        if self._p2g_adapter is not None:
            passe2_p2g(
                mots, self._g2p_tagger, self._p2g_adapter,
                self._lexique, self._config,
            )
        elif self._config.fallback_v2:
            self._fallback_v2(mots)

        # --- Reconstruction ---
        return self._reconstruire(phrase, token_spans, word_span_indices, mots)

    def _fallback_v2(self, mots: list[MotV2]) -> None:
        """Fallback : execute les passes V2 si P2G indisponible."""
        from lectura_correcteur._config import CorrecteurV2Config
        from lectura_correcteur._passe2_pos import passe2_pos
        from lectura_correcteur._passe3_morpho import passe3_morpho

        v2_config = CorrecteurV2Config()

        if self._pos_ngram is not None:
            passe2_pos(
                mots, self._g2p_tagger, self._lexique,
                self._pos_ngram, None, v2_config,
            )
            passe3_morpho(mots, self._lexique, self._pos_ngram, v2_config)

    def _tokenize_with_spans(
        self, phrase: str,
    ) -> list[tuple[str, bool, int, int]]:
        """Tokenise en preservant les spans (start, end) dans la phrase originale."""
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
        """Reconstruit la phrase (identique V2/V3)."""
        replacements: list[tuple[int, int, str]] = []
        all_corrections: list[Correction] = []
        mot_analyses: list[MotAnalyse] = []

        for mi, m in enumerate(mots):
            span_idx = word_span_indices[mi]
            _tok, _iw, start, end = token_spans[span_idx]

            corrige = transferer_casse(m.original, m.forme)

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
