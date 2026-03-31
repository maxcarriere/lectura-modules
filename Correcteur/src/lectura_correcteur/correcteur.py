"""Orchestrateur : chaine les sous-modules orthographe, grammaire, syntaxe.

Pipeline :
1. Tokeniser (tokeniseur du MorphoTagger CRF)
2. Syntaxe (majuscules, espaces)
3. Resegmentation (apostrophes SMS)
4. Analyse morpho (MorphoTagger CRF -> POS, nombre, genre, personne)
5. Orthographe (verification lexique : mot existe ou pas)
6. Grammaire (accords, conjugaison, homophones contextuels)
7. Reconstruction
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur._config import CorrecteurConfig
from lectura_correcteur._morpho import MorphoTagger
from lectura_correcteur._types import (
    Correction,
    MotAnalyse,
    ResultatCorrection,
    TypeCorrection,
)
from lectura_correcteur._utils import PUNCT_RE, reconstruire_phrase
from lectura_correcteur.grammaire import appliquer_grammaire
from lectura_correcteur.orthographe import VerificateurOrthographe
from lectura_correcteur.orthographe._resegmentation import resegmenter
from lectura_correcteur.syntaxe import appliquer_syntaxe


class Correcteur:
    """Pipeline complet de correction orthographique et grammaticale.

    Utilise un MorphoTagger CRF embarque pour l'analyse POS/morpho.
    Pas de G2P, pas de P2G, pas de lexique embarque.
    Le developpeur branche son propre lexique via LexiqueProtocol.
    """

    def __init__(
        self,
        lexique: Any,
        *,
        config: CorrecteurConfig | None = None,
    ) -> None:
        """Initialise le correcteur avec injection de dependances.

        Args:
            lexique: Objet satisfaisant LexiqueProtocol
            config: Configuration du correcteur (optionnelle)
        """
        self._lexique = lexique
        self._config = config or CorrecteurConfig()
        self._tagger = MorphoTagger()
        self._verificateur = VerificateurOrthographe(lexique)

    @property
    def lexique(self):
        return self._lexique

    def corriger(self, phrase: str) -> ResultatCorrection:
        """Pipeline complet de correction.

        Etapes :
        1. Tokenisation (via MorphoTagger CRF)
        2. Syntaxe (ponctuation, majuscules)
        3. Resegmentation (apostrophes SMS)
        4. Analyse morpho (CRF -> POS, genre, nombre, personne)
        5. Orthographe (verification lexique)
        6. Grammaire (accords, conjugaison, homophones)
        7. Reconstruction de la phrase
        """
        if not phrase.strip():
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        # 1. Tokeniser via le tokeniseur du CRF
        raw_tokens = self._tagger.tokenize(phrase)
        tokens = [tok for tok, _is_word in raw_tokens]

        if not tokens:
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        all_corrections: list[Correction] = []

        # 2. Syntaxe
        if self._config.activer_syntaxe:
            tokens, corr_syntaxe = appliquer_syntaxe(tokens)
            all_corrections.extend(corr_syntaxe)

        # 3. Resegmentation
        if self._config.activer_resegmentation:
            tokens = resegmenter(tokens, self._lexique)

        # 4. Separer ponctuation des mots
        is_punct = [bool(PUNCT_RE.match(t)) for t in tokens]
        word_tokens = [t for t, p in zip(tokens, is_punct) if not p]
        word_indices = [i for i, p in enumerate(is_punct) if not p]

        if not word_tokens:
            return ResultatCorrection(
                phrase_originale=phrase,
                phrase_corrigee=reconstruire_phrase(tokens),
            )

        # 4b. Analyse morpho (CRF)
        morpho_results = self._tagger.tag_words(word_tokens)

        # 5. Orthographe (verification lexique)
        if self._config.activer_orthographe:
            analyses = self._verificateur.verifier_phrase(
                word_tokens, morpho_results,
            )
        else:
            analyses = [
                MotAnalyse(
                    original=t,
                    corrige=t,
                    pos=morpho_results[j].get("pos", "") if j < len(morpho_results) else "",
                    dans_lexique=self._lexique.existe(t),
                )
                for j, t in enumerate(word_tokens)
            ]

        # Enrichir analyses avec les POS/morpho du CRF
        for j, analysis in enumerate(analyses):
            if j < len(morpho_results):
                mr = morpho_results[j]
                if not analysis.pos:
                    analysis.pos = mr.get("pos", "")
                morpho_dict = {}
                for key in ("genre", "nombre", "temps", "mode", "personne"):
                    val = mr.get(key)
                    if val is not None:
                        morpho_dict[key] = val
                if not analysis.morpho:
                    analysis.morpho = morpho_dict

        decided_words = [a.corrige for a in analyses]

        # 6. Grammaire
        if self._config.activer_grammaire:
            pos_list = [a.pos for a in analyses]
            morpho_dict_lists: dict[str, list[str]] = {}
            if analyses:
                for feat in ("genre", "nombre", "temps", "mode", "personne"):
                    morpho_dict_lists[feat] = [
                        a.morpho.get(feat, "_") for a in analyses
                    ]

            after_rules, corr_gram = appliquer_grammaire(
                decided_words, pos_list, morpho_dict_lists, self._lexique,
                originaux=word_tokens,
            )
            all_corrections.extend(corr_gram)

            for j in range(len(analyses)):
                if j < len(after_rules) and after_rules[j].lower() != analyses[j].corrige.lower():
                    analyses[j].corrige = after_rules[j]
                    if analyses[j].type_correction == TypeCorrection.AUCUNE:
                        analyses[j].type_correction = TypeCorrection.GRAMMAIRE
        else:
            after_rules = decided_words

        # 7. Reconstruction
        final_tokens = list(tokens)
        for idx, wi in enumerate(word_indices):
            if idx < len(analyses):
                final_tokens[wi] = analyses[idx].corrige

        phrase_corrigee = reconstruire_phrase(final_tokens)

        return ResultatCorrection(
            phrase_originale=phrase,
            phrase_corrigee=phrase_corrigee,
            mots=analyses,
            corrections=all_corrections,
        )
