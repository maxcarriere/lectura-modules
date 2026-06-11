"""Correcteur V5 — Pipeline V1 + P2G comme etiqueteur POS/MORPHO.

Architecture :
  1. Tokenise + Syntaxe + Resegmentation         [identique V1]
  2. G2P -> phones pour chaque mot                [NOUVEAU]
  3. P2G sans ortho_words -> POS + Morpho raw      [NOUVEAU]
  4. Fusion morpho : P2G -> short-form + lexique   [NOUVEAU]
  5. Orthographe (VerificateurOrthographe)         [identique V1]
  6. Re-tag POS des mots corriges (lexique)        [adapte V1]
  7. Grammaire (appliquer_grammaire)               [identique V1]
  8. Post-grammaire + Reconstruction               [identique V1]

Le V5 remplace l'etiquetage LexiqueTagger du V1 par le P2G sans
ortho_words, qui predit le POS/Morpho *attendu* par le contexte
phonetique (independant de l'orthographe fautive). Les regles V1
restent identiques.

Si le moteur P2G n'est pas disponible, fallback vers le V1 standard.
"""

from __future__ import annotations

import logging
from typing import Any

from lectura_correcteur._config import CorrecteurV5Config
from lectura_correcteur._morpho_fusion import (
    convertir_p2g_vers_v1,
    fusionner_avec_lexique,
)
from lectura_correcteur._tagger_lexique import LexiqueTagger, _FUNCTION_WORD_POS
from lectura_correcteur._types import (
    Correction,
    MotAnalyse,
    ResultatCorrection,
    TypeCorrection,
)
from lectura_correcteur._utils import PUNCT_RE, reconstruire_phrase
from lectura_correcteur.correcteur import Correcteur, _reconstruire_avec_insertions
from lectura_correcteur.grammaire import appliquer_grammaire
from lectura_correcteur.orthographe._resegmentation import resegmenter
from lectura_correcteur.orthographe._sms import expander_sms
from lectura_correcteur.syntaxe import appliquer_syntaxe

logger = logging.getLogger(__name__)


class CorrecteurV5(Correcteur):
    """Pipeline V5 : V1 + P2G comme etiqueteur POS/MORPHO.

    Herite de Correcteur (V1) et override corriger() pour intercaler
    le P2G entre la resegmentation et l'orthographe. Les regles de
    grammaire V1 restent identiques.
    """

    def __init__(
        self,
        lexique: Any,
        *,
        config: CorrecteurV5Config | None = None,
        tagger: Any | None = None,
        tokeniseur: Any | None = None,
        g2p: Any | None = None,
    ) -> None:
        # Creer une config V5 si pas fournie
        cfg = config or CorrecteurV5Config()

        # Init V1 parent
        super().__init__(
            lexique,
            config=cfg,
            tagger=tagger,
            tokeniseur=tokeniseur,
            g2p=g2p,
        )

        self._v5_config = cfg

        # G2P tagger (neural, pour phonemisation)
        self._g2p_tagger = self._init_g2p_tagger()

        # P2G adapter (neural, pour POS/Morpho prediction)
        self._p2g_adapter = self._init_p2g_adapter()

        # LexiqueTagger pour le re-tagging des mots corriges
        self._v5_lex_tagger = LexiqueTagger(self._lexique)

        # Fallback : si P2G indisponible, utiliser le V1 tel quel
        if self._p2g_adapter is None and not cfg.fallback_lexique:
            logger.warning("P2G indisponible et fallback_lexique=False")

    def _init_g2p_tagger(self) -> Any | None:
        """Charge le tagger G2P unifie (identique V4)."""
        try:
            from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
            g2p_adapter = creer_adapter_g2p_unifie()
            if g2p_adapter is not None:
                from lectura_correcteur._tagger_hybride import TaggerHybride
                return TaggerHybride(
                    g2p_adapter, self._lexique,
                    lm_homophones=None,
                )
        except Exception:
            logger.debug("G2P Unifie V2 indisponible", exc_info=True)
        return None

    def _init_p2g_adapter(self) -> Any | None:
        """Charge l'adaptateur P2G."""
        try:
            from lectura_correcteur._adapter_p2g import creer_adapter_p2g
            adapter = creer_adapter_p2g()
            if adapter is None:
                logger.warning("P2G indisponible — fallback LexiqueTagger pour POS")
            return adapter
        except Exception:
            logger.debug("P2G indisponible", exc_info=True)
            return None

    @property
    def p2g_disponible(self) -> bool:
        """True si le pipeline P2G est disponible."""
        return (
            self._g2p_tagger is not None
            and self._p2g_adapter is not None
            and self._v5_config.activer_p2g_tagger
        )

    def corriger(self, phrase: str) -> ResultatCorrection:
        """Pipeline V5 complet.

        Si le P2G est indisponible ou desactive, tombe sur le V1 parent.
        """
        # Fallback V1 si P2G pas disponible
        if not self.p2g_disponible:
            if self._v5_config.fallback_lexique:
                return super().corriger(phrase)
            # Pas de fallback : retourner tel quel
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        return self._corriger_v5(phrase)

    def _corriger_v5(self, phrase: str) -> ResultatCorrection:
        """Pipeline V5 : V1 avec POS/Morpho P2G."""
        if not phrase.strip():
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        # 1. Tokeniser
        raw_tokens = self._tokenize(phrase)
        tokens = [tok for tok, _is_word in raw_tokens]
        # Collecter les formes non-mots (formules, chiffres) pour les
        # proteger du pipeline G2P/P2G/Ortho. On stocke les formes plutot
        # que les indices car syntaxe/reseg/SMS peuvent changer la liste.
        _non_word_forms = frozenset(
            tok for tok, iw in raw_tokens if not iw
        )

        if not tokens:
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        all_corrections: list[Correction] = []
        tokens_pre_syntaxe = list(tokens)

        # 2. Syntaxe
        if self._config.activer_syntaxe:
            tokens, corr_syntaxe = appliquer_syntaxe(tokens)
            all_corrections.extend(corr_syntaxe)

        # 3. Resegmentation
        if self._config.activer_resegmentation:
            tokens = resegmenter(tokens, self._lexique)

        # 3b. Expansion SMS
        if self._config.activer_sms:
            tokens = expander_sms(tokens, self._lexique)

        # 4. Separer ponctuation et formules des mots
        # is_skip : True pour ponctuation ET tokens non-mots (formules, chiffres)
        is_punct = [bool(PUNCT_RE.match(t)) for t in tokens]
        is_skip = [
            p or t in _non_word_forms
            for t, p in zip(tokens, is_punct)
        ]
        word_tokens = [t for t, s in zip(tokens, is_skip) if not s]
        word_tokens_orig = [
            t for t, s in zip(tokens_pre_syntaxe, is_skip) if not s
        ]
        word_indices = [i for i, s in enumerate(is_skip) if not s]

        if not word_tokens:
            return ResultatCorrection(
                phrase_originale=phrase,
                phrase_corrigee=reconstruire_phrase(tokens),
            )

        formes = [w.lower() for w in word_tokens]

        # ---- NOUVEAU V5 : P2G-based POS/Morpho ----

        # 4a. G2P phonemisation
        phones, pos_g2p = self._phonemiser(formes)

        # 4b. P2G sans ortho_words -> POS + Morpho (format UD)
        p2g_result = self._appeler_p2g(phones)

        # 4c. Conversion UD -> V1 short-form
        if p2g_result is not None:
            pos_list_p2g, morpho_p2g = convertir_p2g_vers_v1(
                p2g_result, len(formes),
            )
            # 4d. Fusion avec lexique pour combler les trous
            pos_list_p2g, morpho_p2g = fusionner_avec_lexique(
                pos_list_p2g, morpho_p2g, formes, self._lexique,
            )
        else:
            # Fallback: utiliser LexiqueTagger
            pos_list_p2g, morpho_p2g = self._fallback_lex_tags(word_tokens)

        # 4e. Sauvegarder POS P2G bruts pour la detection homophones
        # (avant les overrides qui masquent le signal P2G)
        pos_list_p2g_raw = list(pos_list_p2g)

        # 4f. Overrides mots-outils : le P2G peut confondre des homophones
        # phonetiques (ces/mais, sont/son, des/deux). On force le POS des
        # mots-outils connus, comme le fait le TaggerHybride.
        self._appliquer_overrides_p2g(formes, pos_list_p2g)

        # 5. Orthographe (identique V1)
        # L'orthographe utilise les morpho du LexiqueTagger (identique V1)
        # car le VerificateurOrthographe est calibre sur les POS lexique.
        # Le P2G morpho est reserve pour le re-tag grammaire (etape 6).
        _lex_morpho_for_ortho = self._v5_lex_tagger.tag_words(word_tokens)
        morpho_results = _lex_morpho_for_ortho

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

        # Enrichir analyses avec POS/morpho P2G
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

        # Fallback POS via lexique : corriger si POS P2G manquant et
        # que le lexique ne connait qu'un seul cgram
        for j, analysis in enumerate(analyses):
            if analysis.pos and self._lexique is not None:
                infos = self._lexique.info(analysis.corrige)
                if infos:
                    cgrams = {e["cgram"] for e in infos if e.get("cgram")}
                    if analysis.pos not in cgrams and len(cgrams) == 1:
                        analysis.pos = next(iter(cgrams))

        # 6-7. Pipeline regles V5 (re-tag adapte + grammaire V1)
        after_rules, corrs = self._pipeline_regles_v5(
            analyses, word_tokens, morpho_results, all_corrections,
            word_tokens_orig=word_tokens_orig,
            pos_list_p2g=pos_list_p2g,
            morpho_p2g=morpho_p2g,
            pos_list_p2g_raw=pos_list_p2g_raw,
            pos_g2p=pos_g2p,
        )
        all_corrections = corrs

        decided_words = [a.corrige for a in analyses]

        # 8. Reconstruction (identique V1)
        if len(after_rules) != len(decided_words):
            final_tokens = _reconstruire_avec_insertions(
                tokens, is_skip, word_indices, after_rules, decided_words,
            )
        else:
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

    # ------------------------------------------------------------------
    # Helpers P2G
    # ------------------------------------------------------------------

    def _phonemiser(self, formes: list[str]) -> tuple[list[str], list[str]]:
        """Phonemise chaque mot via le G2P tagger.

        Returns:
            (phones, pos_g2p) : phonemes IPA et POS predits par le G2P.
        """
        phones: list[str] = []
        pos_g2p: list[str] = []
        tags = self._g2p_tagger.tag_words_rich(formes)
        for i, forme in enumerate(formes):
            phone = ""
            pos = ""
            if i < len(tags):
                phone = tags[i].get("g2p", "")
                pos = tags[i].get("pos", "")
            if not phone and hasattr(self._g2p_tagger, "prononcer"):
                phone = self._g2p_tagger.prononcer(forme) or ""
            phones.append(phone if phone else forme)
            pos_g2p.append(pos)
        return phones, pos_g2p

    def _appeler_p2g(self, phones: list[str]) -> dict[str, Any] | None:
        """Appelle P2G sans ortho_words sur la phrase entiere."""
        try:
            return self._p2g_adapter.transcrire_complet(
                phones, ortho_words=None, k=5,
            )
        except Exception:
            logger.warning("P2G transcrire_complet echoue", exc_info=True)
            return None

    def _fallback_lex_tags(
        self, word_tokens: list[str],
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Fallback : tags depuis LexiqueTagger (identique V1)."""
        tags = self._v5_lex_tagger.tag_words(word_tokens)
        n = len(word_tokens)
        pos_list = [t.get("pos", "") for t in tags]
        morpho: dict[str, list[str]] = {}
        for feat in ("genre", "nombre", "temps", "mode", "personne"):
            morpho[feat] = [
                tags[j].get(feat, "_") if j < len(tags) else "_"
                for j in range(n)
            ]
        return pos_list, morpho

    def _appliquer_overrides_p2g(
        self,
        formes: list[str],
        pos_list: list[str],
    ) -> None:
        """Force le POS des mots-outils connus (in-place).

        Le P2G sans ortho_words confond souvent les homophones phonetiques
        pour les mots-outils (ces/mais, sont/son, des/deux). On force le
        POS correct en se basant sur la forme orthographique originale,
        exactement comme le fait _FUNCTION_WORD_POS dans LexiqueTagger.
        """
        for i, forme in enumerate(formes):
            if i >= len(pos_list):
                break
            override = _FUNCTION_WORD_POS.get(forme.lower())
            if override is not None:
                pos_list[i] = override

    def _construire_morpho_results(
        self,
        pos_list: list[str],
        morpho: dict[str, list[str]],
        n: int,
    ) -> list[dict]:
        """Construit list[dict] au format tagger depuis pos_list + morpho."""
        results: list[dict] = []
        for i in range(n):
            d: dict[str, str] = {}
            if i < len(pos_list):
                d["pos"] = pos_list[i]
            for feat in ("genre", "nombre", "temps", "mode", "personne"):
                vals = morpho.get(feat, [])
                if i < len(vals) and vals[i] != "_":
                    d[feat] = vals[i]
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Pipeline regles V5 : re-tag adapte
    # ------------------------------------------------------------------

    def _pipeline_regles_v5(
        self,
        analyses: list[MotAnalyse],
        word_tokens: list[str],
        morpho_results: list[dict],
        corrections: list[Correction],
        word_tokens_orig: list[str] | None = None,
        pos_list_p2g: list[str] | None = None,
        morpho_p2g: dict[str, list[str]] | None = None,
        pos_list_p2g_raw: list[str] | None = None,
        pos_g2p: list[str] | None = None,
    ) -> tuple[list[str], list[Correction]]:
        """Pipeline regles V5 : scoring + grammaire avec re-tag adapte.

        Difference avec V1 : au re-tagging (etape 6), seuls les mots
        corriges par l'orthographe sont retagues via LexiqueTagger.
        Les mots non corriges gardent leur POS/morpho P2G.
        """
        all_corrections = list(corrections)

        # 5b. Scoring unifie (identique V1)
        if self._config.activer_scoring:
            from lectura_correcteur._candidats import generer_candidats
            from lectura_correcteur._scoring import (
                extraire_contexte,
                scorer_candidats,
            )
            from lectura_correcteur.correcteur import _explication_ortho

            for j, analysis in enumerate(analyses):
                dans_lex = analysis.dans_lexique
                if analysis.corrige != analysis.original and not dans_lex:
                    dans_lex = self._lexique.existe(analysis.corrige)
                sugg = analysis.suggestions if not dans_lex else None
                candidats = generer_candidats(
                    analysis.corrige, dans_lex,
                    analysis.pos, analysis.morpho,
                    self._lexique,
                    g2p=self._g2p,
                    config=self._config,
                    suggestions=sugg,
                )
                contexte = extraire_contexte(analyses, j, self._lexique)
                scored = scorer_candidats(
                    candidats, analysis.original,
                    analysis.pos, analysis.morpho,
                    contexte, self._lexique,
                    dans_lexique=dans_lex,
                    config=self._config,
                    g2p=self._g2p,
                )
                if scored:
                    top = scored[0]
                    identite = next(
                        (c for c in scored if c.source == "identite"), None,
                    )
                    seuil = 0.0 if not dans_lex else self._config.seuil_remplacement
                    if identite is None or (top.score - identite.score) > seuil:
                        analysis.corrige = top.forme
                        analysis.confiance = top.score
                        if top.source == "morpho":
                            analysis.type_correction = TypeCorrection.GRAMMAIRE
                        elif top.source != "identite":
                            analysis.type_correction = TypeCorrection.HORS_LEXIQUE
                        if top.pos:
                            analysis.pos = top.pos
                        if top.source != "identite":
                            _regle, _expl = _explication_ortho(top, analysis.original)
                            all_corrections.append(Correction(
                                index=j,
                                original=analysis.original,
                                corrige=top.forme,
                                type_correction=analysis.type_correction,
                                regle=_regle,
                                explication=_expl,
                            ))
                    analysis.suggestions_scored = [
                        (c.forme, c.score) for c in scored[:5]
                    ]

        # 5c. Coherence POS (experimental, OFF par defaut)
        if self._config.activer_coherence:
            from lectura_correcteur._coherence import appliquer_coherence
            corr_coherence = appliquer_coherence(analyses, self._lexique)
            all_corrections.extend(corr_coherence)

        decided_words = [a.corrige for a in analyses]

        # 6. Grammaire (avec re-tag V5)
        if self._config.activer_grammaire:
            # ---- RE-TAG V5 : POS P2G + Morpho P2G ----
            # POS : P2G pour les mots non corriges (meilleur sur texte
            # fautif car independant de l'orthographe), lexique pour les
            # mots corriges par orthographe (la forme a change).
            # Morpho : idem.
            _lex_tags = self._v5_lex_tagger.tag_words(decided_words)

            pos_list: list[str] = []
            for j in range(len(analyses)):
                mot_corrige = analyses[j].corrige.lower() != analyses[j].original.lower()
                if mot_corrige and j < len(_lex_tags) and _lex_tags[j].get("pos"):
                    # Mot corrige par ortho -> POS lexique (forme corrigee)
                    pos_list.append(_lex_tags[j]["pos"])
                elif pos_list_p2g is not None and j < len(pos_list_p2g) and pos_list_p2g[j]:
                    # Mot non corrige -> POS P2G
                    pos_list.append(pos_list_p2g[j])
                elif j < len(_lex_tags) and _lex_tags[j].get("pos"):
                    # Fallback lexique
                    pos_list.append(_lex_tags[j]["pos"])
                else:
                    pos_list.append(analyses[j].pos)

            # Restreindre AUX aux seules formes d'etre/avoir
            from lectura_correcteur.grammaire._donnees import AUXILIAIRES as _AUX_FORMES
            for j in range(len(pos_list)):
                if (
                    pos_list[j] == "AUX"
                    and decided_words[j].lower() not in _AUX_FORMES
                ):
                    pos_list[j] = "VER"
            for j in range(len(analyses)):
                if j < len(pos_list):
                    analyses[j].pos = pos_list[j]

            # Morpho : hybride P2G + lexique
            morpho_dict_lists: dict[str, list[str]] = {}
            for feat in ("genre", "nombre", "temps", "mode", "personne"):
                feat_list: list[str] = []
                for j in range(len(analyses)):
                    mot_corrige = analyses[j].corrige.lower() != analyses[j].original.lower()
                    if mot_corrige and j < len(_lex_tags):
                        # Mot corrige -> morpho lexique
                        feat_list.append(_lex_tags[j].get(feat, "_"))
                    elif morpho_p2g is not None and feat in morpho_p2g and j < len(morpho_p2g[feat]):
                        # Mot non corrige -> morpho P2G
                        feat_list.append(morpho_p2g[feat][j])
                    elif j < len(_lex_tags):
                        # Fallback lexique
                        feat_list.append(_lex_tags[j].get(feat, "_"))
                    else:
                        feat_list.append("_")
                morpho_dict_lists[feat] = feat_list

            # Confiance POS
            _pos_conf = None
            if any(a.confiance_pos < 1.0 for a in analyses):
                _pos_conf = [a.confiance_pos for a in analyses]

            # 6-pre. Accord guide par PM n-gram (identique V1)
            _accord_guidance = None
            if self._config.activer_accord_pm and self._pos_ngram is not None:
                from lectura_correcteur.grammaire._accord_pm import guider_accords_pm
                _accord_guidance = guider_accords_pm(
                    decided_words, pos_list, self._lexique, self._pos_ngram,
                    seuil_violation=self._config.accord_pm_seuil_violation,
                    seuil_delta=self._config.accord_pm_seuil_delta,
                )
                for g in _accord_guidance:
                    if g.index < len(decided_words):
                        ancien = decided_words[g.index]
                        if ancien.lower() != g.forme_suggeree.lower():
                            new_forme = g.forme_suggeree
                            if ancien[0].isupper() and new_forme:
                                new_forme = new_forme[0].upper() + new_forme[1:]
                            decided_words[g.index] = new_forme
                            analyses[g.index].corrige = new_forme
                            if analyses[g.index].type_correction == TypeCorrection.AUCUNE:
                                analyses[g.index].type_correction = TypeCorrection.GRAMMAIRE
                            all_corrections.append(Correction(
                                index=g.index,
                                original=analyses[g.index].original,
                                corrige=new_forme,
                                type_correction=TypeCorrection.GRAMMAIRE,
                                regle="accord_pm",
                                explication=f"Accord PM: '{ancien}' -> '{new_forme}' ({g.pm_tag})",
                            ))

            # 6-homo. Detection homophones P2G (avant les regles V1)
            # Utilise les POS P2G bruts (avant overrides) pour comparer
            # le POS attendu par le contexte phonetique vs le POS de la
            # forme ecrite.
            _p2g_corrected_positions: set[int] = set()
            _pos_for_homo = pos_list_p2g_raw if pos_list_p2g_raw else pos_list_p2g
            if (
                self._v5_config.activer_homophones_p2g
                and _pos_for_homo is not None
            ):
                from lectura_correcteur.grammaire import detecter_homophones_p2g
                decided_words, corr_homo_p2g = detecter_homophones_p2g(
                    decided_words, _pos_for_homo, self._lexique,
                    pos_g2p=pos_g2p,
                )
                if corr_homo_p2g:
                    all_corrections.extend(corr_homo_p2g)
                    # Mettre a jour analyses et pos_list pour les mots corriges
                    for c_hp in corr_homo_p2g:
                        j = c_hp.index
                        _p2g_corrected_positions.add(j)
                        if j < len(analyses):
                            analyses[j].corrige = c_hp.corrige
                            if analyses[j].type_correction == TypeCorrection.AUCUNE:
                                analyses[j].type_correction = TypeCorrection.GRAMMAIRE
                        # Re-tagger le mot corrige via lexique
                        if j < len(pos_list):
                            _new_lex = self._v5_lex_tagger.tag_words([c_hp.corrige])
                            if _new_lex and _new_lex[0].get("pos"):
                                pos_list[j] = _new_lex[0]["pos"]
                                analyses[j].pos = pos_list[j]

            # Les regles V1 homophones tournent toujours (pour ce/se,
            # ou/ou, la/la, etc.) mais les positions deja corrigees par
            # P2G ne seront pas re-touchees car la forme a change.
            after_rules, corr_gram = appliquer_grammaire(
                decided_words, pos_list, morpho_dict_lists, self._lexique,
                originaux=word_tokens_orig if word_tokens_orig else word_tokens,
                activer_negation=self._config.activer_negation,
                pos_confiance=_pos_conf,
                pm_guidance=_accord_guidance,
            )
            all_corrections.extend(corr_gram)

            for j in range(len(analyses)):
                if j < len(after_rules) and after_rules[j].lower() != analyses[j].corrige.lower():
                    analyses[j].corrige = after_rules[j]
                    if analyses[j].type_correction == TypeCorrection.AUCUNE:
                        analyses[j].type_correction = TypeCorrection.GRAMMAIRE

            # 6b. Post-grammar scoring pass (identique V1)
            if self._config.activer_scoring:
                from lectura_correcteur._candidats import generer_candidats
                from lectura_correcteur._scoring import (
                    extraire_contexte,
                    scorer_candidats,
                )
                from lectura_correcteur.correcteur import _explication_ortho

                for j, analysis in enumerate(analyses):
                    if j < len(after_rules):
                        forme_courante = after_rules[j].lower()
                        infos = self._lexique.info(forme_courante)
                        if infos:
                            cgrams = {e["cgram"] for e in infos if e.get("cgram")}
                            if forme_courante != analysis.original.lower() and len(cgrams) == 1:
                                analysis.pos = next(iter(cgrams))
                for j, analysis in enumerate(analyses):
                    if analysis.type_correction != TypeCorrection.AUCUNE:
                        continue
                    contexte = extraire_contexte(analyses, j, self._lexique)
                    if not contexte.get("aux_gauche"):
                        continue
                    candidats = generer_candidats(
                        analysis.corrige, analysis.dans_lexique,
                        analysis.pos, analysis.morpho,
                        self._lexique,
                        g2p=self._g2p,
                        config=self._config,
                    )
                    scored = scorer_candidats(
                        candidats, analysis.original,
                        analysis.pos, analysis.morpho,
                        contexte, self._lexique,
                        config=self._config,
                        g2p=self._g2p,
                    )
                    if scored:
                        top = scored[0]
                        identite = next(
                            (c for c in scored if c.source == "identite"), None,
                        )
                        seuil = self._config.seuil_remplacement
                        if identite is None or (top.score - identite.score) > seuil:
                            analysis.corrige = top.forme
                            analysis.confiance = top.score
                            after_rules[j] = top.forme
                            if top.source == "morpho":
                                analysis.type_correction = TypeCorrection.GRAMMAIRE
                            elif top.source != "identite":
                                analysis.type_correction = TypeCorrection.HORS_LEXIQUE
                            if top.source != "identite":
                                _regle, _expl = _explication_ortho(top, analysis.original)
                                all_corrections.append(Correction(
                                    index=j,
                                    original=analysis.original,
                                    corrige=top.forme,
                                    type_correction=analysis.type_correction,
                                    regle=_regle,
                                    explication=_expl,
                                ))
        else:
            after_rules = decided_words

        # 6c-bis. LM trigramme specialise homophones (identique V1)
        if self._lm_homophones is not None:
            corr_lm_homo = self._verifier_homophones_trigram(analyses)
            all_corrections.extend(corr_lm_homo)
            for j in range(min(len(analyses), len(after_rules))):
                after_rules[j] = analyses[j].corrige

        # 6c. Detection confusions phonetiques par LM generique (identique V1)
        if self._lm is not None:
            corr_lm = self._verifier_homophones_lm(analyses)
            all_corrections.extend(corr_lm)
            for j in range(min(len(analyses), len(after_rules))):
                after_rules[j] = analyses[j].corrige

        # 6d. BiLSTM editeur homophones (identique V1)
        if self._editeur is not None:
            after_rules, corr_editeur = self._appliquer_editeur_homophones(
                after_rules, analyses, morpho_results, word_tokens,
            )
            all_corrections.extend(corr_editeur)
            for j in range(min(len(analyses), len(after_rules))):
                after_rules[j] = analyses[j].corrige

        # 6f. Viterbi POS+Morpho (identique V1)
        if (
            self._config.activer_viterbi_morpho
            and self._pos_ngram is not None
        ):
            from lectura_correcteur._viterbi_morpho import viterbi_morpho
            decided = [a.corrige for a in analyses]
            pos_tags = [a.pos for a in analyses]
            protected = [
                a.type_correction != TypeCorrection.AUCUNE
                for a in analyses
            ]
            vm_results = viterbi_morpho(
                decided,
                pos_tags,
                self._lexique,
                self._pos_ngram,
                bonus_current=self._config.viterbi_morpho_bonus_current,
                w_emission=self._config.viterbi_morpho_w_emission,
                w_transition=self._config.viterbi_morpho_w_transition,
                use_variants=self._config.viterbi_morpho_use_variants,
                protected=protected,
            )
            for j, vmr in enumerate(vm_results):
                if j >= len(analyses) or j >= len(after_rules):
                    break
                if vmr.changed:
                    new_forme = vmr.forme
                    orig = analyses[j].corrige
                    if orig and orig[0].isupper() and new_forme:
                        new_forme = new_forme[0].upper() + new_forme[1:]
                    analyses[j].corrige = new_forme
                    after_rules[j] = new_forme
                    if analyses[j].type_correction == TypeCorrection.AUCUNE:
                        analyses[j].type_correction = TypeCorrection.GRAMMAIRE
                    all_corrections.append(Correction(
                        index=j,
                        original=analyses[j].original,
                        corrige=new_forme,
                        type_correction=TypeCorrection.GRAMMAIRE,
                        regle="viterbi_morpho",
                        explication=(
                            f"Viterbi Morpho : '{orig}' -> '{new_forme}' "
                            f"(PM: {vmr.pm_tag})"
                        ),
                    ))
                if vmr.pos:
                    analyses[j].pos = vmr.pos
                morpho_update = {}
                if vmr.genre != "_":
                    morpho_update["genre"] = vmr.genre.lower()[0] if vmr.genre in ("Masc", "Fem") else vmr.genre
                if vmr.nombre != "_":
                    morpho_update["nombre"] = vmr.nombre.lower()[0] if vmr.nombre in ("Sing", "Plur") else vmr.nombre
                if morpho_update:
                    analyses[j].morpho.update(morpho_update)

        # Coherence post-corrections (identique V1)
        if self._config.activer_coherence:
            from lectura_correcteur._coherence import verifier_coherence_post_corrections
            _retagger_coh = self._v5_lex_tagger
            corr_coherence_post = verifier_coherence_post_corrections(
                analyses, self._lexique, _retagger_coh,
            )
            all_corrections.extend(corr_coherence_post)
            for j in range(min(len(analyses), len(after_rules))):
                after_rules[j] = analyses[j].corrige

        return after_rules, all_corrections
