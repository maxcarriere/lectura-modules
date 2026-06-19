"""Orchestrateur : chaine les sous-modules orthographe, grammaire, syntaxe.

Pipeline :
1. Tokeniser
2. Syntaxe (majuscules, espaces)
3. Resegmentation (apostrophes SMS)
4. Analyse morpho (lookup lexique par defaut, modele injectable)
5. Orthographe (verification lexique : mot existe ou pas)
6. Grammaire (accords, conjugaison, homophones contextuels)
7. Reconstruction
"""

from __future__ import annotations

from typing import Any

from lectura_correcteur._candidats import generer_candidats
from lectura_correcteur._coherence import (
    appliquer_coherence,
    verifier_coherence_post_corrections,
)
from lectura_correcteur._config import CorrecteurConfig
from lectura_correcteur._phones import _RuleBasedG2P
from lectura_correcteur._tagger_lexique import LexiqueTagger
from lectura_correcteur._scoring import (
    extraire_contexte,
    scorer_candidats,
)
from lectura_correcteur._types import (
    Candidat,
    Correction,
    MotAnalyse,
    ResultatCorrection,
    TypeCorrection,
)
from lectura_correcteur._utils import PUNCT_RE, LexiqueNormalise, reconstruire_phrase
from lectura_correcteur.grammaire import appliquer_grammaire
from lectura_correcteur.orthographe import VerificateurOrthographe
from lectura_correcteur.orthographe._resegmentation import resegmenter
from lectura_correcteur.orthographe._suggestions import _variantes_accents
from lectura_correcteur.orthographe._sms import expander_sms
from lectura_correcteur.syntaxe import appliquer_syntaxe


def _explication_ortho(top: Candidat, original: str) -> tuple[str, str]:
    """Retourne (regle, explication) pour une correction orthographique."""
    src = top.source
    if src in ("ortho_d1", "ortho_d2"):
        return "ortho.distance", f"Mot hors lexique, distance {src[-1]}"
    if src == "ortho_suggestion":
        return "ortho.distance", "Mot hors lexique, suggestion orthographique"
    if src == "homophone":
        return "ortho.homophone", f"Homophone : '{original}' -> '{top.forme}'"
    if src == "g2p":
        return "ortho.phonetique", "Correction phonetique (G2P)"
    if src == "phone_proche":
        return "ortho.phonetique", "Mot proche phonetiquement"
    if src == "morpho":
        return "ortho.morpho", "Variante morphologique"
    return "ortho.autre", f"Correction orthographique ({src})"


class Correcteur:
    """Pipeline complet de correction orthographique et grammaticale.

    Par defaut, utilise un tagger par lookup lexique (zero modele).
    Un tagger et/ou G2P externe peuvent etre injectes via les Protocols.
    Le developpeur branche son propre lexique via LexiqueProtocol.
    """

    def __init__(
        self,
        lexique: Any,
        *,
        config: CorrecteurConfig | None = None,
        tagger: Any | None = None,
        tokeniseur: Any | None = None,
        g2p: Any | None = None,
    ) -> None:
        """Initialise le correcteur avec injection de dependances.

        Args:
            lexique: Objet satisfaisant LexiqueProtocol
            config: Configuration du correcteur (optionnelle)
            tagger: Objet satisfaisant TaggerProtocol (optionnel, fallback LexiqueTagger)
            tokeniseur: Objet satisfaisant TokeniseurProtocol (optionnel)
            g2p: Objet satisfaisant G2PProtocol (optionnel)
        """
        self._lexique = LexiqueNormalise(lexique)
        self._config = config or CorrecteurConfig()
        self._tagger = tagger if tagger is not None else LexiqueTagger(self._lexique)
        self._tokeniseur = tokeniseur
        self._g2p = g2p
        # Tagger hybride : G2P contextuel + overrides mots-outils
        if self._config.activer_tagger_hybride and tagger is None:
            from lectura_correcteur._tagger_hybride import TaggerHybride
            from lectura_correcteur._adapter_g2p_unifie import creer_adapter_g2p_unifie
            _g2p_adapter = creer_adapter_g2p_unifie()
            if _g2p_adapter is not None:
                self._tagger = TaggerHybride(
                    _g2p_adapter, self._lexique,
                    seuil_freq_voisin=self._config.seuil_freq_voisin,
                )
                self._lex_tagger = LexiqueTagger(self._lexique)
        # Reutiliser le tagger comme G2P s'il a la capacite prononcer/g2p
        if self._g2p is None and hasattr(self._tagger, "prononcer"):
            self._g2p = self._tagger
        elif self._g2p is None and hasattr(self._tagger, "g2p"):
            self._g2p = self._tagger
        # Fallback : estimateur phonetique par regles (zero dependance)
        if self._g2p is None:
            self._g2p = _RuleBasedG2P()
        # Garder un LexiqueTagger pour les regles de grammaire
        # (regles calibrees sur ses POS, ex: est→ADJ)
        # Ne pas ecraser _lex_tagger si deja positionne par le tagger hybride
        if not hasattr(self, '_lex_tagger') or self._lex_tagger is None:
            self._lex_tagger = (
                LexiqueTagger(self._lexique) if tagger is not None else None
            )
        self._verificateur = VerificateurOrthographe(
            self._lexique, max_suggestions=self._config.max_suggestions,
            distance=self._config.distance_suggestions,
            g2p=self._g2p,
            scoring_actif=self._config.activer_scoring,
        )
        # Charger matrice de transition Viterbi si disponible
        self._transition_matrix = None
        if hasattr(self._tagger, "tag_words_rich"):
            self._init_viterbi()
        # Charger le BiLSTM edit tagger pour homophones si active
        self._editeur = None
        if self._config.activer_editeur_homophones:
            self._init_editeur()
        # Charger le modele de langue n-gram si active
        self._lm = None
        if self._config.activer_lm:
            self._init_lm()
        # Charger le LM trigramme specialise homophones
        self._lm_homophones = None
        if self._config.activer_lm_homophones:
            self._init_lm_homophones()
        # Injecter lm_homophones dans le tagger hybride pour le check ambiguite
        if hasattr(self._tagger, '_lm_homophones') and self._lm_homophones is not None:
            self._tagger._lm_homophones = self._lm_homophones
        # Charger le n-gram POS
        self._pos_ngram = None
        if self._config.activer_pos_ngram:
            self._init_pos_ngram()

    def _init_viterbi(self) -> None:
        """Charge la matrice de transition si le fichier existe."""
        from pathlib import Path
        matrix_path = (
            Path(__file__).parent / "data" / "transition_matrix_bigram.json"
        )
        if matrix_path.exists():
            from lectura_correcteur._viterbi import charger_matrice_transition
            self._transition_matrix = charger_matrice_transition(matrix_path)

    def _init_editeur(self) -> None:
        """Charge le BiLSTM edit tagger si les poids sont disponibles."""
        from pathlib import Path
        data_dir = Path(__file__).parent / "data"
        weights = data_dir / "editeur_weights.json.gz"
        vocab = data_dir / "editeur_vocab.json"
        if weights.exists() and vocab.exists():
            from lectura_correcteur._editeur_numpy import EditeurNumpy
            self._editeur = EditeurNumpy(str(weights), str(vocab))

    def _init_lm(self) -> None:
        """Charge le modele de langue n-gram si le fichier existe."""
        from pathlib import Path
        from lectura_correcteur._language_model import ScorerNgram
        chemin = self._config.chemin_lm
        if not chemin:
            chemin = str(Path(__file__).parent / "data" / "ngram.db")
        if Path(chemin).exists():
            self._lm = ScorerNgram(chemin)

    def _init_lm_homophones(self) -> None:
        """Charge le LM trigramme specialise homophones."""
        from pathlib import Path
        from lectura_correcteur._lm_homophones import LMHomophones
        chemin = self._config.chemin_lm_homophones
        if not chemin:
            chemin = str(Path(__file__).parent / "data" / "homophones_trigrams.db")
        if Path(chemin).exists():
            self._lm_homophones = LMHomophones(chemin, lexique=self._lexique)

    def _init_pos_ngram(self) -> None:
        """Charge le n-gram POS pour validation des corrections."""
        from pathlib import Path
        from lectura_correcteur._pos_ngram import PosNgram
        chemin = self._config.chemin_pos_ngram
        if not chemin:
            chemin = str(Path(__file__).parent / "data" / "pos_ngram.db")
        if Path(chemin).exists():
            self._pos_ngram = PosNgram(chemin)

    @property
    def lexique(self):
        return self._lexique

    def _tokenize(self, phrase: str) -> list[tuple[str, bool]]:
        """Tokenise via le tokeniseur externe ou le tagger CRF."""
        if self._tokeniseur is not None:
            tokens = self._tokeniseur.tokeniser(phrase)
            return _adapter_tokens(tokens)
        return self._tagger.tokenize(phrase)

    def corriger(self, phrase: str) -> ResultatCorrection:
        """Pipeline complet de correction.

        Etapes :
        1. Tokenisation (via tokeniseur externe ou MorphoTagger CRF)
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

        # 1. Tokeniser via tokeniseur externe ou CRF
        raw_tokens = self._tokenize(phrase)
        tokens = [tok for tok, _is_word in raw_tokens]

        if not tokens:
            return ResultatCorrection(
                phrase_originale=phrase, phrase_corrigee=phrase,
            )

        all_corrections: list[Correction] = []

        # Save pre-syntaxe tokens for grammar originaux (before majuscule)
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

        # 4. Separer ponctuation des mots
        is_punct = [bool(PUNCT_RE.match(t)) for t in tokens]
        word_tokens = [t for t, p in zip(tokens, is_punct) if not p]
        word_tokens_orig = [t for t, p in zip(tokens_pre_syntaxe, is_punct) if not p]
        word_indices = [i for i, p in enumerate(is_punct) if not p]

        if not word_tokens:
            return ResultatCorrection(
                phrase_originale=phrase,
                phrase_corrigee=reconstruire_phrase(tokens),
            )

        # 4b. Analyse morpho (CRF ou G2P unifie)
        _rich_tagger = hasattr(self._tagger, "tag_words_rich")
        _dual_tagger = (
            self._config.activer_double_tagging
            and hasattr(self._tagger, "tag_words_dual")
        )
        if _dual_tagger:
            morpho_results = self._tagger.tag_words_dual(word_tokens)
        elif _rich_tagger:
            morpho_results = self._tagger.tag_words_rich(word_tokens)
        else:
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

        # Enrichir analyses avec les POS/morpho du tagger
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
                # Stocker confiance POS si disponible (rich tagger)
                if "pos_scores" in mr:
                    analysis.pos_scores = mr["pos_scores"]
                if "confiance_pos" in mr:
                    analysis.confiance_pos = mr["confiance_pos"]
                # Double tagging : pos_blind + divergence
                if "pos_blind" in mr:
                    analysis.pos_blind = mr["pos_blind"]
                if "divergence_pos" in mr:
                    analysis.divergence_pos = mr["divergence_pos"]

        # 4b-bis. Analyse Viterbi trigramme POS+forme
        if (
            self._config.activer_analyse_viterbi
            and self._pos_ngram is not None
        ):
            from lectura_correcteur._analyse_viterbi import analyse_viterbi
            viterbi_results = analyse_viterbi(
                word_tokens,
                self._lexique,
                self._pos_ngram,
                lm_homophones=self._lm_homophones,
                bonus_original=self._config.viterbi_bonus_original,
                bonus_lm=self._config.viterbi_bonus_lm,
                w_emission=self._config.viterbi_w_emission,
                w_transition=self._config.viterbi_w_transition,
            )
            for j, vr in enumerate(viterbi_results):
                if j >= len(analyses):
                    break
                # Mettre a jour POS
                if vr.pos:
                    analyses[j].pos = vr.pos
                    analyses[j].confiance_pos = vr.confiance
                # Appliquer correction de forme (accent/homophone)
                if vr.changed:
                    # Preserver la casse originale
                    new_forme = vr.forme
                    orig = analyses[j].original
                    if orig and orig[0].isupper() and new_forme:
                        new_forme = new_forme[0].upper() + new_forme[1:]
                    analyses[j].corrige = new_forme
                    analyses[j].type_correction = TypeCorrection.GRAMMAIRE

        # 4c. Viterbi POS disambiguation (si matrice presente + rich tagger)
        elif (
            self._config.activer_viterbi
            and self._transition_matrix is not None
            and _rich_tagger
        ):
            from lectura_correcteur._viterbi import viterbi_decode
            pos_scores_seq = [a.pos_scores for a in analyses]
            # N'appliquer Viterbi que si au moins une position a des alternatives
            if any(len(ps) > 1 for ps in pos_scores_seq):
                viterbi_result = viterbi_decode(
                    pos_scores_seq, self._transition_matrix,
                )
                for j, (v_pos, v_conf) in enumerate(viterbi_result):
                    if v_pos and j < len(analyses):
                        analyses[j].pos = v_pos
                        analyses[j].confiance_pos = v_conf

        # Fallback POS via lexique : corriger si le CRF s'est trompe
        # et que le lexique ne connait qu'un seul cgram pour ce mot
        for j, analysis in enumerate(analyses):
            if analysis.pos and self._lexique is not None:
                infos = self._lexique.info(analysis.corrige)
                if infos:
                    cgrams = {e["cgram"] for e in infos if e.get("cgram")}
                    if analysis.pos not in cgrams and len(cgrams) == 1:
                        analysis.pos = next(iter(cgrams))

        # 5b-6. Pipeline de correction (regles)
        after_rules, corrs = self._pipeline_regles(
            analyses, word_tokens, morpho_results, all_corrections,
            word_tokens_orig=word_tokens_orig,
        )
        all_corrections = corrs

        decided_words = [a.corrige for a in analyses]

        # 7. Reconstruction
        if len(after_rules) != len(decided_words):
            # Grammar rules inserted/removed tokens (e.g. negation "ne")
            # Rebuild by interleaving after_rules with punctuation
            final_tokens = _reconstruire_avec_insertions(
                tokens, is_punct, word_indices, after_rules, decided_words,
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
    # Pipeline : scoring heuristique + regles grammaticales
    # ------------------------------------------------------------------

    def _pipeline_regles(
        self,
        analyses: list[MotAnalyse],
        word_tokens: list[str],
        morpho_results: list[dict],
        corrections: list[Correction],
        word_tokens_orig: list[str] | None = None,
    ) -> tuple[list[str], list[Correction]]:
        """Pipeline : scoring heuristique 8 facteurs + regles grammaticales.

        Retourne (after_rules, corrections) pour la reconstruction.
        """
        all_corrections = list(corrections)

        # 5b. Scoring unifie (si active)
        if self._config.activer_scoring:
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

                    # Populate suggestions_scored with top-5 candidates
                    analysis.suggestions_scored = [
                        (c.forme, c.score) for c in scored[:5]
                    ]

        # 5c. Coherence POS (experimental, OFF par defaut)
        if self._config.activer_coherence:
            corr_coherence = appliquer_coherence(analyses, self._lexique)
            all_corrections.extend(corr_coherence)

        decided_words = [a.corrige for a in analyses]

        # 6. Grammaire
        if self._config.activer_grammaire:
            # Re-tag using corrected forms (not originals) so grammar rules
            # see the POS/morpho of the corrected word (e.g. "dan"->NOM
            # becomes "dans"->PRE after ortho correction).
            _retagger = self._lex_tagger if self._lex_tagger is not None else self._tagger
            _lex_tags = _retagger.tag_words(decided_words)
            _lex_morpho_results = _lex_tags
            pos_list = [t.get("pos", "") for t in _lex_tags]
            # Restreindre AUX aux seules formes d'etre/avoir
            from lectura_correcteur.grammaire._donnees import AUXILIAIRES as _AUX_FORMES
            for j in range(len(pos_list)):
                if (
                    pos_list[j] == "AUX"
                    and decided_words[j].lower() not in _AUX_FORMES
                ):
                    pos_list[j] = "VER"
                    _lex_tags[j]["pos"] = "VER"
            for j, tag in enumerate(_lex_tags):
                if j < len(analyses) and tag.get("pos"):
                    analyses[j].pos = tag["pos"]

            morpho_dict_lists: dict[str, list[str]] = {}
            _morpho_src = _lex_morpho_results if _lex_morpho_results is not None else None
            if analyses:
                for feat in ("genre", "nombre", "temps", "mode", "personne"):
                    if _morpho_src is not None:
                        morpho_dict_lists[feat] = [
                            _morpho_src[j].get(feat, "_") if j < len(_morpho_src) else "_"
                            for j in range(len(analyses))
                        ]
                    else:
                        morpho_dict_lists[feat] = [
                            a.morpho.get(feat, "_") for a in analyses
                        ]

            # Extraire confiance POS si disponible (rich tagger / Viterbi)
            _pos_conf = None
            if any(a.confiance_pos < 1.0 for a in analyses):
                _pos_conf = [a.confiance_pos for a in analyses]

            # 6-pre. Accord guide par PM n-gram
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

            after_rules, corr_gram = appliquer_grammaire(
                decided_words, pos_list, morpho_dict_lists, self._lexique,
                originaux=word_tokens_orig if word_tokens_orig else word_tokens,
                activer_negation=self._config.activer_negation,
                pos_confiance=_pos_conf,
                pm_guidance=_accord_guidance,
                activer_homophones_gram=getattr(self._config, "activer_homophones_gram", True),
                activer_accords=getattr(self._config, "activer_accords", True),
                activer_conjugaisons=getattr(self._config, "activer_conjugaisons", True),
                activer_participes=getattr(self._config, "activer_participes", True),
                activer_pp_etre=getattr(self._config, "activer_pp_etre", True),
            )
            all_corrections.extend(corr_gram)

            for j in range(len(analyses)):
                if j < len(after_rules) and after_rules[j].lower() != analyses[j].corrige.lower():
                    analyses[j].corrige = after_rules[j]
                    if analyses[j].type_correction == TypeCorrection.AUCUNE:
                        analyses[j].type_correction = TypeCorrection.GRAMMAIRE

            # 6b. Post-grammar scoring pass (PP apres AUX)
            if self._config.activer_scoring:
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

        # 6c-bis. LM trigramme specialise homophones (couvre tous les mots)
        if self._lm_homophones is not None:
            corr_lm_homo = self._verifier_homophones_trigram(analyses)
            all_corrections.extend(corr_lm_homo)
            for j in range(min(len(analyses), len(after_rules))):
                after_rules[j] = analyses[j].corrige

        # 6c. Detection confusions phonetiques par LM generique (si active)
        # Place apres les regles de grammaire pour ne pas interferer
        if self._lm is not None:
            corr_lm = self._verifier_homophones_lm(analyses)
            all_corrections.extend(corr_lm)
            for j in range(min(len(analyses), len(after_rules))):
                after_rules[j] = analyses[j].corrige

        # 6d. BiLSTM editeur homophones (si active)
        if self._editeur is not None:
            after_rules, corr_editeur = self._appliquer_editeur_homophones(
                after_rules, analyses, morpho_results, word_tokens,
            )
            all_corrections.extend(corr_editeur)
            for j in range(min(len(analyses), len(after_rules))):
                after_rules[j] = analyses[j].corrige

        # 6e. Post-validation POS n-gram : desactive car le tagger lexique
        # donne des POS erronees pour certains mots (soir=VER, gros=ADV),
        # ce qui cause des faux reverts sur des corrections legitimes.
        # Les guards inline (ratio 5x, POS n-gram local) suffisent.
        # TODO: reactiver quand le tagger est plus precis ou utiliser un
        # tagger contextuel (CRF/BiLSTM).

        # 6f. Viterbi POS+Morpho : validation/correction des traits morpho
        if (
            self._config.activer_viterbi_morpho
            and self._pos_ngram is not None
        ):
            from lectura_correcteur._viterbi_morpho import viterbi_morpho
            decided = [a.corrige for a in analyses]
            pos_tags = [a.pos for a in analyses]
            # Ne pas expanser les variantes pour les mots deja corriges
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
                    # Preserver la casse
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
                # Toujours mettre a jour POS et morpho depuis le Viterbi
                if vmr.pos:
                    analyses[j].pos = vmr.pos
                morpho = {}
                if vmr.genre != "_":
                    morpho["genre"] = vmr.genre.lower()[0] if vmr.genre in ("Masc", "Fem") else vmr.genre
                if vmr.nombre != "_":
                    morpho["nombre"] = vmr.nombre.lower()[0] if vmr.nombre in ("Sing", "Plur") else vmr.nombre
                if morpho:
                    analyses[j].morpho.update(morpho)

        # Couche coherence : re-verification post-corrections
        if self._config.activer_coherence:
            _retagger_coh = self._lex_tagger if self._lex_tagger is not None else self._tagger
            corr_coherence_post = verifier_coherence_post_corrections(
                analyses, self._lexique, _retagger_coh,
            )
            all_corrections.extend(corr_coherence_post)
            # Synchroniser after_rules avec les analyses mises a jour
            for j in range(min(len(analyses), len(after_rules))):
                after_rules[j] = analyses[j].corrige

        return after_rules, all_corrections


    def _verifier_homophones_trigram(
        self,
        analyses: list[MotAnalyse],
    ) -> list[Correction]:
        """Desambiguise les homophones via le LM trigramme specialise.

        Contrairement au LM ngram generique, ce LM peut toucher les
        homophones grammaticaux (a/à, est/et, etc.) car il est specialise
        et tres precis sur ces cas. Il n'intervient que si :
        - le mot n'a PAS deja ete corrige par une regle de grammaire
        - le LM trigramme propose une variante differente avec un score > 0
        - le score du candidat est strictement superieur au score du mot actuel

        Scoring conjoint : pour chaque homophone, on genere aussi des variantes
        accent du contexte (mot precedent/suivant) et on score toutes les
        combinaisons. Si la meilleure combinaison implique un changement
        d'homophone ET un changement d'accent sur le contexte, on applique
        les deux (ex: "on prepare" → "ont préparé").
        """
        corrections: list[Correction] = []
        lm = self._lm_homophones

        # Positions deja corrigees par scoring conjoint (ne pas re-traiter)
        _joint_corrected: set[int] = set()

        for j, analysis in enumerate(analyses):
            if j in _joint_corrected:
                continue

            # Ne pas re-corriger un mot deja touche par les regles
            if analysis.type_correction != TypeCorrection.AUCUNE:
                continue

            mot = analysis.corrige.lower()

            # Verifier si c'est un homophone connu
            if not lm.est_homophone(mot):
                continue

            # Contexte
            ctx_gauche = (
                analyses[j - 1].corrige if j > 0 else None
            )
            ctx_droite = (
                analyses[j + 1].corrige if j + 1 < len(analyses) else None
            )

            # --- Scoring conjoint : tester variantes accent du contexte ---
            homo_candidates = lm.candidats(mot)
            if homo_candidates and len(homo_candidates) >= 2:
                joint_result = self._scorer_conjoint(
                    j, mot, homo_candidates, analyses, lm,
                )
                if joint_result is not None:
                    best_homo, ctx_changes, total_score = joint_result
                    # Appliquer la correction homophone
                    if best_homo.lower() != mot:
                        _best_h = best_homo
                        if analysis.corrige[0].isupper():
                            _best_h = _best_h[0].upper() + _best_h[1:]
                        analysis.corrige = _best_h
                        analysis.type_correction = TypeCorrection.GRAMMAIRE
                        corrections.append(Correction(
                            index=j,
                            original=analysis.original,
                            corrige=_best_h,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="lm_homophones.trigram",
                            explication=(
                                f"Homophone trigram conjoint : '{mot}' -> '{best_homo}' "
                                f"(score conjoint {total_score})"
                            ),
                        ))
                    # Appliquer les corrections de contexte
                    for ctx_idx, ctx_new in ctx_changes:
                        ctx_analysis = analyses[ctx_idx]
                        if ctx_analysis.corrige[0].isupper():
                            ctx_new = ctx_new[0].upper() + ctx_new[1:]
                        ctx_analysis.corrige = ctx_new
                        if ctx_analysis.type_correction == TypeCorrection.AUCUNE:
                            ctx_analysis.type_correction = TypeCorrection.GRAMMAIRE
                        _joint_corrected.add(ctx_idx)
                        corrections.append(Correction(
                            index=ctx_idx,
                            original=ctx_analysis.original,
                            corrige=ctx_new,
                            type_correction=TypeCorrection.GRAMMAIRE,
                            regle="lm_homophones.conjoint",
                            explication=(
                                f"Correction conjointe accent : "
                                f"'{ctx_analysis.original}' -> '{ctx_new}'"
                            ),
                        ))
                    continue

            # --- Scoring simple (fallback) ---
            best, source = lm.meilleur_homophone(mot, ctx_gauche, ctx_droite)

            if source == "PASS" or best.lower() == mot:
                continue

            # Verifier que le LM est strictement meilleur pour le candidat
            score_best = lm.scorer(best, ctx_gauche, ctx_droite)
            score_current = lm.scorer(mot, ctx_gauche, ctx_droite)

            if score_best <= score_current:
                continue

            # Guard ambiguite : quand les deux formes sont vues dans ce
            # contexte (scores tous deux > 0), exiger un ratio suffisant
            # pour eviter les faux positifs sur cas ambigus (ce/se sont)
            _ratio = self._config.lm_homophones_ratio
            if score_current > 0 and score_best < score_current * _ratio:
                continue

            # Guard a/à + PP : ne pas changer "a" → "à" si le mot suivant
            # est un participe passe (pattern "a + PP" = auxiliaire avoir)
            if mot == "a" and best.lower() == "\xe0":
                next_low = ctx_droite.lower() if ctx_droite else ""
                if next_low and self._est_participe_passe(next_low):
                    continue

            # Guard POS n-gram : verifier que la correction ameliore
            # (ou au moins ne degrade pas) la sequence POS
            if self._pos_ngram is not None:
                pos_tags = [a.pos for a in analyses]
                # POS du candidat : lookup dans le lexique
                best_infos = self._lexique.info(best.lower())
                if best_infos:
                    best_pos = max(
                        best_infos,
                        key=lambda e: float(e.get("freq") or 0),
                    ).get("cgram", "")
                    if best_pos and best_pos != analysis.pos:
                        score_pos_current = self._pos_ngram.score_position(
                            pos_tags, j, analysis.pos,
                        )
                        score_pos_best = self._pos_ngram.score_position(
                            pos_tags, j, best_pos,
                        )
                        # Bloquer si le POS n-gram dit que la correction
                        # degrade la sequence POS (avec marge)
                        if score_pos_best < score_pos_current - 0.5:
                            continue

            # Preserver la casse
            if analysis.corrige[0].isupper():
                best = best[0].upper() + best[1:]

            analysis.corrige = best
            analysis.type_correction = TypeCorrection.GRAMMAIRE
            corrections.append(Correction(
                index=j,
                original=analysis.original,
                corrige=best,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="lm_homophones.trigram",
                explication=(
                    f"Homophone trigram : '{mot}' -> '{best}' "
                    f"(score {score_best} vs {score_current})"
                ),
            ))

        return corrections

    def _scorer_conjoint(
        self,
        j: int,
        mot: str,
        homo_candidates: list[tuple[str, float]],
        analyses: list[MotAnalyse],
        lm,
    ) -> tuple[str, list[tuple[int, str]], int] | None:
        """Score conjoint : homophone × variantes accent du contexte.

        Retourne (best_homo, [(ctx_idx, ctx_new), ...], score) si une
        combinaison conjointe gagne, None sinon.
        """
        # Generer les variantes accent pour le contexte gauche et droit
        # Ne pas generer de variantes qui sont des homophones grammaticaux
        # (ex: "a" → "à" doit etre gere par les regles, pas par le scoring)
        _grammar_homos = self._GRAMMAR_HOMOPHONES
        ctx_left_variants: list[str] = []
        ctx_right_variants: list[str] = []

        if j > 0:
            left_word = analyses[j - 1].corrige.lower()
            ctx_left_variants = [left_word]
            # Ne generer des variantes que si le mot de contexte n'a pas
            # deja ete corrige et n'est pas un homophone grammatical
            if (
                analyses[j - 1].type_correction == TypeCorrection.AUCUNE
                and left_word not in _grammar_homos
            ):
                for forme, _freq in _variantes_accents(left_word, self._lexique):
                    if forme not in ctx_left_variants and forme not in _grammar_homos:
                        ctx_left_variants.append(forme)
        else:
            ctx_left_variants = [None]

        if j + 1 < len(analyses):
            right_word = analyses[j + 1].corrige.lower()
            ctx_right_variants = [right_word]
            if (
                analyses[j + 1].type_correction == TypeCorrection.AUCUNE
                and right_word not in _grammar_homos
            ):
                for forme, _freq in _variantes_accents(right_word, self._lexique):
                    if forme not in ctx_right_variants and forme not in _grammar_homos:
                        ctx_right_variants.append(forme)
        else:
            ctx_right_variants = [None]

        # Si aucun contexte n'a de variantes accent, pas besoin de scoring conjoint
        has_left_variants = len(ctx_left_variants) > 1
        has_right_variants = len(ctx_right_variants) > 1
        if not has_left_variants and not has_right_variants:
            return None

        # Scorer toutes les combinaisons
        best_score = -1
        best_homo = mot
        best_ctx_left = ctx_left_variants[0]
        best_ctx_right = ctx_right_variants[0]

        for ortho, _freq in homo_candidates:
            ortho_low = ortho.lower()
            # Filtre same_lemma (coherent avec meilleur_homophone)
            if (mot, ortho_low) in lm._same_lemma_pairs:
                continue
            for cl in ctx_left_variants:
                for cr in ctx_right_variants:
                    score = lm.scorer(ortho_low, cl, cr)
                    if score > best_score:
                        best_score = score
                        best_homo = ortho_low
                        best_ctx_left = cl
                        best_ctx_right = cr

        if best_score <= 0:
            return None

        # Score du mot actuel avec contexte actuel
        baseline_score = lm.scorer(
            mot,
            ctx_left_variants[0],
            ctx_right_variants[0],
        )

        if best_score <= baseline_score:
            return None

        # Determiner les changements
        has_homo_change = (best_homo != mot)
        ctx_changes: list[tuple[int, str]] = []

        if j > 0 and best_ctx_left != ctx_left_variants[0]:
            ctx_changes.append((j - 1, best_ctx_left))
        if j + 1 < len(analyses) and best_ctx_right != ctx_right_variants[0]:
            ctx_changes.append((j + 1, best_ctx_right))

        # Ne retourner que si au moins un changement (homophone ou contexte)
        if not has_homo_change and not ctx_changes:
            return None

        # Guard ambiguite conjoint : quand il n'y a pas de changement
        # de contexte (le gain vient uniquement du changement d'homophone),
        # appliquer le meme seuil de ratio que le scoring simple (5x)
        if not ctx_changes and baseline_score > 0:
            if best_score < baseline_score * 5:
                return None

        # Guard a/à + PP conjoint : ne pas changer "a" → "à" si le
        # contexte droit (potentiellement corrige) est un participe passe
        if mot == "a" and best_homo == "\xe0":
            right_check = best_ctx_right if best_ctx_right else ""
            if isinstance(right_check, str) and right_check and self._est_participe_passe(right_check):
                return None

        # Guard POS n-gram conjoint : verifier que les changements de
        # contexte n'empirent pas la sequence POS
        if self._pos_ngram is not None and ctx_changes:
            pos_tags = [a.pos for a in analyses]
            for ctx_idx, ctx_new in ctx_changes:
                new_infos = self._lexique.info(ctx_new)
                if new_infos:
                    new_pos = max(
                        new_infos,
                        key=lambda e: float(e.get("freq") or 0),
                    ).get("cgram", "")
                    old_pos = analyses[ctx_idx].pos
                    if new_pos and old_pos and new_pos != old_pos:
                        score_old = self._pos_ngram.score_position(
                            pos_tags, ctx_idx, old_pos,
                        )
                        score_new = self._pos_ngram.score_position(
                            pos_tags, ctx_idx, new_pos,
                        )
                        if score_new < score_old - 0.5:
                            return None

        return best_homo, ctx_changes, best_score

    def _est_participe_passe(self, mot: str) -> bool:
        """Verifie si un mot est un participe passe via le lexique.

        Retourne True si le mot a au moins une entree VER avec un suffixe
        typique de PP (-é, -i, -u, -it, -is, etc.) et que le lexique le
        confirme comme verbe.
        """
        mot_low = mot.lower()
        _PP_SUFFIXES = (
            "é", "és", "ée", "ées",
            "i", "is", "ie", "ies",
            "u", "us", "ue", "ues",
            "it", "ite", "ites", "its",
        )
        if not mot_low.endswith(_PP_SUFFIXES):
            return False
        # Exclure les infinitifs et formes non-PP
        if mot_low.endswith(("er", "ir", "re", "oir")):
            return False
        infos = self._lexique.info(mot_low)
        if not infos:
            return False
        for entry in infos:
            cgram = entry.get("cgram", "")
            if cgram in ("VER", "AUX"):
                return True
        return False

    def _post_validation_pos_ngram(
        self,
        analyses: list[MotAnalyse],
        after_rules: list[str],
        corrections: list[Correction],
    ) -> list[int]:
        """Post-validation : revertir les corrections qui degradent le POS n-gram.

        Pour chaque correction grammaticale, retague la phrase avec et sans
        la correction et compare les scores POS des deux sequences. Si
        l'original donne un meilleur score, on revertit.

        Returns:
            Liste des indices revertes.
        """
        reverted: list[int] = []

        # Identifier les positions corrigees par les regles de grammaire
        candidates: list[int] = []
        for j, a in enumerate(analyses):
            if (
                a.type_correction == TypeCorrection.GRAMMAIRE
                and a.corrige.lower() != a.original.lower()
            ):
                candidates.append(j)

        if not candidates:
            return reverted

        _retagger = self._lex_tagger if self._lex_tagger is not None else self._tagger

        # Retagger la phrase avec les corrections
        corrected_words = [a.corrige for a in analyses]
        tags_corrected = _retagger.tag_words(corrected_words)
        pos_corrected = [t.get("pos", "") for t in tags_corrected]

        for j in candidates:
            # Construire la phrase avec le mot original a la position j
            original_words = list(corrected_words)
            original_words[j] = analyses[j].original

            # Retagger avec le mot original
            tags_original = _retagger.tag_words(original_words)
            pos_original = [t.get("pos", "") for t in tags_original]

            # Scorer les deux sequences POS completes
            score_corrected = self._pos_ngram.score_sequence(pos_corrected)
            score_original = self._pos_ngram.score_sequence(pos_original)

            # Si l'original a un meilleur score POS → revertir
            if score_original > score_corrected + 0.5:
                analyses[j].corrige = analyses[j].original
                analyses[j].type_correction = TypeCorrection.AUCUNE
                if j < len(after_rules):
                    after_rules[j] = analyses[j].original
                # Mettre a jour corrected_words et pos_corrected pour les
                # candidats suivants
                corrected_words[j] = analyses[j].original
                tags_corrected = _retagger.tag_words(corrected_words)
                pos_corrected = [t.get("pos", "") for t in tags_corrected]
                reverted.append(j)

        return reverted

    # Paires d'homophones deja traitees par les regles de grammaire.
    # Le LM ne doit pas toucher ces mots (les regles sont plus precises).
    _GRAMMAR_HOMOPHONES = frozenset({
        "et", "est",
        "son", "sont",
        "a", "à",
        "ou", "où",
        "on", "ont",
        "ce", "se",
        "la", "là",
        "leur", "leurs",
        "ça", "sa",
        "ces", "ses",
        "peu", "peut", "peux",
        "ma", "m'a",
        "ta", "t'a",
        "dans", "d'en",
        "sans", "s'en",
        "si", "s'y",
        "ni", "n'y",
        "mais", "mes",
    })

    # Mots-outils courts que le LM n'a pas le droit de changer
    # (trop de faux positifs sur pronoms/determinants).
    _MOTS_OUTILS = frozenset({
        "il", "ils", "elle", "elles", "on", "nous", "vous",
        "le", "la", "les", "de", "des", "du", "un", "une",
        "je", "tu", "me", "te", "ne", "se", "ce",
        "en", "y", "au", "aux", "que", "qui", "dont",
    })

    def _verifier_homophones_lm(
        self,
        analyses: list[MotAnalyse],
    ) -> list[Correction]:
        """Detecte les confusions phonetiques in-lexique via le LM.

        Pour chaque mot in-lexique NON deja corrige par les regles de
        grammaire, cherche les homophones et utilise le LM pour
        determiner si un homophone serait plus adapte au contexte.

        Ne touche pas aux homophones grammaticaux (geres par les regles)
        ni aux mots-outils (trop de FP).
        """
        corrections: list[Correction] = []

        for j, analysis in enumerate(analyses):
            # Ne traiter que les mots non deja corriges
            if analysis.type_correction != TypeCorrection.AUCUNE:
                continue
            if not analysis.dans_lexique:
                continue

            mot = analysis.corrige.lower()

            # Ne pas toucher aux homophones grammaticaux ni mots-outils
            if mot in self._GRAMMAR_HOMOPHONES:
                continue
            if mot in self._MOTS_OUTILS:
                continue

            # Obtenir la prononciation
            phone = self._lexique.phone_de(mot)
            if not phone:
                continue

            # Trouver les homophones
            homos = self._lexique.homophones(phone)
            if not homos:
                continue

            # Filtrer : garder seulement les formes frequentes, distinctes,
            # et qui ne sont pas des homophones grammaticaux/mots-outils
            candidats = [mot]
            for h in homos:
                forme = h.get("ortho", "").lower()
                freq = h.get("freq", 0.0) or 0.0
                if forme == mot:
                    continue
                if freq < 1.0:
                    continue
                if forme in self._GRAMMAR_HOMOPHONES:
                    continue
                if forme in self._MOTS_OUTILS:
                    continue
                if forme not in candidats:
                    candidats.append(forme)

            if len(candidats) < 2:
                continue

            # Construire le contexte
            ctx_gauche = [
                analyses[k].corrige.lower()
                for k in range(max(0, j - 3), j)
            ]
            ctx_droite = [
                analyses[k].corrige.lower()
                for k in range(j + 1, min(len(analyses), j + 2))
            ]

            scored = self._lm.scorer_candidats(candidats, ctx_gauche, ctx_droite)
            if not scored:
                continue

            best_form, best_score = scored[0]
            if best_form == mot:
                continue  # LM confirme le mot actuel

            # Score du mot actuel
            current_score = next(
                (s for f, s in scored if f == mot), -999.0,
            )
            # Marge requise : le LM doit etre nettement meilleur (log10)
            # 4.0 = conservative (haute precision), baisser pour plus de recall
            delta = best_score - current_score
            if delta < 4.0:
                continue

            # Verification supplementaire : l'homophone choisi doit etre
            # au moins aussi frequent que le mot actuel
            freq_actuel = (
                self._lexique.frequence(mot)
                if hasattr(self._lexique, "frequence") else 0.0
            )
            freq_best = (
                self._lexique.frequence(best_form)
                if hasattr(self._lexique, "frequence") else 0.0
            )
            if freq_best < freq_actuel * 0.1:
                continue

            # Appliquer
            analysis.corrige = best_form
            analysis.type_correction = TypeCorrection.GRAMMAIRE
            corrections.append(Correction(
                index=j,
                original=analysis.original,
                corrige=best_form,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="lm.homophone",
                explication=(
                    f"Homophone LM : '{mot}' -> '{best_form}' "
                    f"(delta={delta:+.2f})"
                ),
            ))

        return corrections

    def _appliquer_editeur_homophones(
        self,
        after_rules: list[str],
        analyses: list[MotAnalyse],
        morpho_results: list[dict],
        word_tokens: list[str],
    ) -> tuple[list[str], list[Correction]]:
        """Applique le BiLSTM editeur pour les homophones.

        Strategie :
          - On execute le BiLSTM sur les tokens originaux (avant regles).
          - Pour chaque position ou le BiLSTM predit un HOMO_* avec
            confiance >= seuil, on applique la correction, sauf si les
            regles ont deja corrige vers la meme cible.
          - Si les regles et le BiLSTM sont en desaccord, le BiLSTM gagne
            uniquement si sa confiance est elevee (>= seuil).
        """
        from lectura_correcteur._tags import (
            KEEP,
            TAG2IDX,
            _TAG_TO_HOMO,
            _preserve_case,
        )

        corrections: list[Correction] = []
        seuil = self._config.seuil_editeur

        # Predire sur les tokens originaux avec leur morpho
        tags_scores = self._editeur.predire_tags_avec_scores(
            word_tokens, morpho_results,
        )

        for j, (model_tag, score) in enumerate(tags_scores):
            if j >= len(analyses):
                break

            canon_tag = model_tag if model_tag in TAG2IDX else KEEP

            # Ne garder que les HOMO_* avec confiance suffisante
            if not canon_tag.startswith("HOMO_") or canon_tag == KEEP:
                continue
            if score < seuil:
                continue

            forme_cible = _TAG_TO_HOMO.get(canon_tag)
            if forme_cible is None:
                continue

            mot_original = word_tokens[j]
            mot_apres_regles = after_rules[j].lower()

            if forme_cible.lower() == mot_apres_regles:
                continue  # Les regles ont deja produit la bonne forme

            if forme_cible.lower() == mot_original.lower():
                continue  # Le BiLSTM dit de garder la forme originale

            # Guard: apres apostrophe, "est" est toujours correct
            # (c'est, l'est, n'est, s'est, qu'est → jamais "et")
            if (
                forme_cible == "et"
                and mot_original.lower() == "est"
                and j > 0
                and word_tokens[j - 1].endswith("'")
            ):
                continue

            # Appliquer la correction avec preservation de la casse
            forme_finale = _preserve_case(mot_original, forme_cible)
            after_rules[j] = forme_finale
            analyses[j].corrige = forme_finale
            if analyses[j].type_correction == TypeCorrection.AUCUNE:
                analyses[j].type_correction = TypeCorrection.GRAMMAIRE

            corrections.append(Correction(
                index=j,
                original=analyses[j].original,
                corrige=forme_finale,
                type_correction=TypeCorrection.GRAMMAIRE,
                regle="editeur.homophone",
                explication=(
                    f"Homophone BiLSTM : '{mot_original.lower()}' "
                    f"-> '{forme_cible}' ({score:.0%})"
                ),
            ))

        return after_rules, corrections


def _reconstruire_avec_insertions(
    tokens: list[str],
    is_punct: list[bool],
    word_indices: list[int],
    after_rules: list[str],
    decided_words: list[str],
) -> list[str]:
    """Reconstruit les tokens quand la grammaire a insere/supprime des mots.

    Uses SequenceMatcher to align after_rules to decided_words, then rebuilds
    the token list by placing new/changed words at the right positions.
    """
    from difflib import SequenceMatcher

    # Align after_rules to decided_words to find insertions
    sm = SequenceMatcher(None, decided_words, after_rules)
    opcodes = sm.get_opcodes()

    # Build mapping: for each original word index, what words replace it
    # word_replacements[k] = list of words that replace decided_words[k]
    word_replacements: dict[int, list[str]] = {k: [] for k in range(len(decided_words))}
    # Words inserted before word k
    word_insertions_before: dict[int, list[str]] = {k: [] for k in range(len(decided_words) + 1)}

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for offset in range(i2 - i1):
                word_replacements[i1 + offset] = [after_rules[j1 + offset]]
        elif tag == "replace":
            # Map replaced words
            for offset in range(min(i2 - i1, j2 - j1)):
                word_replacements[i1 + offset] = [after_rules[j1 + offset]]
            # Extra words in after_rules -> insertions after last replaced
            for extra in range(min(i2 - i1, j2 - j1), j2 - j1):
                word_insertions_before.setdefault(i2, []).append(after_rules[j1 + extra])
        elif tag == "insert":
            # New words inserted before position i1
            for offset in range(j2 - j1):
                word_insertions_before.setdefault(i1, []).append(after_rules[j1 + offset])
        elif tag == "delete":
            # Words removed (shouldn't normally happen)
            pass

    # Now rebuild final_tokens
    result: list[str] = []
    word_k = 0  # current original word index

    for i, tok in enumerate(tokens):
        if is_punct[i]:
            result.append(tok)
        else:
            # Emit any insertions before this word
            for w in word_insertions_before.get(word_k, []):
                result.append(w)
            # Emit replacement(s) for this word
            replacements = word_replacements.get(word_k)
            if replacements:
                result.extend(replacements)
            else:
                result.append(tok)
            word_k += 1

    # Emit any trailing insertions (after all original words)
    for w in word_insertions_before.get(len(decided_words), []):
        result.append(w)

    return result


def _adapter_tokens(tokens) -> list[tuple[str, bool]]:
    """Convertit une liste de Token (lectura-tokeniseur) en (text, is_word)."""
    result = []
    for tok in tokens:
        tok_type = getattr(tok, "type", None)
        type_val = tok_type.value if hasattr(tok_type, "value") else str(tok_type)
        if type_val == "separateur" and getattr(tok, "sep_type", "") == "space":
            continue
        is_word = type_val == "mot"
        result.append((tok.text, is_word))
    return result
