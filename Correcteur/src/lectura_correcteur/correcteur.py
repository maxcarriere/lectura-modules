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
from lectura_correcteur._symspell import SymSpellIndex, _obtenir_formes
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
        self._lex_tagger = (
            LexiqueTagger(self._lexique) if tagger is not None else None
        )
        formes = _obtenir_formes(self._lexique)
        self._symspell = SymSpellIndex(formes) if formes is not None else None
        self._verificateur = VerificateurOrthographe(
            self._lexique, max_suggestions=self._config.max_suggestions,
            distance=self._config.distance_suggestions,
            g2p=self._g2p,
            scoring_actif=self._config.activer_scoring,
            symspell=self._symspell,
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

        # 4c. Viterbi POS disambiguation (si matrice presente + rich tagger)
        if (
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

            after_rules, corr_gram = appliquer_grammaire(
                decided_words, pos_list, morpho_dict_lists, self._lexique,
                originaux=word_tokens_orig if word_tokens_orig else word_tokens,
                activer_negation=self._config.activer_negation,
                pos_confiance=_pos_conf,
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

        # 6c. Detection confusions phonetiques par LM (si active)
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
