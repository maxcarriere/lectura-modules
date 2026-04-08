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

from lectura_correcteur._candidats import generer_candidats
from lectura_correcteur._coherence import appliquer_coherence
from lectura_correcteur._config import CorrecteurConfig
from lectura_correcteur._morpho import MorphoTagger
from lectura_correcteur._scoring import extraire_contexte, scorer_candidats
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

    Utilise un MorphoTagger CRF embarque pour l'analyse POS/morpho,
    sauf si un tagger/tokeniseur externe est injecte.
    Pas de G2P, pas de P2G, pas de lexique embarque.
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
            tagger: Objet satisfaisant TaggerProtocol (optionnel, fallback MorphoTagger)
            tokeniseur: Objet satisfaisant TokeniseurProtocol (optionnel)
            g2p: Objet optionnel avec methode prononcer(mot) -> str | None
        """
        self._lexique = lexique
        self._config = config or CorrecteurConfig()
        self._tagger = tagger if tagger is not None else MorphoTagger()
        self._tokeniseur = tokeniseur
        self._g2p = g2p
        self._verificateur = VerificateurOrthographe(
            lexique, max_suggestions=self._config.max_suggestions,
            distance=self._config.distance_suggestions,
            g2p=g2p,
            scoring_actif=self._config.activer_scoring,
        )

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

        # Fallback POS via lexique : corriger si le CRF s'est trompe
        # et que le lexique ne connait qu'un seul cgram pour ce mot
        for j, analysis in enumerate(analyses):
            if analysis.pos and self._lexique is not None:
                infos = self._lexique.info(analysis.corrige)
                if infos:
                    cgrams = {e["cgram"] for e in infos if e.get("cgram")}
                    if analysis.pos not in cgrams and len(cgrams) == 1:
                        analysis.pos = next(iter(cgrams))

        # 5b. Scoring unifie (si active)
        if self._config.activer_scoring:
            for j, analysis in enumerate(analyses):
                # Si l'orthographe a corrige le mot, verifier si la forme
                # corrigee est dans le lexique (pas l'originale)
                dans_lex = analysis.dans_lexique
                if analysis.corrige != analysis.original and not dans_lex:
                    dans_lex = self._lexique.existe(analysis.corrige)
                # Injecter les suggestions du verificateur pour les OOV
                sugg = analysis.suggestions if not dans_lex else None
                candidats = generer_candidats(
                    analysis.corrige, dans_lex,
                    analysis.pos, analysis.morpho,
                    self._lexique, self._g2p, self._config,
                    suggestions=sugg,
                )
                contexte = extraire_contexte(analyses, j, self._lexique)
                scored = scorer_candidats(
                    candidats, analysis.original,
                    analysis.pos, analysis.morpho,
                    contexte, self._lexique,
                    dans_lexique=dans_lex,
                )
                if scored:
                    top = scored[0]
                    identite = next(
                        (c for c in scored if c.source == "identite"), None,
                    )
                    # OOV : seuil=0 (on sait que le mot est faux)
                    seuil = 0.0 if not dans_lex else self._config.seuil_remplacement
                    if identite is None or (top.score - identite.score) > seuil:
                        analysis.corrige = top.forme
                        if top.source == "morpho":
                            analysis.type_correction = TypeCorrection.GRAMMAIRE
                        elif top.source != "identite":
                            analysis.type_correction = TypeCorrection.HORS_LEXIQUE

        # 5c. Coherence POS (experimental, OFF par defaut)
        if self._config.activer_coherence:
            corr_coherence = appliquer_coherence(analyses, self._lexique)
            all_corrections.extend(corr_coherence)

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

            # 6b. Post-grammar scoring pass (PP apres AUX)
            # La grammaire peut corriger des homophones (a/à, on/ont) qui revelent
            # un contexte AUX que le scoring initial ne pouvait pas detecter.
            if self._config.activer_scoring:
                # Mettre a jour les POS apres corrections grammaticales
                for j, analysis in enumerate(analyses):
                    if j < len(after_rules):
                        forme_courante = after_rules[j].lower()
                        infos = self._lexique.info(forme_courante)
                        if infos:
                            cgrams = {e["cgram"] for e in infos if e.get("cgram")}
                            # Si la grammaire a change le mot, mettre a jour le POS
                            if forme_courante != analysis.original.lower() and len(cgrams) == 1:
                                analysis.pos = next(iter(cgrams))
                # Re-run scoring sur les mots non deja corriges
                for j, analysis in enumerate(analyses):
                    if analysis.type_correction != TypeCorrection.AUCUNE:
                        continue  # deja corrige, ne pas re-scorer
                    contexte = extraire_contexte(analyses, j, self._lexique)
                    if not contexte.get("aux_gauche"):
                        continue  # seul interet du post-pass
                    candidats = generer_candidats(
                        analysis.corrige, analysis.dans_lexique,
                        analysis.pos, analysis.morpho,
                        self._lexique, self._g2p, self._config,
                    )
                    scored = scorer_candidats(
                        candidats, analysis.original,
                        analysis.pos, analysis.morpho,
                        contexte, self._lexique,
                    )
                    if scored:
                        top = scored[0]
                        identite = next(
                            (c for c in scored if c.source == "identite"), None,
                        )
                        seuil = self._config.seuil_remplacement
                        if identite is None or (top.score - identite.score) > seuil:
                            analysis.corrige = top.forme
                            after_rules[j] = top.forme
                            if top.source == "morpho":
                                analysis.type_correction = TypeCorrection.GRAMMAIRE
                            elif top.source != "identite":
                                analysis.type_correction = TypeCorrection.HORS_LEXIQUE
        else:
            after_rules = decided_words

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
