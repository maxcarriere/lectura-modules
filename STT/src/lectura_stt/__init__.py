"""Lectura STT — Pipeline STT complet du francais (audio -> texte).

Chaine le decodeur CTC (audio -> phones IPA) avec le pipeline P2G
(phones -> orthographe) pour produire du texte francais.

Copyright (C) 2025-2026 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md

Exemple rapide (mode API, zero config)::

    from lectura_stt import creer_engine
    engine = creer_engine()
    result = engine.transcrire(audio)
    print(result.ipa)    # "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d ."
    print(result.texte)  # "Bonjour le monde."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lectura_decodeur import parse_ctc_output, parse_ctc_v2, ParseResult
from lectura_stt._assembler import assembler_texte, rejoin_elisions
from lectura_stt._segmentation import (
    strip_liaisons,
    split_elisions,
    split_merged_words,
)
from lectura_stt._postprocess import (
    merge_and_rescore,
    try_elision_merges,
    _shift_compound_joins,
)
from lectura_stt._correction import (
    map_tokens_to_words,
    correct_doubtful_words,
)
from lectura_stt._grammar import GrammarLookup, corriger_grammatical

__version__ = "3.2.1"


@dataclass
class ResultatSTT:
    """Resultat d'une transcription STT.

    Attributes
    ----------
    ipa : str
        Chaine IPA brute du decodeur CTC.
        Ex: "b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d ."
    mots_ipa : list[str]
        Mots IPA extraits (phones concatenes par mot).
        Ex: ["bɔ̃ʒuʁ", "lə", "mɔ̃d"]
    texte : str | None
        Texte orthographique reconstruit (None si P2G absent).
        Ex: "Bonjour le monde."
    mots : list[str] | None
        Mots orthographiques individuels (None si P2G absent).
        Ex: ["Bonjour", "le", "monde"]
    ponctuation : list[str]
        Ponctuation detectee.
        Ex: ["."]
    """

    ipa: str
    mots_ipa: list[str]
    texte: str | None
    mots: list[str] | None
    ponctuation: list[str]
    ctc_phones: list[dict] | None = None  # CTC timestamps par phone
    # Chaque dict : {"phone": str, "phone_id": int, "frame": int,
    #                "time_s": float, "confidence": float, "entropy": float}


class STTEngine:
    """Moteur STT : audio → ResultatSTT.

    Combine un engine CTC (transcription phonetique) et un engine P2G optionnel
    (conversion phonetique → orthographe).

    Si un PhoneLexicon est fourni, utilise le pipeline optimal avec
    postprocessing CTC (strip_liaisons, split_elisions, split_merged_words,
    merge_and_rescore, try_elision_merges). Sinon utilise le pipeline
    simplifie (parse_ctc_output → P2G → assembler_texte).
    """

    def __init__(
        self,
        ctc_engine: object,
        p2g_engine: object | None = None,
        p2g_analyser: object | None = None,
        formule_tolerance: str = "stt",
        phone_lexicon: object | None = None,
        phone_correct: bool = False,
        phone_conf_threshold: float = 0.98,
        denoiser: object | None = None,
        number_mode: str = "auto",
        grammar_lookup: GrammarLookup | None = None,
    ) -> None:
        self.ctc = ctc_engine
        self.p2g = p2g_engine
        self._p2g_analyser = p2g_analyser  # lectura_p2g.analyser (avec formules)
        self.formule_tolerance = formule_tolerance
        self.phone_lexicon = phone_lexicon
        self.phone_correct = phone_correct
        self.phone_conf_threshold = phone_conf_threshold
        self.denoiser = denoiser  # CTCDenoiser optionnel (corrige phones avant P2G)
        self.number_mode = number_mode
        self.grammar_lookup = grammar_lookup
        self._subsample_factor: int | None = None  # cache

    def _detect_subsample(self) -> int:
        """Detecte le facteur de subsampling du modele CTC.

        Utilise un court signal probe (1s de silence) pour comparer
        le nombre de frames mel au nombre de frames de sortie.
        """
        if self._subsample_factor is not None:
            return self._subsample_factor

        try:
            from lectura_decodeur._mel import mel_spectrogram
            probe = np.zeros(16000, dtype=np.float32)
            mel = mel_spectrogram(probe, 16000)
            T_mel = mel.shape[-1]
            logits = self.ctc.session.run(None, {"mel": mel})[0]
            T_out = logits.shape[1]
            factor = max(1, round(T_mel / T_out))
        except Exception:
            factor = 4  # defaut conservateur

        self._subsample_factor = factor
        return factor

    def transcrire(
        self, audio: np.ndarray, sr: int = 16000,
        stt_mode: str | None = None,
        with_timestamps: bool = False,
    ) -> ResultatSTT:
        """Transcrit un signal audio en texte.

        Parameters
        ----------
        audio : np.ndarray
            Signal audio PCM float32 mono.
        sr : int
            Sample rate (defaut 16000).
        with_timestamps : bool
            Si True, force transcrire_avec_alternatives() pour obtenir
            les timestamps CTC frame-level et les attache au ResultatSTT.

        Returns
        -------
        ResultatSTT
            Resultat contenant IPA et texte (si P2G disponible).
        """
        # Etape 1 : CTC → IPA (avec alternatives si phone_correct actif ou timestamps demandes)
        ctc_tokens = None
        if ((with_timestamps or (self.phone_correct and self.phone_lexicon is not None))
                and hasattr(self.ctc, "transcrire_avec_alternatives")):
            ipa_str, ctc_tokens = self.ctc.transcrire_avec_alternatives(
                audio, sr=sr, top_k=5,
            )
        else:
            ipa_str = self.ctc.transcrire(audio, sr=sr)

        # Enrichir timestamps (frame_end ajouté par le décodeur)
        # Le modèle CTC a un subsampling 4× sur le mel (hop=160),
        # donc chaque frame de sortie = 4 * 160 / 16000 = 0.04s
        if ctc_tokens is not None:
            subsample = self._detect_subsample()
            frame_duration = subsample * 160 / 16000  # secondes par frame
            for t in ctc_tokens:
                t["time_s"] = t["frame"] * frame_duration
                t["time_end_s"] = (t.get("frame_end", t["frame"]) + 1) * frame_duration

        # Choix du pipeline : optimal si PhoneLexicon disponible, sinon simplifie
        if self.phone_lexicon is not None and self.p2g is not None:
            result = self._transcrire_optimal(ipa_str, ctc_tokens=ctc_tokens, stt_mode=stt_mode)
        else:
            result = self._transcrire_simple(ipa_str, stt_mode=stt_mode)

        # Attacher les timestamps CTC au resultat
        if with_timestamps and ctc_tokens is not None:
            result.ctc_phones = ctc_tokens

        return result

    def _transcrire_simple(self, ipa_str: str, stt_mode: str | None = None) -> ResultatSTT:
        """Pipeline simplifie (sans PhoneLexicon)."""
        parsed = parse_ctc_output(ipa_str)

        texte: str | None = None
        mots_ortho: list[str] | None = None

        if self.p2g is not None and parsed.mots_ipa:
            mots_ortho = self._p2g_convertir(parsed.mots_ipa, stt_mode=stt_mode)
            texte = assembler_texte(mots_ortho, parsed.ponctuation_finale)

        ponctuation = [parsed.ponctuation_finale] if parsed.ponctuation_finale else []

        return ResultatSTT(
            ipa=ipa_str,
            mots_ipa=parsed.mots_ipa,
            texte=texte,
            mots=mots_ortho,
            ponctuation=ponctuation,
        )

    def _transcrire_optimal(
        self, ipa_str: str, ctc_tokens: list[dict] | None = None,
        stt_mode: str | None = None,
    ) -> ResultatSTT:
        """Pipeline optimal avec postprocessing CTC.

        Sequence :
            parse_ctc_v2 → extract word segments + compound_joins
            → strip_liaisons(ipa_words, lexicon)
            → split_elisions(ipa_words, lexicon)
            → split_merged_words(ipa_words, lexicon)
            → correct_doubtful_words(ipa_words, word_stats) [si phone_correct]
            → P2G analyser_v2(ipa_words) avec lex_select
            → merge_and_rescore(ipa, ortho, pos_conf, lexicon, p2g)
            → try_elision_merges(ipa, ortho, lexicon)
            → rejoin_elisions(ortho, ipa, compound_joins) + ponctuation
        """
        lexicon = self.phone_lexicon

        # 1. Parse CTC v2 → segments enrichis
        segments = parse_ctc_v2(ipa_str)
        word_segs = [s for s in segments if s["type"] == "word"]
        punct_segs = [s for s in segments if s["type"] == "punct"]

        if not word_segs:
            ponctuation = [punct_segs[0]["value"]] if punct_segs else []
            return ResultatSTT(
                ipa=ipa_str, mots_ipa=[], texte=None, mots=None,
                ponctuation=ponctuation,
            )

        ipa_words = [s["ipa"] for s in word_segs]
        compound_joins: set[int] = {
            i for i, s in enumerate(word_segs) if s.get("compound_after")
        }
        ponctuation_finale = punct_segs[-1]["value"] if punct_segs else ""

        # 1b. CTCDenoiser (optionnel, AVANT segmentation/strip)
        if self.denoiser is not None and hasattr(self.denoiser, "corriger"):
            # Separer les mots normaux des clitiques/speciaux
            normal_indices = [
                i for i, s in enumerate(word_segs)
                if not s.get("is_clitic")
            ]
            normal_words = [ipa_words[i] for i in normal_indices]

            if normal_words:
                corrected = self.denoiser.corriger(
                    normal_words,
                    self.denoiser._char2idx,
                    self.denoiser._idx2char,
                )
                # Re-injecter les mots corriges + recalculer compound_joins
                if corrected:
                    ipa_before = list(ipa_words)
                    if len(corrected) == len(normal_indices):
                        # Meme nombre de mots : remplacer 1:1
                        new_ipa = list(ipa_words)
                        for idx, ni in enumerate(normal_indices):
                            new_ipa[ni] = corrected[idx]
                    else:
                        # Re-segmentation : utiliser les mots corriges directement
                        new_ipa = corrected
                    ipa_words = new_ipa
                    compound_joins = _shift_compound_joins(
                        ipa_before, ipa_words, compound_joins,
                    )

        # 2. Strip liaisons
        ipa_words = strip_liaisons(ipa_words, lexicon)

        # 3. Split elisions (avec shift des compound_joins)
        ipa_before = list(ipa_words)
        ipa_words = split_elisions(ipa_words, lexicon)
        compound_joins = _shift_compound_joins(ipa_before, ipa_words, compound_joins)

        # 4. Split merged words (avec shift des compound_joins)
        ipa_before = list(ipa_words)
        ipa_words = split_merged_words(ipa_words, lexicon)
        compound_joins = _shift_compound_joins(ipa_before, ipa_words, compound_joins)

        # 4b. Correction phonetique CTC (optionnelle)
        if self.phone_correct and ctc_tokens is not None:
            vocab_inv = getattr(self.ctc, "vocab_inv", None)
            if vocab_inv is not None:
                word_stats = map_tokens_to_words(ctc_tokens, vocab_inv)
                ipa_before = list(ipa_words)
                ipa_words, _corrections = correct_doubtful_words(
                    ipa_words, word_stats, lexicon,
                    conf_threshold=self.phone_conf_threshold,
                )
                if _corrections:
                    compound_joins = _shift_compound_joins(
                        ipa_before, ipa_words, compound_joins,
                    )

        # 5. P2G conversion via analyser_v2
        ortho_words, pos_conf = self._p2g_convertir_v2(ipa_words, stt_mode=stt_mode)

        # 6. Merge and rescore (fusion mots sur-segmentes)
        ipa_before = list(ipa_words)
        ipa_words, ortho_words, _merge_actions = merge_and_rescore(
            ipa_words, ortho_words, pos_conf, lexicon, self.p2g,
        )
        compound_joins = _shift_compound_joins(ipa_before, ipa_words, compound_joins)

        # 7. Try elision merges (clitiques elides)
        ipa_before = list(ipa_words)
        ipa_words, ortho_words, _eli_actions = try_elision_merges(
            ipa_words, ortho_words, lexicon,
        )
        compound_joins = _shift_compound_joins(ipa_before, ipa_words, compound_joins)

        # 7b. Post-traitement grammatical (regles structurelles + lexique)
        if self.grammar_lookup is not None:
            ortho_words, _gram_actions = corriger_grammatical(
                ortho_words, self.grammar_lookup,
            )

        # 8. Valider compound_joins contre le lexique : ne garder un join
        #    que si l'IPA fusionne (mots[i]+mots[i+1]) correspond a un
        #    compose reel dans le lexique (ex. "katʁvɛ̃"+"duz" ->
        #    "quatre-vingt-douze").  Cela elimine les faux positifs du CTC
        #    comme "vingt-quatre" + "ans" -> "vingt-quatre-ans".
        if compound_joins:
            valid_joins: set[int] = set()
            for i in sorted(compound_joins):
                if i + 1 >= len(ipa_words):
                    continue
                merged = ipa_words[i] + ipa_words[i + 1]
                # Recherche exacte dans le lexique
                ortho_merged = lexicon.best_ortho(merged)
                if ortho_merged and "-" in ortho_merged:
                    valid_joins.add(i)
                    continue
                # Recherche avec perturbations CTC, en verifiant
                # que l'ortho trouvee commence bien par ortho_words[i]-
                entries, _ = lexicon.all_entries_with_perturbations(merged)
                prefix = ortho_words[i] + "-"
                for entry in entries:
                    eo = entry.get("ortho", "")
                    if eo and "-" in eo and eo.startswith(prefix):
                        valid_joins.add(i)
                        break
            compound_joins = valid_joins

        # 9. Reconstruction du texte
        texte = rejoin_elisions(ortho_words, ipa_words, compound_joins)
        if ponctuation_finale and texte:
            texte += ponctuation_finale

        ponctuation = [ponctuation_finale] if ponctuation_finale else []

        return ResultatSTT(
            ipa=ipa_str,
            mots_ipa=ipa_words,
            texte=texte,
            mots=ortho_words,
            ponctuation=ponctuation,
        )

    def _p2g_convertir_v2(
        self, mots_ipa: list[str], stt_mode: str | None = None,
    ) -> tuple[list[str], list[float]]:
        """Convertit des mots IPA via analyser_v2 (pipeline optimal).

        Retourne (ortho_words, pos_confidences).
        """
        default_conf = [0.5] * len(mots_ipa)

        # Nettoyer les apostrophes d'elision
        mots_clean = []
        for mot in mots_ipa:
            if "'" in mot:
                parts = mot.split("'", 1)
                mots_clean.append(parts[0])
                if parts[1]:
                    mots_clean.append(parts[1])
            else:
                mots_clean.append(mot)

        # Tenter analyser_v2 (retourne ortho + confiance_pos)
        if hasattr(self.p2g, "analyser_v2"):
            try:
                result = self.p2g.analyser_v2(mots_clean)
                if result and "ortho" in result:
                    ortho = result["ortho"]
                    pos_conf = result.get("confiance_pos", [0.5] * len(ortho))
                    return ortho, pos_conf
            except Exception:
                pass

        # Fallback vers le pipeline classique
        ortho = self._p2g_convertir(mots_ipa, stt_mode=stt_mode)
        return ortho, [0.5] * len(ortho)

    def transcrire_batch(
        self, audios: list[np.ndarray], sr: int = 16000,
    ) -> list[ResultatSTT]:
        """Transcrit un batch de signaux audio.

        Parameters
        ----------
        audios : list[np.ndarray]
            Liste de signaux audio PCM float32 mono.
        sr : int
            Sample rate (defaut 16000).

        Returns
        -------
        list[ResultatSTT]
            Liste de resultats.
        """
        return [self.transcrire(audio, sr=sr) for audio in audios]

    def _p2g_convertir(self, mots_ipa: list[str], stt_mode: str | None = None) -> list[str]:
        """Convertit des mots IPA en orthographe via le P2G.

        Utilise le pipeline complet (lectura_p2g.analyser) si disponible,
        incluant formules (nombres, sigles), noms propres et entites.
        Sinon utilise le graphemiseur directement (sans formules).
        """
        # Nettoyer les apostrophes d'elision du parsing CTC
        mots_clean = []
        for mot in mots_ipa:
            if "'" in mot:
                parts = mot.split("'", 1)
                mots_clean.append(parts[0])
                if parts[1]:
                    mots_clean.append(parts[1])
            else:
                mots_clean.append(mot)

        # 1. Pipeline P2G complet (formules + noms propres + entites)
        if self._p2g_analyser is not None:
            import inspect
            sig = inspect.signature(self._p2g_analyser)
            kwargs: dict = {"engine": self.p2g}
            if "formule_tolerance" in sig.parameters:
                kwargs["formule_tolerance"] = self.formule_tolerance
            if "number_mode" in sig.parameters:
                kwargs["number_mode"] = self.number_mode
            if "stt_mode" in sig.parameters and stt_mode is not None:
                kwargs["stt_mode"] = stt_mode
            result = self._p2g_analyser(mots_clean, **kwargs)
            return result.get("ortho", mots_clean)

        # 2. Graphemiseur seul (sans formules)
        if hasattr(self.p2g, "analyser"):
            result = self.p2g.analyser(mots_clean)
            return result.get("ortho", mots_clean)

        if hasattr(self.p2g, "convertir"):
            return self.p2g.convertir(mots_clean)

        return mots_clean

    def __repr__(self) -> str:
        p2g_status = type(self.p2g).__name__ if self.p2g else "None"
        pipeline = "+formules" if self._p2g_analyser else ""
        lex = "+lexicon" if self.phone_lexicon else ""
        phcor = "+phcor" if self.phone_correct else ""
        den = "+denoiser" if self.denoiser else ""
        gram = "+grammar" if self.grammar_lookup else ""
        return f"STTEngine(ctc={type(self.ctc).__name__}, p2g={p2g_status}{pipeline}{lex}{phcor}{den}{gram})"


def creer_engine(
    mode: str = "auto",
    ctc_kwargs: dict | None = None,
    p2g_kwargs: dict | None = None,
    formule_tolerance: str = "stt",
    phone_correct: bool = False,
    phone_conf_threshold: float = 0.98,
    denoiser_path: str | None = None,
    number_mode: str = "auto",
    lm_path: str | None = None,
    beam_alpha: float = 0.3,
    beam_beta: float = 0.5,
    beam_width: int = 10,
) -> STTEngine:
    """Factory pour creer un engine STT.

    Cree un engine CTC puis tente d'ajouter un engine P2G si disponible.

    Parameters
    ----------
    mode : str
        Mode d'inference pour le CTC (``"auto"``, ``"onnx"``, ``"api"``).
    ctc_kwargs : dict | None
        Arguments supplementaires pour ``lectura_decodeur.creer_engine()``.
    p2g_kwargs : dict | None
        Arguments supplementaires pour le P2G engine.
    formule_tolerance : str
        Tolerance de reconnaissance des formules :
        - ``"stt"``   : tolerant STT (defaut — normalisation vocalique, Levenshtein)
        - ``"exact"`` : IPA exact uniquement
    phone_correct : bool
        Activer la correction phonetique CTC (confiance + lexique).
    phone_conf_threshold : float
        Seuil de confiance pour la correction phonetique (defaut 0.98).
    denoiser_path : str | None
        Chemin vers le modele CTCDenoiser (.pt). Si None, tente la
        detection automatique dans ~/.lectura/models/denoiser/.
    number_mode : str
        Mode de detection des nombres :
        - ``"auto"``  : rejette les homophones ambigus isoles (defaut)
        - ``"num"``   : agressif (tous les nombres isoles convertis)
        - ``"texte"`` : pas de conversion numerique
    lm_path : str | None
        Chemin vers un modele de langue KenLM (.bin) pour beam search
        CTC. Si None, tente la detection automatique dans
        ~/.lectura/models/lm/. Si aucun modele trouve, utilise le
        decodage greedy standard.
    beam_alpha : float
        Poids du LM pour le beam search (defaut: 0.3).
    beam_beta : float
        Bonus de longueur pour le beam search (defaut: 0.5).
    beam_width : int
        Largeur du beam (defaut: 10).

    Returns
    -------
    STTEngine
        Engine pret a transcrire.

    Cascade P2G :
    1. ``lectura_p2g`` (pipeline complet avec formules)
    2. ``lectura_graphemiseur`` (modele core seul)
    3. None (mode phones uniquement)
    """
    from lectura_decodeur import creer_engine as creer_decodeur

    # Auto-detection du LM si non fourni
    if lm_path is None:
        lm_path = _find_lm_path()

    ctc_kw = dict(ctc_kwargs) if ctc_kwargs else {}
    ctc_kw.setdefault("mode", mode)
    if lm_path is not None:
        ctc_kw.setdefault("lm_path", lm_path)
        ctc_kw.setdefault("beam_alpha", beam_alpha)
        ctc_kw.setdefault("beam_beta", beam_beta)
        ctc_kw.setdefault("beam_width", beam_width)
    ctc_engine = creer_decodeur(**ctc_kw)

    p2g_kw = dict(p2g_kwargs) if p2g_kwargs else {}
    p2g_engine, p2g_analyser = _creer_p2g(**p2g_kw)

    # Creer le PhoneLexicon si un P2G est disponible
    phone_lexicon = _creer_phone_lexicon(p2g_engine)

    # Re-activer lex_select pour le pipeline STT complet
    if p2g_engine is not None and hasattr(p2g_engine, "apply_lex_select"):
        p2g_engine.apply_lex_select = True

    # Charger le CTCDenoiser (optionnel)
    denoiser = _creer_denoiser(denoiser_path, p2g_engine)

    # Charger le GrammarLookup pour le post-traitement grammatical (optionnel)
    grammar_lookup = _creer_grammar_lookup(phone_lexicon)

    return STTEngine(
        ctc_engine, p2g_engine, p2g_analyser,
        formule_tolerance=formule_tolerance,
        phone_lexicon=phone_lexicon,
        phone_correct=phone_correct,
        phone_conf_threshold=phone_conf_threshold,
        denoiser=denoiser,
        number_mode=number_mode,
        grammar_lookup=grammar_lookup,
    )


def _creer_grammar_lookup(phone_lexicon: object | None) -> GrammarLookup | None:
    """Cree un GrammarLookup depuis la DB du PhoneLexicon.

    Degradation gracieuse : retourne None si echec.
    """
    import logging

    logger = logging.getLogger(__name__)

    if phone_lexicon is None:
        return None

    db_path = getattr(phone_lexicon, "_db_path", None)
    if not db_path:
        return None

    try:
        return GrammarLookup(db_path)
    except Exception as e:
        logger.info("GrammarLookup non charge: %s", e)
        return None


def _creer_denoiser(
    denoiser_path: str | None,
    p2g_engine: object | None,
) -> object | None:
    """Charge un CTCDenoiser si disponible.

    Cascade :
    1. Chemin explicite (denoiser_path)
    2. ~/.lectura/models/denoiser/denoiser_best.pt
    3. None (degradation gracieuse)

    Le denoiser a besoin du char2idx du P2G pour fonctionner.
    """
    from pathlib import Path
    import json
    import logging

    logger = logging.getLogger(__name__)

    # Trouver le fichier denoiser
    pt_path = None
    if denoiser_path and Path(denoiser_path).exists():
        pt_path = Path(denoiser_path)
    else:
        # Detection auto
        candidates = [
            Path.home() / ".lectura" / "models" / "denoiser" / "denoiser_best.pt",
            Path.home() / ".lectura" / "models" / "stt" / "denoiser_best.pt",
        ]
        for c in candidates:
            if c.exists():
                pt_path = c
                break

    if pt_path is None:
        return None

    # Charger le vocabulaire P2G (necessaire pour corriger)
    vocab_path = _find_p2g_vocab(p2g_engine)
    if vocab_path is None:
        logger.info("CTCDenoiser trouve mais pas de vocab P2G — ignore")
        return None

    try:
        import torch
        from lectura_graphemiseur.denoiser import CTCDenoiser

        with open(vocab_path) as f:
            vocab_data = json.load(f)
        char2idx = vocab_data["vocabs"]["char2idx"]
        idx2char = {v: k for k, v in char2idx.items()}

        model = CTCDenoiser.load(str(pt_path))
        # Attacher le vocabulaire pour corriger()
        model._char2idx = char2idx
        model._idx2char = idx2char
        logger.info("CTCDenoiser charge depuis %s", pt_path)
        return model
    except (ImportError, FileNotFoundError, Exception) as e:
        logger.info("CTCDenoiser non charge: %s", e)
        return None


def _find_p2g_vocab(p2g_engine: object | None) -> str | None:
    """Trouve le fichier vocab P2G pour le denoiser."""
    from pathlib import Path

    # Chercher dans les attributs du P2G engine
    if p2g_engine is not None:
        for attr in ("_models_dir", "models_dir", "_vocab_path"):
            val = getattr(p2g_engine, attr, None)
            if val:
                p = Path(val)
                if p.is_file() and p.name.endswith("_vocab.json"):
                    return str(p)
                if p.is_dir() or p.is_file():
                    d = p if p.is_dir() else p.parent
                    for name in ("unifie_p2g_v7_vocab.json",
                                 "unifie_p2g_v6_vocab.json"):
                        candidate = d / name
                        if candidate.exists():
                            return str(candidate)

    # Chercher dans les emplacements standards
    candidates = [
        Path.home() / ".lectura" / "models" / "p2g",
    ]
    try:
        import lectura_graphemiseur
        pkg_dir = Path(lectura_graphemiseur.__file__).parent / "modeles"
        candidates.append(pkg_dir)
    except (ImportError, Exception):
        pass

    for d in candidates:
        for name in ("unifie_p2g_v7_vocab.json", "unifie_p2g_v6_vocab.json"):
            candidate = d / name
            if candidate.exists():
                return str(candidate)

    return None


def _creer_p2g(**kwargs: object) -> tuple[object | None, object | None]:
    """Tente de creer un engine P2G selon les packages installes.

    Retourne (engine, analyser_fn) ou (None, None).

    Cascade :
    1. lectura_p2g (pipeline complet avec formules, sigles, noms propres)
    2. lectura_graphemiseur (modele core seul)
    3. None
    """
    # 1. Pipeline P2G complet (formules + noms propres)
    try:
        from lectura_p2g import creer_engine as creer_p2g
        from lectura_p2g import analyser as p2g_analyser
        return creer_p2g(**kwargs), p2g_analyser
    except ImportError:
        pass

    # 2. Graphemiseur seul (sans formules)
    try:
        from lectura_graphemiseur import creer_engine as creer_graphemiseur
        return creer_graphemiseur(**kwargs), None
    except ImportError:
        pass

    # 3. Pas de P2G disponible
    return None, None


def _creer_phone_lexicon(p2g_engine: object | None) -> object | None:
    """Cree un PhoneLexicon a partir de la DB du graphemiseur.

    Cascade :
    1. phone_lexicon.db du graphemiseur (si disponible)
    2. lexique_correcteur.db (si lectura_correcteur installe)
    3. None
    """
    from pathlib import Path

    if p2g_engine is None:
        return None

    from lectura_stt._lexicon import PhoneLexicon

    # 1. DB du graphemiseur (phone_lexicon.db dans le meme dossier que les modeles)
    db_path = _find_lexicon_db(p2g_engine)
    if db_path:
        return PhoneLexicon(db_path)

    return None


def _find_lexicon_db(p2g_engine: object) -> str | None:
    """Cherche la DB lexique associee au P2G engine."""
    from pathlib import Path

    # Chercher phone_lexicon.db dans les attributs du P2G
    for attr in ("_models_dir", "models_dir", "_db_path"):
        val = getattr(p2g_engine, attr, None)
        if val:
            p = Path(val)
            if p.is_file():
                p = p.parent
            candidates = [
                p / "phone_lexicon.db",
                p / "lexique_lectura_v5.db",
            ]
            for c in candidates:
                if c.exists():
                    return str(c)

    # Chercher dans le package lectura_graphemiseur (modeles/)
    try:
        import lectura_graphemiseur
        pkg_dir = Path(lectura_graphemiseur.__file__).parent / "modeles"
        for name in ("phone_lexicon.db", "lexique_lectura_v5.db"):
            candidate = pkg_dir / name
            if candidate.exists():
                return str(candidate)
    except (ImportError, Exception):
        pass

    # Chercher dans ~/.lectura/models/ et sous-dossiers courants
    home_models = Path.home() / ".lectura" / "models"
    if home_models.exists():
        for subdir in ("", "p2g", "stt"):
            d = home_models / subdir if subdir else home_models
            if not d.exists():
                continue
            for name in ("phone_lexicon.db", "lexique_lectura_v5.db"):
                candidate = d / name
                if candidate.exists():
                    return str(candidate)

    # Chercher via lectura_correcteur
    try:
        from lectura_correcteur import _find_db_path
        db = _find_db_path()
        if db:
            return db
    except (ImportError, Exception):
        pass

    return None


def _find_lm_path() -> str | None:
    """Cherche un modele KenLM phone-level pour le beam search CTC.

    Cascade :
    1. ~/.lectura/models/lm/phone_lm_5gram.bin
    2. ~/.lectura/models/lm/*.bin (premier trouve)
    3. None (degradation gracieuse → greedy decoding)
    """
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    lm_dir = Path.home() / ".lectura" / "models" / "lm"
    if not lm_dir.exists():
        return None

    # Chercher le modele par defaut
    default = lm_dir / "phone_lm_5gram.bin"
    if default.exists():
        logger.info("KenLM phone LM trouve: %s", default)
        return str(default)

    # Chercher tout .bin dans le dossier
    for bin_file in sorted(lm_dir.glob("*.bin")):
        logger.info("KenLM phone LM trouve: %s", bin_file)
        return str(bin_file)

    return None
