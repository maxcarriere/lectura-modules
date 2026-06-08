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

from lectura_stt._parse_ctc import parse_ctc_output, parse_ctc_v2, ParseResult
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

__version__ = "2.1.0"


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
    ) -> None:
        self.ctc = ctc_engine
        self.p2g = p2g_engine
        self._p2g_analyser = p2g_analyser  # lectura_p2g.analyser (avec formules)
        self.formule_tolerance = formule_tolerance
        self.phone_lexicon = phone_lexicon

    def transcrire(self, audio: np.ndarray, sr: int = 16000) -> ResultatSTT:
        """Transcrit un signal audio en texte.

        Parameters
        ----------
        audio : np.ndarray
            Signal audio PCM float32 mono.
        sr : int
            Sample rate (defaut 16000).

        Returns
        -------
        ResultatSTT
            Resultat contenant IPA et texte (si P2G disponible).
        """
        # Etape 1 : CTC → IPA
        ipa_str = self.ctc.transcrire(audio, sr=sr)

        # Choix du pipeline : optimal si PhoneLexicon disponible, sinon simplifie
        if self.phone_lexicon is not None and self.p2g is not None:
            return self._transcrire_optimal(ipa_str)
        return self._transcrire_simple(ipa_str)

    def _transcrire_simple(self, ipa_str: str) -> ResultatSTT:
        """Pipeline simplifie (sans PhoneLexicon)."""
        parsed = parse_ctc_output(ipa_str)

        texte: str | None = None
        mots_ortho: list[str] | None = None

        if self.p2g is not None and parsed.mots_ipa:
            mots_ortho = self._p2g_convertir(parsed.mots_ipa)
            texte = assembler_texte(mots_ortho, parsed.ponctuation_finale)

        ponctuation = [parsed.ponctuation_finale] if parsed.ponctuation_finale else []

        return ResultatSTT(
            ipa=ipa_str,
            mots_ipa=parsed.mots_ipa,
            texte=texte,
            mots=mots_ortho,
            ponctuation=ponctuation,
        )

    def _transcrire_optimal(self, ipa_str: str) -> ResultatSTT:
        """Pipeline optimal avec postprocessing CTC.

        Sequence :
            parse_ctc_v2 → extract word segments + compound_joins
            → strip_liaisons(ipa_words, lexicon)
            → split_elisions(ipa_words, lexicon)
            → split_merged_words(ipa_words, lexicon)
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

        # 5. P2G conversion via analyser_v2
        ortho_words, pos_conf = self._p2g_convertir_v2(ipa_words)

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

        # 8. Reconstruction du texte
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
        self, mots_ipa: list[str],
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
        ortho = self._p2g_convertir(mots_ipa)
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

    def _p2g_convertir(self, mots_ipa: list[str]) -> list[str]:
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
            result = self._p2g_analyser(
                mots_clean, engine=self.p2g,
                formule_tolerance=self.formule_tolerance,
            )
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
        return f"STTEngine(ctc={type(self.ctc).__name__}, p2g={p2g_status}{pipeline}{lex})"


def creer_engine(
    mode: str = "auto",
    ctc_kwargs: dict | None = None,
    p2g_kwargs: dict | None = None,
    formule_tolerance: str = "stt",
) -> STTEngine:
    """Factory pour creer un engine STT.

    Cree un engine CTC puis tente d'ajouter un engine P2G si disponible.

    Parameters
    ----------
    mode : str
        Mode d'inference pour le CTC (``"auto"``, ``"onnx"``, ``"api"``).
    ctc_kwargs : dict | None
        Arguments supplementaires pour ``lectura_ctc.creer_engine()``.
    p2g_kwargs : dict | None
        Arguments supplementaires pour le P2G engine.
    formule_tolerance : str
        Tolerance de reconnaissance des formules :
        - ``"stt"``   : tolerant STT (defaut — normalisation vocalique, Levenshtein)
        - ``"exact"`` : IPA exact uniquement

    Returns
    -------
    STTEngine
        Engine pret a transcrire.

    Cascade P2G :
    1. ``lectura_p2g`` (pipeline complet avec formules)
    2. ``lectura_graphemiseur`` (modele core seul)
    3. None (mode phones uniquement)
    """
    from lectura_ctc import creer_engine as creer_ctc

    ctc_kw = dict(ctc_kwargs) if ctc_kwargs else {}
    ctc_kw.setdefault("mode", mode)
    ctc_engine = creer_ctc(**ctc_kw)

    p2g_kw = dict(p2g_kwargs) if p2g_kwargs else {}
    p2g_engine, p2g_analyser = _creer_p2g(**p2g_kw)

    # Creer le PhoneLexicon si un P2G est disponible
    phone_lexicon = _creer_phone_lexicon(p2g_engine)

    # Re-activer lex_select pour le pipeline STT complet
    if p2g_engine is not None and hasattr(p2g_engine, "apply_lex_select"):
        p2g_engine.apply_lex_select = True

    return STTEngine(
        ctc_engine, p2g_engine, p2g_analyser,
        formule_tolerance=formule_tolerance,
        phone_lexicon=phone_lexicon,
    )


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

    # Chercher dans ~/.lectura/models/
    home_models = Path.home() / ".lectura" / "models"
    if home_models.exists():
        for name in ("phone_lexicon.db", "lexique_lectura_v5.db"):
            candidate = home_models / name
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
