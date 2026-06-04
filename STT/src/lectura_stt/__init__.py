"""Lectura STT — Pipeline STT complet du francais (audio -> texte).

Chaine le decodeur CTC (audio -> phones IPA) avec le pipeline P2G
(phones -> orthographe) pour produire du texte francais.

Copyright (C) 2025 Max Carriere
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

from lectura_stt._parse_ctc import parse_ctc_output
from lectura_stt._assembler import assembler_texte

__version__ = "1.2.0"


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
    """

    def __init__(
        self,
        ctc_engine: object,
        p2g_engine: object | None = None,
        p2g_analyser: object | None = None,
        formule_tolerance: str = "stt",
    ) -> None:
        self.ctc = ctc_engine
        self.p2g = p2g_engine
        self._p2g_analyser = p2g_analyser  # lectura_p2g.analyser (avec formules)
        self.formule_tolerance = formule_tolerance

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

        # Etape 2 : parse IPA → mots
        parsed = parse_ctc_output(ipa_str)

        # Etape 3 : P2G → orthographe (si disponible)
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
        # (l'elision est geree par _assembler.py)
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
        return f"STTEngine(ctc={type(self.ctc).__name__}, p2g={p2g_status}{pipeline})"


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

    return STTEngine(ctc_engine, p2g_engine, p2g_analyser, formule_tolerance=formule_tolerance)


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
