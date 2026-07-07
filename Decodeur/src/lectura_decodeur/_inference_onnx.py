"""Backend ONNX Runtime pour le modele CTC.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from lectura_decodeur._mel import mel_spectrogram
from lectura_decodeur._decode import (
    ctc_greedy_decode,
    ctc_greedy_decode_with_alternatives,
    ids_vers_phones,
)

logger = logging.getLogger(__name__)


class OnnxCTCEngine:
    """Inference CTC via ONNX Runtime.

    Parameters
    ----------
    onnx_path : Path
        Chemin vers le modele ONNX (``phone_ctc_int8.onnx``).
    vocab_path : Path
        Chemin vers le fichier vocabulaire (``vocab_phones.json``).
    lm_path : str | None
        Chemin vers un modele KenLM (.bin) pour beam search.
        Si None, utilise le decodage greedy standard.
    beam_alpha : float
        Poids du LM pour le beam search (defaut: 0.3).
    beam_beta : float
        Bonus de longueur pour le beam search (defaut: 0.5).
    beam_width : int
        Largeur du beam (defaut: 10).
    """

    def __init__(
        self,
        onnx_path: Path,
        vocab_path: Path,
        lm_path: str | None = None,
        beam_alpha: float = 0.3,
        beam_beta: float = 0.5,
        beam_width: int = 10,
    ) -> None:
        import onnxruntime as ort

        logger.info("Chargement modele CTC ONNX : %s", onnx_path)
        self.session = ort.InferenceSession(str(onnx_path))

        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)

        # vocab : {"[PAD]": 0, "a": 3, ...}
        self.vocab_inv: dict[int, str] = {v: k for k, v in vocab.items()}
        self.blank_id: int = vocab.get("[PAD]", 0)

        # Beam search optionnel
        self._beam_decoder = None
        if lm_path is not None:
            try:
                from lectura_decodeur._beam_search import PhoneLMBeamDecoder
                self._beam_decoder = PhoneLMBeamDecoder(
                    vocab=vocab,
                    lm_path=lm_path,
                    alpha=beam_alpha,
                    beta=beam_beta,
                    beam_width=beam_width,
                )
                logger.info(
                    "Beam search actif (alpha=%.2f, beta=%.2f, width=%d)",
                    beam_alpha, beam_beta, beam_width,
                )
            except ImportError:
                logger.warning(
                    "kenlm non installe — beam search desactive. "
                    "pip install kenlm pour activer."
                )

    def transcrire(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Transcrit un signal audio en chaine IPA.

        Parameters
        ----------
        audio : np.ndarray
            Signal PCM float32 mono, shape (N,).
        sr : int
            Sample rate (doit etre 16000).

        Returns
        -------
        str
            Chaine IPA, ex: ``"b ɔ̃ ʒ u ʁ | l ə | m ɔ̃ d"``.
        """
        mel = mel_spectrogram(audio, sr)  # (1, 1, 80, T)
        logits = self.session.run(None, {"mel": mel})[0]  # (1, T', V)
        if self._beam_decoder is not None:
            ids = self._beam_decoder.decode_logits(logits[0])
        else:
            ids = ctc_greedy_decode(logits[0], blank_id=self.blank_id)
        return ids_vers_phones(ids, self.vocab_inv)

    def transcrire_avec_alternatives(
        self, audio: np.ndarray, sr: int = 16000, top_k: int = 5,
    ) -> tuple[str, list[dict]]:
        """Transcrit avec top-K alternatives softmax par token.

        Parameters
        ----------
        audio : np.ndarray
            Signal PCM float32 mono.
        sr : int
            Sample rate (defaut 16000).
        top_k : int
            Nombre d'alternatives par token.

        Returns
        -------
        tuple[str, list[dict]]
            (ipa_str, tokens) ou tokens contient phone_id, confidence,
            entropy, alternatives pour chaque token emis.
        """
        mel = mel_spectrogram(audio, sr)
        logits = self.session.run(None, {"mel": mel})[0]  # (1, T', V)
        tokens = ctc_greedy_decode_with_alternatives(
            logits[0], blank_id=self.blank_id, top_k=top_k,
        )
        # Enrichir chaque token avec la clé "phone" (IPA string)
        for t in tokens:
            t["phone"] = self.vocab_inv.get(t["phone_id"], "")
        ids = [t["phone_id"] for t in tokens]
        ipa_str = ids_vers_phones(ids, self.vocab_inv)
        return ipa_str, tokens

    def transcrire_batch(
        self, audios: list[np.ndarray], sr: int = 16000,
    ) -> list[str]:
        """Transcrit un batch d'audios en chaines IPA.

        Les audios sont traites sequentiellement (pas de padding batch
        pour eviter les artefacts de longueur variable).
        """
        return [self.transcrire(audio, sr) for audio in audios]

    def __repr__(self) -> str:
        return f"OnnxCTCEngine(vocab_size={len(self.vocab_inv)})"
