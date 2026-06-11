"""Lectura CTC — Decodeur phonetique CTC du francais.

Architecture : CNN-BiGRU-CTC medium (10.6M params, ONNX INT8 = 38 Mo)

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — voir LICENCE-COMMERCIALE.md

Deux backends d'inference :
  - ONNX Runtime  (rapide, ~10ms/s audio)  — necessite modeles locaux
  - API           (serveur Lectura)         — mode par defaut

Exemple rapide (mode API, zero config)::

    from lectura_ctc import creer_engine
    engine = creer_engine()
    result = engine.transcrire(audio)

Exemple avec backend local::

    from lectura_ctc import creer_engine
    engine = creer_engine(mode="onnx")
    result = engine.transcrire(audio)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

__version__ = "2.0.0"

_MODELES_DIR = Path(__file__).parent / "modeles"
_DATA_DIR = Path(__file__).parent / "data"


class CTCEngine(Protocol):
    """Interface publique pour les engines CTC."""

    def transcrire(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Audio PCM float32 mono → chaine IPA avec separateurs."""
        ...

    def transcrire_batch(
        self, audios: list[np.ndarray], sr: int = 16000,
    ) -> list[str]:
        """Batch de transcriptions."""
        ...


def _resoudre_modeles_dir(models_dir: str | Path | None = None) -> Path | None:
    """Cascade de resolution du dossier modeles.

    Ordre de priorite :
    1. Parametre explicite ``models_dir``
    2. Variable d'environnement ``LECTURA_MODELS_DIR`` / ctc
    3. Dossier utilisateur ``~/.lectura/models/ctc/``
    4. Site-packages (dossier ``modeles/`` du package installe)
    """
    candidats: list[Path] = []
    if models_dir is not None:
        candidats.append(Path(models_dir))
    env = os.environ.get("LECTURA_MODELS_DIR")
    if env:
        candidats.append(Path(env) / "ctc")
    candidats.append(Path.home() / ".lectura" / "models" / "ctc")
    candidats.append(_MODELES_DIR)

    for d in candidats:
        if (d / "phone_ctc_int8.onnx").exists():
            return d
    return None


def _resoudre_vocab() -> Path:
    """Retourne le chemin vers vocab_phones.json (embarque dans data/)."""
    vocab = _DATA_DIR / "vocab_phones.json"
    if vocab.exists():
        return vocab
    raise FileNotFoundError(
        f"Vocabulaire introuvable : {vocab}. "
        "Verifiez l'installation du package lectura-ctc."
    )


def creer_engine(
    mode: str = "auto",
    models_dir: str | Path | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    lm_path: str | None = None,
    beam_alpha: float = 0.3,
    beam_beta: float = 0.5,
    beam_width: int = 10,
) -> CTCEngine:
    """Factory pour creer un engine d'inference CTC.

    Parameters
    ----------
    mode : str
        ``"auto"`` (defaut) : ONNX si modeles presents, sinon API.
        ``"onnx"`` : force le backend ONNX Runtime.
        ``"api"`` : force le backend API.
    models_dir : str | Path | None
        Chemin vers le dossier contenant ``phone_ctc_int8.onnx``.
        Si None, cascade automatique.
    api_url : str | None
        URL du serveur Lectura (pour mode API).
    api_key : str | None
        Cle API optionnelle.
    lm_path : str | None
        Chemin vers un modele KenLM (.bin) pour beam search.
        Si None, utilise le decodage greedy standard.
    beam_alpha : float
        Poids du LM pour le beam search (defaut: 0.3).
    beam_beta : float
        Bonus de longueur pour le beam search (defaut: 0.5).
    beam_width : int
        Largeur du beam (defaut: 10).

    Returns
    -------
    CTCEngine
        Engine pret a transcrire.

    Raises
    ------
    RuntimeError
        Si aucun backend n'est disponible.
    """
    if mode == "api":
        from lectura_ctc._inference_api import ApiCTCEngine
        return ApiCTCEngine(api_url=api_url, api_key=api_key)

    resolved_dir = _resoudre_modeles_dir(models_dir)

    if mode == "auto" and resolved_dir is None:
        from lectura_ctc._inference_api import ApiCTCEngine
        return ApiCTCEngine(api_url=api_url, api_key=api_key)

    if mode in ("auto", "onnx"):
        try:
            from lectura_ctc._inference_onnx import OnnxCTCEngine
            base = resolved_dir or _MODELES_DIR
            onnx_path = base / "phone_ctc_int8.onnx"
            vocab_path = _resoudre_vocab()
            return OnnxCTCEngine(
                onnx_path, vocab_path,
                lm_path=lm_path,
                beam_alpha=beam_alpha,
                beam_beta=beam_beta,
                beam_width=beam_width,
            )
        except (ImportError, FileNotFoundError, Exception) as exc:
            if mode == "onnx":
                raise RuntimeError(
                    f"Backend ONNX indisponible : {exc}. "
                    "Installez onnxruntime (pip install lectura-ctc[onnx]) "
                    "et les modeles."
                ) from exc

    raise RuntimeError(
        f"Aucun backend d'inference disponible (mode={mode!r}). "
        "Verifiez que les modeles sont installes ou utilisez mode='api'."
    )
