"""Lectura STT Formules — STT dedie formules avec vocabulaire semantique.

Modele CTC autonome entraine sur des formules, avec un vocabulaire de
sortie semantique (~87 tokens) au lieu de phonemes IPA.

Phases 1-3 : vocabulaire + tokenizer + corpus + training + inference.

Copyright (C) 2025 Max Carriere
Licence : AGPL-3.0-or-later — voir LICENCE.txt
"""

from __future__ import annotations

from pathlib import Path

from lectura_stt_formules._vocab import (
    VOCAB,
    NUM_TOKENS,
    ORTHO_TO_TOKENS,
    token_ids_to_names,
    vocab_to_json,
    # Tokens de controle
    BLANK,
    SPACE,
)
from lectura_stt_formules._tokenizer import events_to_token_sequence

__version__ = "0.1.0"


def creer_engine(
    mode: str = "auto",
    models_dir: str | Path | None = None,
):
    """Factory pour le moteur d'inference FormulaCTC.

    Args:
        mode: ``"auto"`` ou ``"local"`` (pas d'API pour ce module).
        models_dir: chemin vers le dossier contenant formula_ctc*.onnx.
            Si None, utilise la cascade de resolution standard.

    Returns:
        OnnxFormulaEngine

    Raises:
        FileNotFoundError: si aucun modele ONNX n'est trouve.
        ImportError: si onnxruntime n'est pas installe.
        ValueError: si mode n'est pas ``"auto"`` ou ``"local"``.
    """
    if mode not in ("auto", "local"):
        raise ValueError(f"mode doit etre 'auto' ou 'local', recu : {mode!r}")

    from lectura_stt_formules._inference import OnnxFormulaEngine

    return OnnxFormulaEngine(models_dir=models_dir)


def transcrire(audio_path: str | Path, **kwargs) -> dict:
    """Convenience : cree un engine et transcrit un fichier.

    Args:
        audio_path: chemin vers le fichier audio (WAV, FLAC, etc.)
        **kwargs: passes a creer_engine() (mode, models_dir, etc.)

    Returns:
        dict avec tokens, names, logits (cf. OnnxFormulaEngine.transcrire).
    """
    engine = creer_engine(**kwargs)
    return engine.transcrire(audio_path)
