"""Retimbrage optionnel via OpenVoice (lectura-vc-zeroshot).

Post-traitement pour remplacer le timbre "moyen" du diphone
par un timbre coherent issu d'une reference locuteur.

Supporte :
  - Preset par nom ("siwis", "bernard", etc.)
  - Fichier audio de reference (chemin)
  - Liste de references (poids egaux → SE moyenne)
  - Dict pondere ({"siwis": 0.5, "nadine": 0.5} → blend)

Le curseur 'voix_variante' (-1 a +1) decale les formants :
  -1 = grave/masculin, 0 = neutre, +1 = aigu/enfant
Formule : sr_override = 22050 / (2 ** variante)

Requires: pip install 'lectura-tts-diphone[vc]'
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

OV_SR = 22050

# Type pour voix polymorphe
VoixSpec = str | Path | list[str] | dict[str, float] | None


def variante_to_sr_override(variante: float) -> int | None:
    """Convertit un curseur variante en sr_override pour le trick SR.

    Parameters
    ----------
    variante : float entre -1 et +1.
        0.0 = neutre (pas de trick).
        +1  = formants montes (enfant/aigu), sr_override = 11025.
        -1  = formants baisses (homme/grave), sr_override = 44100.

    Returns
    -------
    int ou None si neutre.
    """
    if abs(variante) < 0.01:
        return None
    return max(8000, min(48000, int(OV_SR / (2 ** variante))))


def _make_cache_key(reference: VoixSpec, sr_override: int | None) -> tuple:
    """Construit une cle de cache hashable pour la reference."""
    if isinstance(reference, dict):
        ref_key = tuple(sorted(reference.items()))
    elif isinstance(reference, list):
        ref_key = tuple(reference)
    elif isinstance(reference, (str, Path)):
        ref_key = str(reference)
    else:
        ref_key = id(reference)
    return (ref_key, sr_override)


class RetimbreEngine:
    """Cache les sessions ONNX et les speaker embeddings.

    Le SE de la reference est extrait/resolu une seule fois et cache.
    Seul le SE source (audio synthetise) est recalcule a chaque appel.
    """

    def __init__(self, vc_models_dir: str | Path | None = None):
        self._engine = None       # ZeroShotEngine (lazy)
        self._vc_models_dir = vc_models_dir
        self._se_cache: dict[tuple, np.ndarray] = {}

    def _ensure_loaded(self):
        """Lazy-load le ZeroShotEngine."""
        if self._engine is None:
            try:
                from lectura_vc_zeroshot import creer_engine
            except ImportError:
                raise ImportError(
                    "lectura-vc-zeroshot requis pour le retimbre. "
                    "Installez avec: pip install 'lectura-tts-diphone[vc]'"
                )
            self._engine = creer_engine(models_dir=self._vc_models_dir)
        return self._engine

    def _get_tgt_se(self, engine, reference, ref_sr, sr_override):
        """Resout et cache le SE cible.

        Utilise engine.resolve_target_se() pour gerer les presets,
        listes, dicts ponderes, et fichiers audio.
        """
        cache_key = _make_cache_key(reference, sr_override)

        if cache_key not in self._se_cache:
            tgt_se = engine.resolve_target_se(
                reference, ref_sr=ref_sr, sr_override=sr_override,
            )
            self._se_cache[cache_key] = tgt_se
            logger.debug("SE cible resolue et cachee (key=%s)", cache_key)
        else:
            logger.debug("SE cible depuis cache (key=%s)", cache_key)

        return self._se_cache[cache_key]

    def retimbre(
        self,
        audio: np.ndarray,
        sr: int,
        reference: VoixSpec,
        ref_sr: int | None = None,
        variante: float = 0.0,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Applique le retimbre OpenVoice.

        Parameters
        ----------
        audio : audio source float32.
        sr : sample rate source.
        reference : voix cible — polymorphe :
            - str : nom de preset ("siwis") ou chemin fichier audio.
            - list[str] : plusieurs references (poids egaux).
            - dict[str, float] : blend pondere.
              Ex: {"siwis": 0.5, "nadine": 0.3, "ezwa": 0.2}
        ref_sr : sample rate de la reference si array.
        variante : curseur -1 (grave) a +1 (aigu).
        tau : parametre OpenVoice.

        Returns
        -------
        (audio_retimbre @ OV_SR Hz, OV_SR)
        """
        engine = self._ensure_loaded()
        sr_override = variante_to_sr_override(variante)

        logger.info(
            "Retimbre: reference=%s, variante=%.2f, sr_override=%s, tau=%.2f",
            reference, variante, sr_override, tau,
        )

        # SE cible (reference) — cachee
        tgt_se = self._get_tgt_se(engine, reference, ref_sr, sr_override)

        # SE source (audio synthetise) — recalculee a chaque fois
        src_se = engine.extract_se(audio, sr=sr)

        return engine.convert_from_se(
            audio=audio,
            src_se=src_se,
            tgt_se=tgt_se,
            sr_in=sr,
            tau=tau,
        )
