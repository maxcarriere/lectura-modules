"""Post-filtre WORLD conditionne par paire de phones.

Charge un modele ONNX (PhonePairMLP) qui predit un residuel delta_log_sp
et delta_ap pour restaurer le contraste spectral perdu par le moyennage
des diphones.

Le modele prend en entree :
  - log_sp (n_frames, 372) : enveloppe spectrale en log
  - phone_a_ids (n_frames,) : ID du phone gauche
  - phone_b_ids (n_frames,) : ID du phone droit

Et produit :
  - delta_log_sp (n_frames, 372) : correction spectrale
  - delta_ap (n_frames, 372) : correction d'aperiodicite
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Phone vocabulary for the postfilter (mirrors phoneme_vocab.json "#" entry)
# Only phones used in diphone keys (no punctuation, no <PAD>/<UNK>)
_DIPHONE_PHONES = [
    "#",  # silence boundary
    "a", "e", "i", "o", "u", "y",
    "\u025b", "\u0254", "\u00f8", "\u0153", "\u0259", "\u0251",  # vowels
    "\u025b\u0303", "\u0254\u0303", "\u0251\u0303", "\u0153\u0303",  # nasal vowels
    "p", "b", "t", "d", "k", "\u0261",  # stops (g = U+0261)
    "f", "v", "s", "z", "\u0283", "\u0292",  # fricatives
    "m", "n", "\u0272",  # nasals
    "l", "\u0281",  # liquids
    "j", "w", "\u0265",  # glides
    "t\u0283", "d\u0292",  # affricates
    "x", "\u0263", "\u014b", "\u0279",  # rares
]

# Build phone → ID mapping (used at inference)
PHONE2ID: dict[str, int] = {ph: i for i, ph in enumerate(_DIPHONE_PHONES)}
N_PHONES = len(_DIPHONE_PHONES)

# Aliases for common variants
_PHONE_ALIASES: dict[str, str] = {
    "g": "\u0261",      # ASCII g → IPA g
    "\u0265": "\u0265",  # turned h (already in vocab)
}


def _resolve_phone_id(phone: str) -> int:
    """Resolve a phone string to its embedding ID."""
    if phone in PHONE2ID:
        return PHONE2ID[phone]
    alias = _PHONE_ALIASES.get(phone)
    if alias is not None and alias in PHONE2ID:
        return PHONE2ID[alias]
    # Unknown phone → map to "#" (silence, ID=0)
    return 0


def _parse_diphone_key(key: str) -> tuple[int, int]:
    """Parse a diphone key like 'b-a' into (phone_a_id, phone_b_id)."""
    parts = key.split("-", 1)
    if len(parts) != 2:
        return 0, 0
    return _resolve_phone_id(parts[0]), _resolve_phone_id(parts[1])


class SPPostFilter:
    """Post-filtre spectral ONNX conditionne par paire de phones.

    Charge sp_postfilter.onnx et applique une correction per-frame
    de l'enveloppe spectrale et de l'aperiodicite.
    """

    def __init__(self, onnx_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime requis pour le post-filtre spectral. "
                "pip install onnxruntime"
            )

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._n_sp_bins: int | None = None
        log.info("SPPostFilter charge: %s", onnx_path)

    def apply(
        self,
        sp: np.ndarray,
        ap: np.ndarray,
        diphone_keys: list[str],
        boundaries: list[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Appliquer le post-filtre sur les parametres WORLD concatenes.

        Args:
            sp: (n_frames, n_bins) enveloppe spectrale (lineaire)
            ap: (n_frames, n_bins) aperiodicite [0, 1]
            diphone_keys: liste des cles diphone (ex: ["#-b", "b-a", "a-#"])
            boundaries: liste de (start_frame, end_frame) par diphone

        Returns:
            sp_out: (n_frames, n_bins) enveloppe corrigee
            ap_out: (n_frames, n_bins) aperiodicite corrigee
        """
        n_frames, n_bins = sp.shape

        # Build per-frame phone IDs from diphone boundaries
        phone_a_ids = np.zeros(n_frames, dtype=np.int64)
        phone_b_ids = np.zeros(n_frames, dtype=np.int64)

        for key, (start, end) in zip(diphone_keys, boundaries):
            a_id, b_id = _parse_diphone_key(key)
            # Clamp to valid range
            start = max(0, start)
            end = min(end, n_frames)
            if start < end:
                phone_a_ids[start:end] = a_id
                phone_b_ids[start:end] = b_id

        # Prepare input
        log_sp = np.log(np.maximum(sp, 1e-10)).astype(np.float32)

        # The ONNX model may expect a specific number of SP bins
        # If the SP has more bins than the model expects, truncate for inference
        # then apply the correction only to the truncated portion
        model_bins = self._get_model_bins(log_sp)
        if model_bins is not None and model_bins < n_bins:
            log_sp_input = log_sp[:, :model_bins]
        else:
            log_sp_input = log_sp
            model_bins = n_bins

        # Run ONNX inference
        outputs = self._session.run(
            None,
            {
                "log_sp": log_sp_input,
                "phone_a_ids": phone_a_ids,
                "phone_b_ids": phone_b_ids,
            },
        )
        delta_log_sp = outputs[0]  # (n_frames, model_bins)
        delta_ap = outputs[1]      # (n_frames, model_bins)

        # Apply corrections
        sp_out = sp.copy()
        ap_out = ap.copy()

        # SP: exp(log_sp + delta_log_sp)
        corrected_log_sp = log_sp[:, :model_bins] + delta_log_sp
        sp_out[:, :model_bins] = np.exp(corrected_log_sp)
        sp_out = np.maximum(sp_out, 1e-10)

        # AP: clip(ap + delta_ap, 0, 1)
        ap_out[:, :model_bins] = np.clip(
            ap[:, :model_bins] + delta_ap, 0.0, 1.0
        )

        return sp_out, ap_out

    def _get_model_bins(self, log_sp: np.ndarray) -> int | None:
        """Determine the number of SP bins the model expects."""
        if self._n_sp_bins is not None:
            return self._n_sp_bins

        # Get from ONNX model input shape
        inputs = self._session.get_inputs()
        for inp in inputs:
            if inp.name == "log_sp":
                shape = inp.shape
                if len(shape) >= 2 and isinstance(shape[1], int):
                    self._n_sp_bins = shape[1]
                    return self._n_sp_bins
        return None
