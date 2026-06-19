"""Gestion des signatures de timbre vocal.

Signatures pre-calculees et chargement de signatures externes.
Chaque signature est un vecteur cepstral (DCT du log SP moyen)
qui capture l'identite vocale d'un locuteur.

Signatures incluses :
  - "neutre" : identite (banque actuelle, changement minimal)
  - "homme" : tilt spectral sombre, energie basses renforcee
  - "enfant" : tilt releve, formants decales vers le haut
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── Signatures pre-calculees ──────────────────────────────────────
#
# Construites par transformation parametrique de la signature "neutre".
# La signature neutre est un vecteur nul (pas de transfert).
# Les autres signatures modifient les coefficients cepstraux cles :
#   [0]   : energie globale
#   [1-2] : tilt spectral (pente du spectre)
#   [16+] : texture vocale (detail spectral fin)

_N_CEPS = 400  # dimension par defaut (correspond a fft_size/2+1 pour 44100 Hz)


def _make_neutre() -> np.ndarray:
    """Signature neutre : vecteur nul (aucun transfert)."""
    return np.zeros(_N_CEPS, dtype=np.float64)


def _make_homme() -> np.ndarray:
    """Signature masculine : tilt sombre, basses renforcees."""
    sig = np.zeros(_N_CEPS, dtype=np.float64)
    # Energie globale legerement plus forte
    sig[0] = 0.3
    # Tilt spectral : pente negative (plus d'energie en basses frequences)
    sig[1] = -0.5
    sig[2] = -0.3
    # Texture : legere coloration sombre en haute frequence
    for i in range(16, min(60, _N_CEPS)):
        sig[i] = -0.05 * (1.0 - (i - 16) / 44.0)
    return sig


def _make_enfant() -> np.ndarray:
    """Signature enfant : tilt releve, energie HF."""
    sig = np.zeros(_N_CEPS, dtype=np.float64)
    # Energie globale legerement reduite
    sig[0] = -0.15
    # Tilt spectral : pente positive (plus d'energie en hautes frequences)
    sig[1] = 0.4
    sig[2] = 0.25
    # Texture : brillance ajoutee en haute frequence
    for i in range(16, min(60, _N_CEPS)):
        sig[i] = 0.04 * (1.0 - (i - 16) / 44.0)
    return sig


# Dict des signatures disponibles
BUILTIN_SIGNATURES: dict[str, np.ndarray] = {
    "neutre": _make_neutre(),
    "homme": _make_homme(),
    "enfant": _make_enfant(),
}

# Base F0 suggeree par signature (en Hz)
SUGGESTED_BASE_F0: dict[str, float] = {
    "neutre": 175.0,
    "homme": 120.0,
    "enfant": 280.0,
}


def list_signatures() -> list[str]:
    """Liste les noms de signatures disponibles."""
    return sorted(BUILTIN_SIGNATURES.keys())


def load_signature(name_or_path: str) -> np.ndarray:
    """Charger une signature par nom ou depuis un fichier JSON.

    Args:
        name_or_path: nom d'une signature pre-calculee (ex: "homme")
            ou chemin vers un fichier .json contenant un vecteur cepstral.

    Returns:
        Vecteur cepstral 1D (np.float64)

    Raises:
        ValueError: si le nom est inconnu et le fichier n'existe pas
    """
    # Signature pre-calculee ?
    if name_or_path in BUILTIN_SIGNATURES:
        return BUILTIN_SIGNATURES[name_or_path].copy()

    # Fichier externe ?
    path = Path(name_or_path)
    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return np.array(data, dtype=np.float64)
        elif isinstance(data, dict) and "signature" in data:
            return np.array(data["signature"], dtype=np.float64)
        else:
            raise ValueError(
                f"Format de signature invalide dans {path}. "
                "Attendu: liste de floats ou dict avec cle 'signature'."
            )

    raise ValueError(
        f"Signature inconnue: {name_or_path!r}. "
        f"Signatures disponibles: {list_signatures()}"
    )
