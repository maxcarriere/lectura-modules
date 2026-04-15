"""Chargeur paresseux des donnees metier du G2P.

Charge le fichier donnees_g2p.json une seule fois (singleton)
et expose les constantes sous forme de fonctions.
"""

from __future__ import annotations

import json
from importlib import resources

_donnees: dict | None = None


def _charger() -> dict:
    global _donnees
    if _donnees is None:
        ref = resources.files("lectura_nlp.data").joinpath("donnees_g2p.json")
        try:
            _donnees = json.loads(ref.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Fichier de donnees donnees_g2p.json introuvable. "
                "Installez le package complet : pip install lectura-g2p"
            ) from None
    return _donnees


def ponctuation_lecture() -> dict[str, tuple[str, str]]:
    raw = _charger()["ponctuation_lecture"]
    return {k: tuple(v) for k, v in raw.items()}


def denasalisation() -> dict[str, str]:
    return _charger()["denasalisation"]


def liaison_consonnes() -> dict[str, str]:
    return _charger()["liaison_consonnes"]


def voyelles() -> set[str]:
    return set(_charger()["voyelles"])


def consonnes() -> set[str]:
    return set(_charger()["consonnes"])


def semi_voyelles() -> set[str]:
    return set(_charger()["semi_voyelles"])


def nasales() -> set[str]:
    return set(_charger()["nasales"])


def liquides() -> set[str]:
    return set(_charger()["liquides"])


def occlusives() -> set[str]:
    return set(_charger()["occlusives"])


def fricatives() -> set[str]:
    return set(_charger()["fricatives"])


def kept_intact() -> set[str]:
    return set(_charger()["kept_intact"])
