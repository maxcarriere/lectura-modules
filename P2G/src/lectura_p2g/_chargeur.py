"""Chargeur paresseux des donnees metier du P2G.

Charge le fichier donnees_p2g.json une seule fois (singleton)
et expose les constantes sous forme de fonctions.
"""

from __future__ import annotations

import json
from importlib import resources

_donnees: dict | None = None


def _charger() -> dict:
    global _donnees
    if _donnees is None:
        ref = resources.files("lectura_p2g.data").joinpath("donnees_p2g.json")
        try:
            _donnees = json.loads(ref.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Fichier de donnees donnees_p2g.json introuvable. "
                "Installez le package complet : pip install lectura-p2g"
            ) from None
    return _donnees


def homophones_pos() -> dict[tuple[str, str], str]:
    raw = _charger()["homophones_pos"]
    return {tuple(k.split("|")): v for k, v in raw.items()}


def determinants_pluriel() -> frozenset[str]:
    return frozenset(_charger()["determinants_pluriel"])


def determinants_singulier() -> frozenset[str]:
    return frozenset(_charger()["determinants_singulier"])


def invariables_pluriel() -> frozenset[str]:
    return frozenset(_charger()["invariables_pluriel"])


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
