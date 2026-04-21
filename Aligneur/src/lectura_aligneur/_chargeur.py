"""Chargeur paresseux des donnees metier de l'Aligneur.

Charge le fichier donnees_aligneur.json une seule fois (singleton)
et expose les constantes derivees : alphabet IPA, tables phoneme-grapheme,
mappings eSpeak, consonnes de liaison, lettres muettes, et les sets
voyelles/consonnes/semi-voyelles.
"""

from __future__ import annotations

import json
from importlib import resources

_donnees: dict | None = None


def _charger() -> dict:
    global _donnees
    if _donnees is None:
        ref = resources.files("lectura_aligneur.data").joinpath("donnees_aligneur.json")
        try:
            _donnees = json.loads(ref.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Fichier de donnees donnees_aligneur.json introuvable. "
                "Installez le package complet : pip install lectura-aligneur"
            ) from None
    return _donnees


def alphabet_ipa() -> dict[str, dict]:
    return _charger()["alphabet_ipa"]


def phone_to_graphemes() -> dict[str, list[str]]:
    return _charger()["phone_to_graphemes"]


def espeak_to_ipa() -> dict[str, str]:
    return _charger()["espeak_to_ipa"]


def liaison_consonnes() -> dict[str, str]:
    return _charger()["liaison_consonnes"]


def lettres_muettes_possibles() -> set[str]:
    return set(_charger()["lettres_muettes_possibles"])


# Sets derives de l'alphabet IPA
_voyelles: set[str] | None = None
_consonnes: set[str] | None = None
_semi_voyelles: set[str] | None = None


def voyelles() -> set[str]:
    global _voyelles
    if _voyelles is None:
        _voyelles = {
            ph for ph, meta in alphabet_ipa().items()
            if meta.get("type") == "voyelle"
            and meta.get("sous_type") == "orale"
        }
    return _voyelles


def consonnes() -> set[str]:
    global _consonnes
    if _consonnes is None:
        _consonnes = {
            ph for ph, meta in alphabet_ipa().items()
            if meta.get("type") == "consonne"
        }
    return _consonnes


def semi_voyelles() -> set[str]:
    global _semi_voyelles
    if _semi_voyelles is None:
        _semi_voyelles = {
            ph for ph, meta in alphabet_ipa().items()
            if meta.get("type") == "semi-voyelle"
        }
    return _semi_voyelles
