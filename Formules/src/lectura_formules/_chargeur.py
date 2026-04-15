"""Chargeur paresseux des donnees metier de Formules.

Charge le fichier donnees_formules.json une seule fois (singleton)
et expose les constantes sous forme de fonctions.

En mode Niveau 1 (donnees absentes), leve FileNotFoundError
pour que le code appelant puisse deleguer a l'API.
"""

from __future__ import annotations

import json
from importlib import resources

_donnees: dict | None = None
_donnees_absentes: bool = False


def _charger() -> dict:
    global _donnees, _donnees_absentes
    if _donnees_absentes:
        raise FileNotFoundError("donnees_formules.json absent (mode API)")
    if _donnees is None:
        ref = resources.files("lectura_formules.data").joinpath("donnees_formules.json")
        try:
            _donnees = json.loads(ref.read_text(encoding="utf-8"))
        except FileNotFoundError:
            _donnees_absentes = True
            raise FileNotFoundError(
                "Fichier de donnees donnees_formules.json introuvable. "
                "Les appels seront delegues au serveur Lectura."
            ) from None
    return _donnees


def donnees_disponibles() -> bool:
    """Retourne True si les donnees locales sont chargees ou chargeables."""
    global _donnees, _donnees_absentes
    if _donnees is not None:
        return True
    if _donnees_absentes:
        return False
    try:
        _charger()
        return True
    except FileNotFoundError:
        return False


def unites() -> dict[str, tuple[str, str, int]]:
    """Retourne les unites sous forme de dict[str, tuple[str, str, int]]."""
    raw = _charger()["unites"]
    return {k: tuple(v) for k, v in raw.items()}


def lettres() -> dict[str, tuple[str, str]]:
    raw = _charger()["lettres"]
    return {k: tuple(v) for k, v in raw.items()}


def symboles() -> dict[str, tuple[str, str]]:
    raw = _charger()["symboles"]
    return {k: tuple(v) for k, v in raw.items()}


def grec() -> dict[str, tuple[str, str]]:
    raw = _charger()["grec"]
    return {k: tuple(v) for k, v in raw.items()}


def ordinaux() -> dict[str, tuple[str, str]]:
    raw = _charger()["ordinaux"]
    return {k: tuple(v) for k, v in raw.items()}


def mois() -> dict[int, tuple[str, str]]:
    raw = _charger()["mois"]
    return {int(k): tuple(v) for k, v in raw.items()}


def virgule() -> tuple[str, str]:
    return tuple(_charger()["virgule"])


def fois() -> tuple[str, str]:
    return tuple(_charger()["fois"])


def dix() -> tuple[str, str]:
    return tuple(_charger()["dix"])


def exposant() -> tuple[str, str]:
    return tuple(_charger()["exposant"])


def echelles() -> dict[int, tuple[str, str, bool]]:
    raw = _charger()["echelles"]
    return {int(k): tuple(v) for k, v in raw.items()}


def heure_words() -> dict[str, tuple[str, str]]:
    raw = _charger()["heure_words"]
    return {k: tuple(v) for k, v in raw.items()}


def devises() -> dict[str, dict]:
    return _charger()["devises"]


def pourcent_words() -> dict[str, tuple[str, str]]:
    raw = _charger()["pourcent"]
    return {k: tuple(v) for k, v in raw.items()}


def gps_directions() -> dict[str, tuple[str, str, str]]:
    raw = _charger()["gps_directions"]
    return {k: tuple(v) for k, v in raw.items()}


def gps_units() -> dict[str, tuple[str, str]]:
    raw = _charger()["gps_units"]
    return {k: tuple(v) for k, v in raw.items()}


def intervalle_bounds() -> set[str]:
    return set(_charger()["intervalle_bounds"])


def romains_int_to_roman() -> list[tuple[int, str]]:
    raw = _charger()["romains_int_to_roman"]
    return [tuple(pair) for pair in raw]


def romains_values() -> list[tuple[str, int]]:
    raw = _charger()["romains_values"]
    return [tuple(pair) for pair in raw]


def romains_single() -> dict[str, int]:
    return _charger()["romains_single"]
