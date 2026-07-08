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


def unites_mesure() -> dict[str, tuple[str, str]]:
    """Retourne les unites de mesure sous forme de dict[str, tuple[str, str]]."""
    raw = _charger()["unites_mesure"]
    return {k: tuple(v) for k, v in raw.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Tables inverses (reconnaissance)
# ══════════════════════════════════════════════════════════════════════════════

_inv_phone_nombre_cache: dict[str, tuple[int, str]] | None = None
_inv_fr_nombre_cache: dict[str, tuple[int, str]] | None = None
_inv_phone_mois_cache: dict[str, int] | None = None
_inv_fr_mois_cache: dict[str, int] | None = None
_inv_phone_symbole_cache: dict[str, str] | None = None
_inv_fr_symbole_cache: dict[str, str] | None = None
_inv_phone_lettre_cache: dict[str, str] | None = None
_inv_fr_lettre_cache: dict[str, str] | None = None
_inv_phone_grec_cache: dict[str, str] | None = None
_inv_phone_devise_cache: dict[str, tuple[str, str]] | None = None
_inv_phone_ordinal_cache: dict[str, str] | None = None
_inv_phone_direction_cache: dict[str, str] | None = None


def inv_phone_nombre() -> dict[str, tuple[int, str]]:
    """IPA -> (valeur_numerique, cle_dans_unites).

    Ex: "kaʁɑ̃t" -> (40, "40"), "e ɛ̃" -> (1, "et_1")
    """
    global _inv_phone_nombre_cache
    if _inv_phone_nombre_cache is None:
        raw = _charger()["unites"]
        table: dict[str, tuple[int, str]] = {}
        for key, entry in raw.items():
            phone = entry[1]
            val = entry[2]
            if phone not in table:
                table[phone] = (val, key)
        _inv_phone_nombre_cache = table
    return _inv_phone_nombre_cache


def inv_fr_nombre() -> dict[str, tuple[int, str]]:
    """Texte francais -> (valeur_numerique, cle_dans_unites).

    Ex: "quarante" -> (40, "40"), "et un" -> (1, "et_1")
    """
    global _inv_fr_nombre_cache
    if _inv_fr_nombre_cache is None:
        raw = _charger()["unites"]
        table: dict[str, tuple[int, str]] = {}
        for key, entry in raw.items():
            fr = entry[0]
            val = entry[2]
            if fr not in table:
                table[fr] = (val, key)
        _inv_fr_nombre_cache = table
    return _inv_fr_nombre_cache


def inv_phone_mois() -> dict[str, int]:
    """IPA -> numero de mois (1-12)."""
    global _inv_phone_mois_cache
    if _inv_phone_mois_cache is None:
        raw = _charger()["mois"]
        _inv_phone_mois_cache = {v[1]: int(k) for k, v in raw.items()}
    return _inv_phone_mois_cache


def inv_fr_mois() -> dict[str, int]:
    """Texte francais -> numero de mois (1-12)."""
    global _inv_fr_mois_cache
    if _inv_fr_mois_cache is None:
        raw = _charger()["mois"]
        _inv_fr_mois_cache = {v[0]: int(k) for k, v in raw.items()}
    return _inv_fr_mois_cache


def inv_phone_symbole() -> dict[str, str]:
    """IPA -> caractere symbole."""
    global _inv_phone_symbole_cache
    if _inv_phone_symbole_cache is None:
        raw = _charger()["symboles"]
        table: dict[str, str] = {}
        for sym, entry in raw.items():
            phone = entry[1]
            if phone not in table:
                table[phone] = sym
        _inv_phone_symbole_cache = table
    return _inv_phone_symbole_cache


def inv_fr_symbole() -> dict[str, str]:
    """Texte francais -> caractere symbole."""
    global _inv_fr_symbole_cache
    if _inv_fr_symbole_cache is None:
        raw = _charger()["symboles"]
        table: dict[str, str] = {}
        for sym, entry in raw.items():
            fr = entry[0]
            if fr not in table:
                table[fr] = sym
        _inv_fr_symbole_cache = table
    return _inv_fr_symbole_cache


def inv_phone_lettre() -> dict[str, str]:
    """IPA -> lettre (uppercase)."""
    global _inv_phone_lettre_cache
    if _inv_phone_lettre_cache is None:
        raw = _charger()["lettres"]
        table: dict[str, str] = {}
        for letter, entry in raw.items():
            phone = entry[1]
            # Preference pour les majuscules
            if phone not in table or letter.isupper():
                table[phone] = letter.upper()
        _inv_phone_lettre_cache = table
    return _inv_phone_lettre_cache


def inv_fr_lettre() -> dict[str, str]:
    """Texte francais -> lettre (uppercase)."""
    global _inv_fr_lettre_cache
    if _inv_fr_lettre_cache is None:
        raw = _charger()["lettres"]
        table: dict[str, str] = {}
        for letter, entry in raw.items():
            fr = entry[0]
            if fr not in table or letter.isupper():
                table[fr] = letter.upper()
        _inv_fr_lettre_cache = table
    return _inv_fr_lettre_cache


def inv_phone_grec() -> dict[str, str]:
    """IPA -> caractere grec (lowercase prefere)."""
    global _inv_phone_grec_cache
    if _inv_phone_grec_cache is None:
        raw = _charger()["grec"]
        table: dict[str, str] = {}
        for char, entry in raw.items():
            phone = entry[1]
            if phone not in table or char.islower():
                table[phone] = char
        _inv_phone_grec_cache = table
    return _inv_phone_grec_cache


def inv_phone_devise() -> dict[str, tuple[str, str]]:
    """IPA -> (code_ISO, "major"/"minor").

    Ex: "øʁo" -> ("EUR", "major"), "sɑ̃tim" -> ("EUR", "minor")
    """
    global _inv_phone_devise_cache
    if _inv_phone_devise_cache is None:
        raw = _charger()["devises"]
        table: dict[str, tuple[str, str]] = {}
        for code, info in raw.items():
            for phone in info.get("phone_maj", []):
                if phone and phone not in table:
                    table[phone] = (code, "major")
            if info.get("phone_min"):
                for phone in info["phone_min"]:
                    if phone and phone not in table:
                        table[phone] = (code, "minor")
        _inv_phone_devise_cache = table
    return _inv_phone_devise_cache


def inv_phone_ordinal() -> dict[str, str]:
    """IPA suffixe ordinal -> cardinal de base.

    Ex: "dozjɛm" -> "douze"
    """
    global _inv_phone_ordinal_cache
    if _inv_phone_ordinal_cache is None:
        raw = _charger()["ordinaux"]
        table: dict[str, str] = {}
        for cardinal, entry in raw.items():
            if cardinal.startswith("_"):
                continue
            phone = entry[1]
            if phone not in table:
                table[phone] = cardinal
        _inv_phone_ordinal_cache = table
    return _inv_phone_ordinal_cache


def inv_phone_direction() -> dict[str, str]:
    """IPA -> direction GPS (N/S/E/O)."""
    global _inv_phone_direction_cache
    if _inv_phone_direction_cache is None:
        raw = _charger()["gps_directions"]
        table: dict[str, str] = {}
        for code, entry in raw.items():
            phone = entry[1]
            # W et O mappent au meme IPA, preference pour O
            if phone not in table or code == "O":
                table[phone] = code
        _inv_phone_direction_cache = table
    return _inv_phone_direction_cache
