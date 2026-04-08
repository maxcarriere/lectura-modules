"""Registre de moteurs TTS avec auto-enregistrement.

Chaque fichier engines/*.py appelle register() à l'import pour
s'enregistrer dans le registre global.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from lectura_tts.models import TTSEngine


@dataclass(frozen=True)
class EngineParam:
    """Décrit un paramètre configurable d'un moteur TTS."""

    name: str              # clé machine ("speed", "pitch")
    label: str             # label GUI ("Vitesse (mots/min)")
    type: str              # "int" | "float" | "str" | "bool" | "choice"
    default: Any = None
    choices: list[str] = field(default_factory=list)
    min_val: float | None = None
    max_val: float | None = None


@dataclass(frozen=True)
class EngineInfo:
    """Informations complètes sur un moteur TTS."""

    key: str                              # "espeak", "edge", ...
    name: str                             # "eSpeak-NG"
    description: str
    supports_phonemes: bool
    supports_text: bool
    requires_internet: bool
    requires_api_key: bool
    install_instructions: str
    check_available: Callable[[], bool]
    factory: Callable[[dict], TTSEngine]
    params: list[EngineParam] = field(default_factory=list)


# ── Registre global ──

_registry: dict[str, EngineInfo] = {}


def register(info: EngineInfo) -> None:
    """Enregistre un moteur TTS dans le registre global."""
    _registry[info.key] = info


def get_all_engines() -> list[EngineInfo]:
    """Retourne la liste de tous les moteurs enregistrés."""
    return list(_registry.values())


def get_available_engines() -> list[EngineInfo]:
    """Retourne la liste des moteurs disponibles (installés)."""
    result = []
    for info in _registry.values():
        try:
            if info.check_available():
                result.append(info)
        except Exception:
            pass
    return result


def is_available(key: str) -> bool:
    """Vérifie si un moteur est enregistré et disponible."""
    info = _registry.get(key)
    if info is None:
        return False
    try:
        return info.check_available()
    except Exception:
        return False


def get_engine_info(key: str) -> EngineInfo | None:
    """Retourne les infos d'un moteur par sa clé."""
    return _registry.get(key)


def create_engine(key: str, params: dict | None = None) -> TTSEngine:
    """Crée une instance de moteur TTS par sa clé et ses paramètres."""
    info = _registry.get(key)
    if info is None:
        raise KeyError(f"Moteur TTS inconnu : {key!r}")
    return info.factory(params or {})
