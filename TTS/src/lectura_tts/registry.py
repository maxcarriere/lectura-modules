"""Registre de moteurs TTS avec auto-enregistrement.

Chaque fichier engines/*.py appelle register() à l'import pour
s'enregistrer dans le registre global.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from lectura_tts.models import TTSEngine


@dataclass(frozen=True)
class EngineParam:
    """Décrit un paramètre configurable d'un moteur TTS."""

    name: str              # clé machine ("speed", "pitch")
    label: str             # label GUI ("Vitesse (mots/min)")
    type: str              # "int" | "float" | "str" | "bool" | "choice" | "file"
    default: Any = None
    choices: list[str] = field(default_factory=list)
    min_val: float | None = None
    max_val: float | None = None
    role: str = ""         # "voice", "speed", "pitch", ou "" (avancé)
    file_filter: str = ""  # filtre pour QFileDialog (type="file")


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
    category: str = "optional"            # "cloud" ou "optional"
    install_command: str = ""             # ex: "pip install lectura-tts[piper]"
    uninstall_packages: list[str] = field(default_factory=list)
    uninstall_command: str = ""           # ex: "sudo apt remove espeak-ng"
    model_urls: list[tuple[str, str]] = field(default_factory=list)
    # Liste de (url_téléchargement, chemin_destination) pour les fichiers modèle
    check_modules: list[str] = field(default_factory=list)
    # Modules à purger de sys.modules pour re-tester is_available()
    pip_packages: list[str] = field(default_factory=list)
    # Packages PyPI top-level à télécharger (dépendances transitives résolues dynamiquement)
    license_notice: str = ""
    # Disclaimer affiché avant installation en mode frozen


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
    engine = info.factory(params or {})
    from lectura_tts.cache import CachedTTSEngine
    return CachedTTSEngine(engine, key, params or {})


def get_cloud_engines() -> list[EngineInfo]:
    """Moteurs cloud, toujours affichés."""
    return [info for info in _registry.values() if info.category == "cloud"]


def get_optional_engines() -> list[EngineInfo]:
    """Moteurs optionnels, affichés seulement si installés."""
    return [info for info in _registry.values() if info.category == "optional"]


def get_builtin_engines() -> list[EngineInfo]:
    """Moteurs intégrés (subprocess / système), toujours affichés."""
    return [info for info in _registry.values() if info.category == "builtin"]


def get_extension_engines() -> list[EngineInfo]:
    """Moteurs extensions, installables à la demande."""
    return [info for info in _registry.values() if info.category == "extension"]


def get_primary_params(key: str) -> list[EngineParam]:
    """Params avec role non-vide (voice/speed/pitch)."""
    info = get_engine_info(key)
    return [p for p in info.params if p.role] if info else []


def get_advanced_params(key: str) -> list[EngineParam]:
    """Params sans role (avancés, settings only)."""
    info = get_engine_info(key)
    return [p for p in info.params if not p.role] if info else []


def invalidate_check(key: str) -> None:
    """Purge sys.modules des modules listés dans check_modules pour un moteur.

    Cela force check_available() à ré-importer le module au prochain appel,
    ce qui est nécessaire après pip install/uninstall.
    """
    import importlib

    info = _registry.get(key)
    if info is None:
        return
    for mod_name in info.check_modules:
        # Purger le module et tous ses sous-modules
        to_remove = [k for k in sys.modules if k == mod_name or k.startswith(mod_name + ".")]
        for k in to_remove:
            del sys.modules[k]
    importlib.invalidate_caches()
