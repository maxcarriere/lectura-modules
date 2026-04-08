"""lectura-tts — Abstraction multi-moteurs TTS pour Lectura.

Exports publics :
    - TTSEngine, TTSResult, PhonemeTiming (models)
    - InputType, Granularity, SyllabeTiming, GroupeTiming (models)
    - EngineInfo, EngineParam (registry)
    - register, get_all_engines, get_available_engines, is_available,
      get_engine_info, create_engine (registry)
    - iter_phonemes (ipa)
    - align_tts_to_syllables (aligner)
"""

from lectura_tts.models import (
    Granularity,
    GroupeTiming,
    InputType,
    PhonemeTiming,
    SyllabeTiming,
    TTSEngine,
    TTSResult,
)
from lectura_tts.registry import (
    EngineInfo,
    EngineParam,
    create_engine,
    get_all_engines,
    get_available_engines,
    get_engine_info,
    is_available,
    register,
)

# Charger tous les moteurs → déclenche register() pour chacun
import lectura_tts.engines  # noqa: F401

__all__ = [
    "TTSEngine",
    "TTSResult",
    "PhonemeTiming",
    "InputType",
    "Granularity",
    "SyllabeTiming",
    "GroupeTiming",
    "EngineInfo",
    "EngineParam",
    "register",
    "get_all_engines",
    "get_available_engines",
    "is_available",
    "get_engine_info",
    "create_engine",
]
