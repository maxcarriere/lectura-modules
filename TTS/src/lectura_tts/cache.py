"""Cache LRU thread-safe pour résultats TTS."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from lectura_tts.models import TTSResult


class SynthesisCache:
    """Cache LRU thread-safe pour résultats TTS."""

    def __init__(self, max_entries: int = 500):
        self._cache: OrderedDict[str, TTSResult] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries

    def get(self, key: str) -> TTSResult | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, result: TTSResult) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = result
            else:
                self._cache[key] = result
                if len(self._cache) > self._max:
                    self._cache.popitem(last=False)


# Singleton global
_global_cache = SynthesisCache()


class CachedTTSEngine:
    """Wrapper transparent avec cache."""

    def __init__(self, wrapped, engine_key: str, params: dict):
        self._wrapped = wrapped
        self._engine_key = engine_key
        # Clé de cache basée sur TOUS les paramètres (évite les oublis par nom)
        self._params_hash = hashlib.md5(
            str(sorted(params.items())).encode()
        ).hexdigest()[:12]

    def _make_key(self, text: str, input_type: str) -> str:
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{self._engine_key}::{input_type}::{self._params_hash}::{text_hash}"

    def synthesize(self, text: str) -> TTSResult:
        key = self._make_key(text, "text")
        cached = _global_cache.get(key)
        if cached is not None:
            return cached
        result = self._wrapped.synthesize(text)
        _global_cache.put(key, result)
        return result

    def synthesize_phonemes(self, phonemes_ipa: str) -> TTSResult:
        key = self._make_key(phonemes_ipa, "phonemes")
        cached = _global_cache.get(key)
        if cached is not None:
            return cached
        result = self._wrapped.synthesize_phonemes(phonemes_ipa)
        _global_cache.put(key, result)
        return result
