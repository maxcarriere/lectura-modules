"""G2P flexible — Protocol + backends Local/Api/Custom.

Convertit du texte en groupes prosodiques avec phones IPA.

Backends :
  - LecturaLocalG2P : import lecteur_syllabique.Pipeline (lectura-g2p)
  - LecturaApiG2P : POST /g2p/phonemize (urllib, zero deps)
  - CallableG2P : wraps any callable(text) → list[dict]
  - auto_detect_g2p() : local si disponible, sinon API

Chaque backend retourne une liste de groupes prosodiques :
  [{"phones": ["b", "ɔ̃"], "boundary": "comma"}, ...]
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Callable, Protocol, runtime_checkable

log = logging.getLogger(__name__)

# Ponctuation → boundary type
_PUNCT_MAP = {",": "comma", ".": "period", "?": "question", "!": "exclamation"}
_PUNCT_BOUNDARIES = {"comma", "period", "question", "exclamation"}


@runtime_checkable
class G2PBackend(Protocol):
    """Protocole pour un backend G2P."""

    def phonemize(self, text: str) -> list[dict]:
        """Convertit du texte en groupes prosodiques.

        Returns:
            Liste de dicts avec :
                - "phones": list[str] — phones IPA du groupe
                - "boundary": str — type de frontiere apres le groupe
                  ("comma", "period", "question", "exclamation", "none")
        """
        ...


class LecturaLocalG2P:
    """G2P via lecteur_syllabique.Pipeline (import local)."""

    def __init__(self) -> None:
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from lecteur_syllabique.core.pipeline import Pipeline
        self._pipeline = Pipeline()

    def phonemize(self, text: str) -> list[dict]:
        """Texte → groupes prosodiques via Lectura Pipeline."""
        self._ensure_loaded()
        from lectura_tts_diphone.phonemes import ipa_to_phones

        input_text = text.strip()
        if not input_text:
            return []
        # Ajouter ponctuation finale si absente
        if input_text[-1] not in ".?!":
            input_text += "."

        result = self._pipeline.run_steps(input_text)
        groups_out = result.presentation.syllabified

        # Collecter syllabes + boundaries par groupe
        syllables: list[str] = []
        boundaries: list[str] = []

        for gi, group in enumerate(groups_out):
            boundary = "none"

            if gi + 1 < len(groups_out):
                # Detecter ponctuation entre groupes
                next_g = groups_out[gi + 1]
                try:
                    cur_end = max(c.span[1] for c in group.components
                                  if hasattr(c, "span"))
                    nxt_start = min(c.span[0] for c in next_g.components
                                    if hasattr(c, "span"))
                    between = input_text[cur_end:nxt_start]
                except (ValueError, AttributeError):
                    between = ""

                for char, btype in _PUNCT_MAP.items():
                    if char in between:
                        boundary = btype
                        break

                if boundary == "none":
                    boundary = "word"
            else:
                # Dernier groupe : ponctuation terminale
                try:
                    cur_end = max(c.span[1] for c in group.components
                                  if hasattr(c, "span"))
                    trailing = input_text[cur_end:]
                    for char, btype in _PUNCT_MAP.items():
                        if char in trailing:
                            boundary = btype
                            break
                except (ValueError, AttributeError):
                    pass

            n_syls = len(group.syllabes)
            for si, syl in enumerate(group.syllabes):
                syllables.append(syl.phone)
                boundaries.append(boundary if si == n_syls - 1 else "none")

        # Regrouper en groupes prosodiques (split sur ponctuation)
        prosodic_groups: list[dict] = []
        current_phones: list[str] = []
        current_word_boundaries: list[int] = []

        for si, syl in enumerate(syllables):
            current_phones.extend(ipa_to_phones(syl))
            bnd = boundaries[si]

            is_punct = bnd in _PUNCT_BOUNDARIES
            is_last = si == len(syllables) - 1

            if bnd == "word" and not is_last:
                current_word_boundaries.append(len(current_phones))

            if is_punct or is_last:
                if current_phones:
                    prosodic_groups.append({
                        "phones": current_phones,
                        "boundary": bnd,
                        "word_boundaries": current_word_boundaries[:],
                    })
                    current_phones = []
                    current_word_boundaries = []

        return prosodic_groups


class LecturaApiG2P:
    """G2P via API Lectura (urllib seul, zero deps)."""

    DEFAULT_URL = "http://localhost:8000"

    def __init__(self, api_url: str | None = None) -> None:
        self._api_url = (api_url or self.DEFAULT_URL).rstrip("/")

    def phonemize(self, text: str) -> list[dict]:
        """Texte → groupes prosodiques via API."""
        from lectura_tts_diphone.phonemes import ipa_to_phones

        url = f"{self._api_url}/g2p/phonemize"
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            log.error("G2P API error: %s", e)
            return []

        # L'API retourne des syllabes + boundaries
        syllables = data.get("syllables", [])
        boundaries = data.get("boundaries", [])

        # Regrouper en groupes prosodiques
        prosodic_groups: list[dict] = []
        current_phones: list[str] = []

        for si, syl in enumerate(syllables):
            current_phones.extend(ipa_to_phones(syl))
            bnd = boundaries[si] if si < len(boundaries) else "none"

            is_punct = bnd in _PUNCT_BOUNDARIES
            is_last = si == len(syllables) - 1

            if is_punct or is_last:
                if current_phones:
                    prosodic_groups.append({
                        "phones": current_phones,
                        "boundary": bnd,
                    })
                    current_phones = []

        return prosodic_groups


class CallableG2P:
    """Wrapper pour un callable custom (text) → list[dict]."""

    def __init__(self, fn: Callable[[str], list[dict]]) -> None:
        self._fn = fn

    def phonemize(self, text: str) -> list[dict]:
        return self._fn(text)


def auto_detect_g2p(
    preference: str | None = None,
    api_url: str | None = None,
) -> G2PBackend:
    """Auto-detecte le meilleur backend G2P disponible.

    Args:
        preference: "local", "api", ou None (auto)
        api_url: URL pour le backend API

    Returns:
        G2PBackend instance
    """
    if preference == "api":
        return LecturaApiG2P(api_url=api_url)

    if preference == "local" or preference is None:
        try:
            import lecteur_syllabique  # noqa: F401
            log.debug("G2P local disponible (lecteur_syllabique)")
            return LecturaLocalG2P()
        except ImportError:
            if preference == "local":
                raise ImportError(
                    "lecteur_syllabique non installe. "
                    "Installez lectura-g2p ou utilisez preference='api'."
                )
            log.debug("G2P local non disponible, fallback vers API")

    return LecturaApiG2P(api_url=api_url)
