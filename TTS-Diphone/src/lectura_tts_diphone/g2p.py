"""G2P flexible — Protocol + backends Local/Nlp/Api/Custom.

Convertit du texte en groupes prosodiques avec phones IPA.

Backends (ordre de detection) :
  1. LecturaLocalG2P : lecteur_syllabique.Pipeline (Lectura Edition)
  2. LecturaNlpG2P : lectura_nlp + lectura_tokeniseur (pip install)
  3. LecturaApiG2P : POST /g2p/analyser (urllib, zero deps)
  4. CallableG2P : wraps any callable(text) → list[dict]

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
_PUNCT_MAP = {"...": "suspensive", ",": "comma", ".": "period", "?": "question", "!": "exclamation"}
_PUNCT_BOUNDARIES = {"comma", "period", "question", "exclamation", "suspensive"}

# Ponctuation terminale de phrase
_SENTENCE_PUNCT = {".", "?", "!", "\u2026", "..."}


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
    """G2P via lecteur_syllabique.Pipeline (Lectura Edition complet)."""

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


class LecturaNlpG2P:
    """G2P via lectura_nlp + lectura_tokeniseur (modules PyPI standards)."""

    def __init__(self) -> None:
        self._g2p = None

    def _ensure_loaded(self) -> None:
        if self._g2p is not None:
            return
        from lectura_nlp import creer_engine
        self._g2p = creer_engine(mode="auto")

    def phonemize(self, text: str) -> list[dict]:
        """Texte → groupes prosodiques via tokeniseur + G2P + formules."""
        self._ensure_loaded()
        from lectura_tokeniseur import tokenise
        from lectura_tts_diphone.phonemes import ipa_to_phones

        input_text = text.strip()
        if not input_text:
            return []

        tokens_raw = tokenise(input_text)

        # Enrichir les formules si le module est disponible
        try:
            from lectura_formules import enrichir_formules
            enrichir_formules(tokens_raw)
        except ImportError:
            pass

        # Extraire mots, formules et ponctuation dans l'ordre
        # entries : liste ordonnee de {"type": "mot"/"formule", ...}
        entries: list[dict] = []
        punct_after: dict[int, str] = {}

        for tok in tokens_raw:
            ttype = tok.type.value if hasattr(tok.type, "value") else tok.type.name
            if ttype in ("mot", "MOT"):
                entries.append({"type": "mot", "text": tok.text})
            elif ttype in ("formule", "FORMULE"):
                lecture = getattr(tok, "lecture", None)
                if lecture and getattr(lecture, "phone", ""):
                    # IPA pre-calculee par lectura_formules
                    for comp_ipa in lecture.phone.split():
                        entries.append({"type": "formule", "ipa": comp_ipa})
                else:
                    # Pas de lecture dispo — traiter comme mot
                    entries.append({"type": "mot", "text": tok.text})
            elif ttype in ("ponctuation", "PONCTUATION"):
                bnd = _PUNCT_MAP.get(tok.text.strip())
                if bnd and entries:
                    punct_after[len(entries) - 1] = bnd

        if not entries:
            return []

        # G2P batch pour les mots uniquement
        mot_indices = [i for i, e in enumerate(entries) if e["type"] == "mot"]
        mot_words = [entries[i]["text"] for i in mot_indices]

        if mot_words:
            result = self._g2p.analyser(mot_words)
            ipas = result.get("g2p", [])
            for idx, ipa in zip(mot_indices, ipas):
                entries[idx]["ipa"] = ipa

        # Construire groupes prosodiques
        groups: list[dict] = []
        current_phones: list[str] = []
        current_word_boundaries: list[int] = []

        for i, entry in enumerate(entries):
            ipa = entry.get("ipa", "")
            if not ipa:
                continue
            phones = ipa_to_phones(ipa)
            if not phones:
                continue

            # Marquer la frontiere de mot pour micro-pauses
            if current_phones:
                current_word_boundaries.append(len(current_phones))
            current_phones.extend(phones)

            bnd = punct_after.get(i)
            if bnd:
                groups.append({
                    "phones": current_phones,
                    "boundary": bnd,
                    "word_boundaries": current_word_boundaries,
                })
                current_phones = []
                current_word_boundaries = []

        # Dernier groupe sans ponctuation finale
        if current_phones:
            groups.append({
                "phones": current_phones,
                "boundary": "period",
                "word_boundaries": current_word_boundaries,
            })

        return groups


class LecturaApiG2P:
    """G2P via API Lectura (urllib seul, zero deps)."""

    DEFAULT_URL = "https://api.lec-tu-ra.com"

    def __init__(self, api_url: str | None = None) -> None:
        self._api_url = (api_url or self.DEFAULT_URL).rstrip("/")

    def phonemize(self, text: str) -> list[dict]:
        """Texte → groupes prosodiques via API."""
        from lectura_tts_diphone.phonemes import ipa_to_phones

        # Tokeniser le texte cote client (split basique)
        input_text = text.strip()
        if not input_text:
            return []

        # Split en mots + ponctuation
        import re
        parts = re.split(r'(\s+|[.,?!]+|\.\.\.)', input_text)
        words = []
        punct_after: dict[int, str] = {}

        for part in parts:
            part = part.strip()
            if not part:
                continue
            bnd = _PUNCT_MAP.get(part)
            if bnd:
                if words:
                    punct_after[len(words) - 1] = bnd
            elif not part.isspace():
                words.append(part)

        if not words:
            return []

        # Appel API G2P
        url = f"{self._api_url}/g2p/analyser"
        payload = json.dumps({"tokens": words}).encode("utf-8")
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

        ipas = data.get("g2p", [])

        # Construire groupes prosodiques
        groups: list[dict] = []
        current_phones: list[str] = []

        for i, ipa in enumerate(ipas):
            phones = ipa_to_phones(ipa)
            if not phones:
                continue
            current_phones.extend(phones)

            bnd = punct_after.get(i)
            if bnd:
                groups.append({"phones": current_phones, "boundary": bnd})
                current_phones = []

        if current_phones:
            groups.append({"phones": current_phones, "boundary": "period"})

        return groups


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

    Cascade :
      1. lecteur_syllabique (Lectura Edition complet)
      2. lectura_nlp + lectura_tokeniseur (modules PyPI)
      3. API Lectura (zero deps)

    Args:
        preference: "local", "api", ou None (auto)
        api_url: URL pour le backend API

    Returns:
        G2PBackend instance
    """
    if preference == "api":
        return LecturaApiG2P(api_url=api_url)

    if preference == "local" or preference is None:
        # 1. lecteur_syllabique (pipeline complet Lectura Edition)
        try:
            import lecteur_syllabique  # noqa: F401
            log.debug("G2P local disponible (lecteur_syllabique)")
            return LecturaLocalG2P()
        except ImportError:
            pass

        # 2. lectura_nlp + lectura_tokeniseur (modules PyPI)
        try:
            import lectura_nlp  # noqa: F401
            import lectura_tokeniseur  # noqa: F401
            log.debug("G2P disponible (lectura_nlp + lectura_tokeniseur)")
            return LecturaNlpG2P()
        except ImportError:
            if preference == "local":
                raise ImportError(
                    "G2P local non disponible. Installez avec : "
                    "pip install 'lectura-tts-diphone[g2p]'"
                )
            log.debug("G2P local non disponible, fallback vers API")

    return LecturaApiG2P(api_url=api_url)
