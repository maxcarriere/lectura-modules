"""Stub _spellcheck — le source original a ete perdu.

Fournit les interfaces minimales pour que les imports ne cassent pas.
"""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FRENCH_ALPHABET = set("abcdefghijklmnopqrstuvwxyzàâæçéèêëîïôœùûüÿ-'")


class OrthoIndex:
    """Index orthographique pour la correction OOV (stub)."""

    def __init__(self, db_path: str | Path, **kwargs: Any):
        self._db_path = str(db_path)
        logger.debug("OrthoIndex stub loaded (no spellcheck available)")

    def suggest(self, word: str, **kwargs: Any) -> list[str]:
        return []


def spellcheck_oov(
    ortho_words: list[str],
    ortho_index: OrthoIndex | None = None,
    **kwargs: Any,
) -> list[str]:
    """Stub — retourne les mots inchanges."""
    return list(ortho_words)
