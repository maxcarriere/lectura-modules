"""Segmentation en phrases (stub pour futur developpement)."""

from __future__ import annotations

import re


def segmenter_phrases(texte: str) -> list[str]:
    """Segmente un texte en phrases."""
    if not texte.strip():
        return []

    phrases = re.split(r'(?<=[.!?…])\s+', texte.strip())
    return [p for p in phrases if p.strip()]
