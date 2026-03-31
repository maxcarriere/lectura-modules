"""Utilitaires partages pour le lexique."""

from __future__ import annotations


def normaliser_ortho(mot: str) -> str:
    """Normalise un mot pour comparaison : strip + lower."""
    return mot.strip().lower()
