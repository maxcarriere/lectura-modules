"""Fonctions de semantique : synonymes, antonymes, definition."""

from __future__ import annotations

from typing import Any


def synonymes(entries: list[dict[str, Any]]) -> list[str]:
    """Parse la colonne 'synonymes' (separateur ;).

    Retourne [] si colonne absente ou vide.
    """
    for e in entries:
        raw = e.get("synonymes", "")
        if raw:
            return [s.strip() for s in str(raw).split(";") if s.strip()]
    return []


def antonymes(entries: list[dict[str, Any]]) -> list[str]:
    """Parse la colonne 'antonymes' (separateur ;).

    Retourne [] si colonne absente ou vide.
    """
    for e in entries:
        raw = e.get("antonymes", "")
        if raw:
            return [s.strip() for s in str(raw).split(";") if s.strip()]
    return []


def definition(entries: list[dict[str, Any]]) -> list[str]:
    """Retourne les definitions distinctes.

    Retourne [] si colonne absente ou vide.
    """
    defs: list[str] = []
    seen: set[str] = set()
    for e in entries:
        raw = e.get("definition", "")
        if raw:
            d = str(raw).strip()
            if d and d not in seen:
                seen.add(d)
                defs.append(d)
    return defs
