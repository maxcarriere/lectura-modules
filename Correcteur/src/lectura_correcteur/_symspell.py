"""Index SymSpell pour la generation rapide de candidats.

Pre-calcule les deletions a distance 1 de chaque mot du lexique a l'init.
A la requete, genere delete-1 et delete-2 du mot inconnu pour un lookup O(1).
"""

from __future__ import annotations

from typing import Any


class SymSpellIndex:
    """Index de deletion pre-calcule pour suggestions rapides.

    A la construction, genere toutes les delete-1 de chaque forme du lexique
    et les stocke dans un dict inverse {deletion: {formes_originales}}.

    A la requete, genere delete-1 et delete-2 du mot d'entree et cherche
    les correspondances dans l'index.
    """

    __slots__ = ("_index", "_formes")

    def __init__(self, formes: frozenset[str]) -> None:
        """Construit l'index a partir d'un ensemble de formes.

        Args:
            formes: Ensemble de toutes les formes du lexique (minuscules).
        """
        self._formes = formes
        self._index: dict[str, set[str]] = {}
        for mot in formes:
            # Indexer le mot lui-meme (pour trouver les insertions d=1)
            self._index.setdefault(mot, set()).add(mot)
            # Indexer ses deletions d=1
            for d in _deletions(mot):
                self._index.setdefault(d, set()).add(mot)

    _EMPTY: frozenset[str] = frozenset()

    def suggestions(self, mot: str, max_n: int = 500) -> list[str]:
        """Genere des candidats pour un mot inconnu.

        Strategie :
        - Le mot lui-meme (lookup direct, pour trouver les mots a d=0)
        - Delete-1 du mot d'entree → trouve substitutions et deletions d=1
        - Delete-2 du mot d'entree → trouve candidats a d<=2
        - Le mot d'entree comme cle dans l'index → trouve insertions d=1

        Args:
            mot: Mot a corriger (minuscule).
            max_n: Nombre max de candidats retournes.

        Returns:
            Liste de formes candidates (sans doublons).
        """
        low = mot.lower()
        seen: set[str] = set()
        result: list[str] = []
        _empty = self._EMPTY

        def _add(candidates) -> None:
            for c in candidates:
                if c not in seen and c != low:
                    seen.add(c)
                    result.append(c)
                    if len(result) >= max_n:
                        return

        # d=0 : le mot lui-meme dans l'index (insertions)
        _add(self._index.get(low, _empty))

        # d=1 : deletions du mot d'entree
        del1 = _deletions(low)
        for d in del1:
            _add(self._index.get(d, _empty))
            if len(result) >= max_n:
                break

        # d=2 : deletions des deletions
        if len(result) < max_n:
            for d in del1:
                for d2 in _deletions(d):
                    _add(self._index.get(d2, _empty))
                    if len(result) >= max_n:
                        break
                if len(result) >= max_n:
                    break

        return result[:max_n]


def _deletions(mot: str) -> list[str]:
    """Genere toutes les formes obtenues en supprimant 1 caractere."""
    return [mot[:i] + mot[i + 1:] for i in range(len(mot))]


def _obtenir_formes(lexique: Any) -> frozenset[str] | None:
    """Extrait l'ensemble des formes d'un lexique par duck typing.

    Essaie dans l'ordre :
    1. lexique.toutes_formes() -> frozenset
    2. lexique._formes (frozenset ou dict.keys())

    Retourne None si l'extraction echoue.
    """
    if hasattr(lexique, "toutes_formes"):
        try:
            return frozenset(lexique.toutes_formes())
        except Exception:
            pass
    if hasattr(lexique, "_formes"):
        f = lexique._formes
        if isinstance(f, frozenset):
            return f
        if isinstance(f, dict):
            return frozenset(f.keys())
    if hasattr(lexique, "_formes_set"):
        f = lexique._formes_set
        if isinstance(f, frozenset):
            return f
    return None
