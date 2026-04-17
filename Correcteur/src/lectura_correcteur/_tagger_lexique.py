"""Tagger fallback par lookup lexique — zero modele, zero dependance.

Pour chaque mot, interroge lexique.info() et retourne le POS/morpho
de l'entree la plus frequente. Pas de desambiguation contextuelle :
"mange" sera toujours tague selon son entree la plus frequente.

Utilise comme fallback quand aucun modele POS/Morpho n'est injecte.
"""

from __future__ import annotations

import re
from typing import Any

# Regex tokenisation française (elisions + mots + ponctuation)
_TOKEN_RE = re.compile(
    r"(?:[dlnmtscjqDLNMTSCJQ]|[Qq]u|[Ll]orsqu|[Pp]uisqu|[Jj]usqu|[Qq]uelqu)"
    r"['\u2019]"
    r"|[\w]+(?:-[\w]+)*"
    r"|[^\s\w]+",
)


class LexiqueTagger:
    """Tagger POS/Morpho par simple lookup lexique (plus frequente entree).

    Satisfait TaggerProtocol.
    """

    def __init__(self, lexique: Any) -> None:
        self._lexique = lexique

    def tokenize(self, text: str) -> list[tuple[str, bool]]:
        """Tokenise via regex."""
        result: list[tuple[str, bool]] = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group()
            is_word = tok[0].isalpha() or tok[0] == "_"
            result.append((tok, is_word))
        return result

    def tag_words(self, words: list[str]) -> list[dict]:
        """Tague chaque mot par lookup lexique (entree la plus frequente)."""
        results: list[dict] = []
        for word in words:
            d: dict[str, str] = {}
            infos = self._lexique.info(word.lower()) if hasattr(self._lexique, "info") else []
            if infos:
                best = max(infos, key=lambda e: float(e.get("freq") or 0))
                if best.get("cgram"):
                    d["pos"] = best["cgram"]
                for feat in ("genre", "nombre", "temps", "mode", "personne"):
                    val = best.get(feat)
                    if val is not None:
                        d[feat] = val
            results.append(d)
        return results
