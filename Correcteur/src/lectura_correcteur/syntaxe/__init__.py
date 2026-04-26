"""Sous-module syntaxe : ponctuation, segmentation et elision."""

from lectura_correcteur.syntaxe._elision import appliquer_elision
from lectura_correcteur.syntaxe._ponctuation import verifier_ponctuation
from lectura_correcteur.syntaxe._segmentation import segmenter_phrases

__all__ = [
    "appliquer_elision",
    "appliquer_syntaxe",
    "segmenter_phrases",
    "verifier_ponctuation",
]


def appliquer_syntaxe(
    tokens: list[str],
) -> tuple[list[str], list]:
    """Applique toutes les regles syntaxiques sur les tokens.

    Returns:
        Tuple (tokens_corriges, corrections).
    """
    result = list(tokens)
    corrections = []

    corr_ponct = verifier_ponctuation(result)
    corrections.extend(corr_ponct)

    # Elision desactivee (trop agressive sur corpus bien forme)
    # result = appliquer_elision(result)

    return result, corrections
