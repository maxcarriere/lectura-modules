"""lectura-correcteur : correcteur orthographique et grammatical Lectura.

Pipeline : Tokenisation -> Syntaxe -> Resegmentation -> Morpho CRF
           -> Orthographe -> Grammaire -> Reconstruction

Usage rapide::

    from lectura_lexique import Lexique
    from lectura_correcteur import Correcteur

    lex = Lexique("lexique.db")
    correcteur = Correcteur(lex)
    result = correcteur.corriger("Les enfant mange des pomme.")
    print(result.phrase_corrigee)  # "Les enfants mangent des pommes."
"""

__version__ = "1.0.0"

from lectura_correcteur._types import (
    Correction,
    MotAnalyse,
    ResultatCorrection,
    TaggerProtocol,
    TokeniseurProtocol,
    TypeCorrection,
)
from lectura_correcteur.correcteur import Correcteur

__all__ = [
    "Correcteur",
    "Correction",
    "MotAnalyse",
    "ResultatCorrection",
    "TaggerProtocol",
    "TokeniseurProtocol",
    "TypeCorrection",
]
