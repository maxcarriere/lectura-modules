"""lectura-lexique : module partage d'acces au lexique Lectura.

Usage::

    from lectura_lexique import Lexique

    with Lexique("lexique.db") as lex:
        print(lex.existe("chat"))       # True
        print(lex.phone_de("chat"))     # "ʃa"
        print(lex.homophones("ʃa"))     # [{"ortho": "chat", ...}, ...]
"""

__version__ = "1.3.0"

from lectura_lexique._types import (
    Categorie,
    Concept,
    EntreeForme,
    EntreeLemme,
    EntreeLexicale,
    LexiqueProtocol,
    SensDefinition,
)
from lectura_lexique._utils import generer_phones_proches, normaliser_ortho
from lectura_lexique.lexique import Lexique

__all__ = [
    "Categorie",
    "Concept",
    "EntreeForme",
    "EntreeLemme",
    "EntreeLexicale",
    "Lexique",
    "LexiqueProtocol",
    "SensDefinition",
    "generer_phones_proches",
    "normaliser_ortho",
]
