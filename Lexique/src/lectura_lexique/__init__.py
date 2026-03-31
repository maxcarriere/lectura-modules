"""lectura-lexique : module partage d'acces au lexique Lectura.

Usage::

    from lectura_lexique import Lexique

    with Lexique("lexique.db") as lex:
        print(lex.existe("chat"))       # True
        print(lex.phone_de("chat"))     # "ʃa"
        print(lex.homophones("ʃa"))     # [{"ortho": "chat", ...}, ...]
"""

__version__ = "1.0.0"

from lectura_lexique._types import EntreeLexicale, LexiqueProtocol
from lectura_lexique._utils import normaliser_ortho
from lectura_lexique.lexique import Lexique

__all__ = [
    "EntreeLexicale",
    "Lexique",
    "LexiqueProtocol",
    "normaliser_ortho",
]
