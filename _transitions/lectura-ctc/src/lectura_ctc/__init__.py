"""lectura-ctc est deprecie. Utiliser lectura-decodeur a la place."""

import warnings as _w
_w.warn(
    "lectura-ctc est deprecie. Installer lectura-decodeur a la place : "
    "pip install lectura-decodeur",
    DeprecationWarning, stacklevel=2,
)
del _w

from lectura_decodeur import *  # noqa: F401, F403
from lectura_decodeur import __version__
