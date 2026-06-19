"""lectura-tts-monospeaker est deprecie. Utiliser lectura-monospeaker a la place."""

import warnings as _w
_w.warn(
    "lectura-tts-monospeaker est deprecie. Installer lectura-monospeaker a la place : "
    "pip install lectura-monospeaker",
    DeprecationWarning, stacklevel=2,
)
del _w

from lectura_monospeaker import *  # noqa: F401, F403
from lectura_monospeaker import __version__
