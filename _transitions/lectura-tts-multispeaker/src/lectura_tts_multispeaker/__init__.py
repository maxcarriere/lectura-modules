"""lectura-tts-multispeaker est deprecie. Utiliser lectura-multispeaker a la place."""

import warnings as _w
_w.warn(
    "lectura-tts-multispeaker est deprecie. Installer lectura-multispeaker a la place : "
    "pip install lectura-multispeaker",
    DeprecationWarning, stacklevel=2,
)
del _w

from lectura_multispeaker import *  # noqa: F401, F403
from lectura_multispeaker import __version__
