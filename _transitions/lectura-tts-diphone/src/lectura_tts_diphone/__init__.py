"""lectura-tts-diphone est deprecie. Utiliser lectura-diphone a la place."""

import warnings as _w
_w.warn(
    "lectura-tts-diphone est deprecie. Installer lectura-diphone a la place : "
    "pip install lectura-diphone",
    DeprecationWarning, stacklevel=2,
)
del _w

from lectura_diphone import *  # noqa: F401, F403
from lectura_diphone import __version__
