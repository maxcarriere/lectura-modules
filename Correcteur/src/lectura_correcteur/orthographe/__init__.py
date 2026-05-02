"""Sous-module orthographe : verification et correction orthographique."""

from lectura_correcteur.orthographe._suggestions import suggerer
from lectura_correcteur.orthographe._verificateur import VerificateurOrthographe

__all__ = ["VerificateurOrthographe", "suggerer"]
