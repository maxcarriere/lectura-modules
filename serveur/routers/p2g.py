"""Router P2G — POST /p2g/analyser"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Engine charge une seule fois (singleton)
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from lectura_graphemiseur import creer_engine
        _engine = creer_engine(mode="local")
        logger.info("P2G engine charge (mode local)")
    return _engine


class AnalyserRequest(BaseModel):
    ipa_words: list[str]


@router.post("/analyser")
async def analyser(req: AnalyserRequest):
    """Analyse P2G d'une liste de mots IPA.

    Returns
    -------
    dict
        {"ipa_words": [...], "ortho": [...], "pos": [...], "morpho": {...}}
    """
    engine = _get_engine()
    result = engine.analyser(req.ipa_words)
    return result
