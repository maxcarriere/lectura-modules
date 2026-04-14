"""Router G2P — POST /g2p/analyser"""

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
        from lectura_nlp import creer_engine
        _engine = creer_engine(mode="local")
        logger.info("G2P engine charge (mode local)")
    return _engine


class AnalyserRequest(BaseModel):
    tokens: list[str]


@router.post("/analyser")
async def analyser(req: AnalyserRequest):
    """Analyse G2P d'une liste de tokens.

    Returns
    -------
    dict
        {"tokens": [...], "g2p": [...], "pos": [...], "liaison": [...], "morpho": {...}}
    """
    engine = _get_engine()
    result = engine.analyser(req.tokens)
    return result
