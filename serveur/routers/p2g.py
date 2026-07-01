"""Router P2G — POST /p2g/analyser"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Engine charge une seule fois (singleton)
_engine = None
_aligner = None


def _get_engine():
    global _engine
    if _engine is None:
        from lectura_graphemiseur import creer_engine
        _engine = creer_engine(mode="local")
        logger.info("P2G engine charge (mode local)")
    return _engine


def _get_aligner():
    global _aligner
    if _aligner is None:
        from lectura_aligneur.lectura_aligneur import LecturaSyllabeur
        _aligner = LecturaSyllabeur()
        logger.info("P2G aligner (LecturaSyllabeur) charge")
    return _aligner


class AnalyserRequest(BaseModel):
    ipa_words: list[str]


@router.post("/analyser")
async def analyser(req: AnalyserRequest):
    """Analyse P2G d'une liste de mots IPA.

    Returns
    -------
    dict
        {"ipa_words": [...], "ortho": [...], "pos": [...], "morpho": {...},
         "alignments": [...]}
    """
    engine = _get_engine()
    aligner = _get_aligner()

    import lectura_p2g
    return lectura_p2g.analyser(req.ipa_words, engine=engine, aligner=aligner)
