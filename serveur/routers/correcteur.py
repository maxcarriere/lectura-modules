"""Router Correcteur — POST /correcteur/corriger"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Correcteur charge une seule fois (singleton)
_correcteur = None


def _get_correcteur():
    global _correcteur
    if _correcteur is None:
        from lectura_lexique import Lexique
        from lectura_correcteur import Correcteur

        # Sur le VPS, le lexique est installe dans /opt/lectura/
        import os
        db_path = os.environ.get(
            "LECTURA_LEXIQUE_DB",
            "/opt/lectura/lexique_lectura.db",
        )
        lex = Lexique(db_path)
        _correcteur = Correcteur(lex)
        # Warmup
        _correcteur.corriger("test")
        logger.info("Correcteur charge (lexique: %s)", db_path)
    return _correcteur


class CorrigerRequest(BaseModel):
    phrase: str


class CorrectionItem(BaseModel):
    index: int
    original: str
    corrige: str
    type_correction: str
    regle: str = ""
    explication: str = ""


class CorrigerResponse(BaseModel):
    phrase_originale: str
    phrase_corrigee: str
    corrections: list[CorrectionItem]


@router.post("/corriger")
async def corriger(req: CorrigerRequest) -> CorrigerResponse:
    """Corrige une phrase (orthographe + grammaire).

    Returns
    -------
    CorrigerResponse
        phrase_originale, phrase_corrigee, corrections[]
    """
    correcteur = _get_correcteur()
    result = correcteur.corriger(req.phrase)
    return CorrigerResponse(
        phrase_originale=result.phrase_originale,
        phrase_corrigee=result.phrase_corrigee,
        corrections=[
            CorrectionItem(
                index=c.index,
                original=c.original,
                corrige=c.corrige,
                type_correction=c.type_correction.value,
                regle=c.regle,
                explication=c.explication,
            )
            for c in result.corrections
        ],
    )
