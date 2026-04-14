"""Router Formules — POST /formules/lire, /formules/lire_nombre"""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class LireRequest(BaseModel):
    formule_type: str
    text: str
    span: list[int] = [0, 0]
    feminin: bool = False


class LireNombreRequest(BaseModel):
    text: str
    feminin: bool = False


@router.post("/lire")
async def lire(req: LireRequest):
    """Lecture d'une formule (nombre, sigle, date, etc.)."""
    from lectura_formules import lire_formule
    result = lire_formule(
        req.formule_type,
        req.text,
        span=tuple(req.span),
        feminin=req.feminin,
    )
    return asdict(result)


@router.post("/lire_nombre")
async def lire_nombre(req: LireNombreRequest):
    """Lecture d'un nombre."""
    from lectura_formules import lire_nombre as _lire_nombre
    result = _lire_nombre(req.text, feminin=req.feminin)
    return asdict(result)
