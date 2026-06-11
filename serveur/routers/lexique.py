"""Router Lexique -- Endpoints REST pour le Lexique Lectura v6."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Lexique charge une seule fois (singleton)
_lexique = None


def _get_lexique():
    global _lexique
    if _lexique is None:
        from lectura_lexique import Lexique

        db_path = os.environ.get(
            "LECTURA_LEXIQUE_V6_DB",
            "/opt/lectura/lexique_lectura_v6.db",
        )
        _lexique = Lexique(db_path)
        logger.info("Lexique v6 charge (db: %s)", db_path)
    return _lexique


# ── Modeles Pydantic ──────────────────────────────────────────────────


class FormeResult(BaseModel):
    ortho: str
    phone: str | None = None
    lemme: str | None = None
    cgram: str | None = None
    genre: str | None = None
    multext: str | None = None
    freq_composite: float | None = None
    nb_syllabes: int | None = None
    syllabes: str | None = None


class LemmeDetail(BaseModel):
    id: int
    lemme: str
    cgram: str
    genre: str | None = None
    freq_composite: float | None = None
    definitions: list[dict] = []
    formes: list[dict] = []
    synonymes: list[dict] = []
    antonymes: list[dict] = []
    derives: list[dict] = []
    hyperonymes: list[dict] = []
    apparentes: list[dict] = []


class EntiteDetail(BaseModel):
    id: int
    label: str
    description: str | None = None
    qid: str | None = None
    source: str | None = None
    type_entite: str | None = None
    _proprietes: dict = {}
    _categories: list[dict] = []
    _composants: list[dict] = []
    _exemples: list[str] = []

    class Config:
        # Allow fields starting with underscore
        populate_by_name = True


class CategorieNode(BaseModel):
    id: int
    label: str
    type: str | None = None
    qid: str | None = None
    description: str | None = None


class RelationsResult(BaseModel):
    lemme: str
    cgram: str | None = None
    synonymes: list[dict] = []
    antonymes: list[dict] = []
    derives: list[dict] = []
    hyperonymes: list[dict] = []
    apparentes: list[dict] = []


# ── Endpoints ─────────────────────────────────────────────────────────


_ALLOWED_MODES = {"exact", "prefix", "contains", "suffix", "phonetique"}


@router.get("/rechercher")
async def rechercher(
    q: str = Query(..., min_length=1, max_length=200, description="Terme de recherche"),
    mode: str = Query("exact", description="Mode: exact|prefix|contains|suffix|phonetique"),
    cgram: str | None = Query(None, max_length=20, description="Filtre categorie grammaticale"),
    limit: int = Query(100, ge=1, le=500, description="Nombre max de resultats"),
) -> list[dict]:
    """Recherche de formes dans le lexique."""
    if mode not in _ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Mode invalide. Modes autorises: {', '.join(sorted(_ALLOWED_MODES))}")

    lex = _get_lexique()

    # Construire le pattern selon le mode (jamais de regex libre)
    import re as _re
    q_escaped = _re.escape(q)

    if mode == "exact":
        pattern = f"^{q_escaped}$"
    elif mode == "prefix":
        pattern = f"^{q_escaped}"
    elif mode == "suffix":
        pattern = f"{q_escaped}$"
    elif mode == "contains":
        pattern = q_escaped
    elif mode == "phonetique":
        results = lex.rechercher(q_escaped, champ="phone", limite=limit)
        if cgram:
            results = [r for r in results if r.get("cgram") == cgram]
        return results
    else:
        pattern = f"^{q_escaped}$"

    results = lex.rechercher(pattern, champ="ortho", limite=limit)
    if cgram:
        results = [r for r in results if r.get("cgram") == cgram]
    return results


@router.get("/lemme/{lemme}")
async def detail_lemme(
    lemme: str,
    cgram: str | None = Query(None, description="Filtre categorie grammaticale"),
) -> dict:
    """Detail complet d'un lemme avec relations semantiques."""
    lex = _get_lexique()
    info = lex.info_lemme(lemme, cgram)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Lemme '{lemme}' introuvable")

    lemme_id = info["id"]
    result = dict(info)
    result["definitions"] = lex.definitions(lemme, cgram)
    result["formes"] = lex.formes_de(lemme, cgram)
    result["synonymes"] = lex.synonymes_de(lemme_id)
    result["antonymes"] = lex.antonymes_de(lemme_id)
    result["derives"] = lex.derives_de(lemme_id)
    result["hyperonymes"] = lex.hyperonymes_de(lemme_id)
    result["apparentes"] = lex.apparentes_sem(lemme_id)
    return result


@router.get("/formes/{lemme}")
async def formes(
    lemme: str,
    cgram: str | None = Query(None, description="Filtre categorie grammaticale"),
) -> list[dict]:
    """Formes flechies d'un lemme."""
    lex = _get_lexique()
    return lex.formes_de(lemme, cgram)


@router.get("/conjugaison/{verbe}")
async def conjugaison(verbe: str) -> dict:
    """Table de conjugaison complete d'un verbe."""
    lex = _get_lexique()
    result = lex.conjuguer(verbe)
    if not result:
        raise HTTPException(status_code=404, detail=f"Verbe '{verbe}' introuvable")
    return result


@router.get("/definitions/{lemme}")
async def definitions(
    lemme: str,
    cgram: str | None = Query(None, description="Filtre categorie grammaticale"),
) -> list[dict]:
    """Definitions et exemples d'un mot."""
    lex = _get_lexique()
    return lex.definitions(lemme, cgram)


@router.get("/entite/{id_or_qid}")
async def detail_entite(id_or_qid: str) -> dict:
    """Detail d'une entite par ID numerique ou QID Wikidata."""
    lex = _get_lexique()

    # Detecter si c'est un QID (commence par Q) ou un ID numerique
    if id_or_qid.upper().startswith("Q") and id_or_qid[1:].isdigit():
        result = lex.entite_par_qid(id_or_qid.upper())
    else:
        try:
            entite_id = int(id_or_qid)
        except ValueError:
            raise HTTPException(status_code=400, detail="ID invalide (entier ou QID attendu)")
        result = lex.entite_detail(entite_id)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Entite '{id_or_qid}' introuvable")
    return result


@router.get("/entites")
async def rechercher_entites(
    q: str = Query(..., min_length=1, max_length=200, description="Terme de recherche"),
    limit: int = Query(50, ge=1, le=500, description="Nombre max de resultats"),
) -> list[dict]:
    """Recherche d'entites par mot-cle."""
    lex = _get_lexique()
    return lex.rechercher_entites(q, limite=limit)


@router.get("/categories")
async def lister_categories() -> list[dict]:
    """Liste des categories semantiques."""
    lex = _get_lexique()
    return lex.lister_categories()


@router.get("/categories/{cat_id}/entites")
async def entites_par_categorie(
    cat_id: int,
    inclure_descendants: bool = Query(False, description="Inclure les sous-categories"),
    limit: int = Query(100, ge=1, le=500, description="Nombre max de resultats"),
) -> list[dict]:
    """Entites d'une categorie donnee."""
    lex = _get_lexique()
    # Recuperer le label de la categorie
    conn = lex._get_conn()
    row = conn.execute("SELECT label FROM categories WHERE id = ?", (cat_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Categorie {cat_id} introuvable")
    label = row[0]
    return lex.entites_par_categorie(label, inclure_descendants=inclure_descendants, limite=limit)


@router.get("/relations/{lemme}")
async def relations(
    lemme: str,
    cgram: str | None = Query(None, description="Filtre categorie grammaticale"),
) -> dict:
    """Relations semantiques d'un lemme."""
    lex = _get_lexique()
    info = lex.info_lemme(lemme, cgram)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Lemme '{lemme}' introuvable")

    lemme_id = info["id"]
    return {
        "lemme": info["lemme"],
        "cgram": info.get("cgram"),
        "synonymes": lex.synonymes_de(lemme_id),
        "antonymes": lex.antonymes_de(lemme_id),
        "derives": lex.derives_de(lemme_id),
        "hyperonymes": lex.hyperonymes_de(lemme_id),
        "apparentes": lex.apparentes_sem(lemme_id),
    }
