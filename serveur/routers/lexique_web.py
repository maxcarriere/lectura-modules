"""Router Lexique Web — Pages HTML publiques pour lexique.lec-tu-ra.com.

Reutilise le singleton _get_lexique() du router API (pas de 2e connexion BDD).
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from urllib.parse import quote as urlquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.templating import Jinja2Templates

from serveur.routers.lexique import _get_lexique

logger = logging.getLogger(__name__)

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ── Validation ───────────────────────────────────────────────────────

_MAX_LEMME = 100
_MAX_SEARCH = 200
_QID_RE = re.compile(r"^Q\d{1,10}$")
_SLUG_RE = re.compile(r"^[\w\-àâäéèêëïîôùûüÿçœæ ]{1,200}$", re.UNICODE)
_PER_PAGE = 50


def _validate_lemme(lemme: str) -> str:
    if len(lemme) > _MAX_LEMME:
        raise HTTPException(status_code=400, detail="Lemme trop long")
    return lemme


def _validate_qid(qid: str) -> str:
    if not _QID_RE.match(qid.upper()):
        raise HTTPException(status_code=400, detail="QID invalide")
    return qid.upper()


def _slugify(text: str) -> str:
    """Cree un slug URL-safe a partir d'un texte."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s\-àâäéèêëïîôùûüÿçœæ]", "", slug, flags=re.UNICODE)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    return slug


def _unslugify(slug: str) -> str:
    """Reconvertit un slug en label approximatif."""
    return slug.replace("-", " ")


def _trier_entites_par_popularite(entites: list[dict], limite: int) -> list[dict]:
    """Enrichit les entites avec notoriete/pageviews et trie par popularite."""
    if not entites:
        return entites
    lex = _get_lexique()
    ids = [e["id"] for e in entites]
    placeholders = ",".join("?" * len(ids))
    conn = lex._get_conn()
    try:
        rows = conn.execute(
            f"SELECT entite_id, cle, valeur FROM entite_proprietes "
            f"WHERE cle IN ('notoriete','pageviews_fr') "
            f"AND entite_id IN ({placeholders})",
            ids,
        ).fetchall()
        scores: dict[int, dict] = {}
        for row in rows:
            scores.setdefault(row[0], {})
            try:
                scores[row[0]][row[1]] = float(row[2])
            except (ValueError, TypeError):
                pass
        for e in entites:
            s = scores.get(e["id"], {})
            e["notoriete"] = s.get("notoriete", 0)
            e["pageviews_fr"] = s.get("pageviews_fr", 0)
    except Exception:
        pass
    entites.sort(
        key=lambda e: (e.get("notoriete", 0), e.get("pageviews_fr", 0)),
        reverse=True,
    )
    return entites[:limite]


# ── Stats (cached) ───────────────────────────────────────────────────

_stats_cache: dict | None = None


def _get_stats() -> dict:
    global _stats_cache
    if _stats_cache is not None:
        return _stats_cache
    lex = _get_lexique()
    conn = lex._get_conn()
    try:
        nb_lemmes = conn.execute("SELECT COUNT(*) FROM lemmes").fetchone()[0]
    except Exception:
        nb_lemmes = 0
    try:
        nb_entites = conn.execute("SELECT COUNT(*) FROM entites").fetchone()[0]
    except Exception:
        try:
            nb_entites = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        except Exception:
            nb_entites = 0
    try:
        nb_categories = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    except Exception:
        nb_categories = 0

    def _count(sql: str) -> int:
        try:
            return conn.execute(sql).fetchone()[0]
        except Exception:
            return 0

    _stats_cache = {
        "nb_lemmes": nb_lemmes,
        "nb_entites": nb_entites,
        "nb_categories": nb_categories,
        "nb_formes": _count("SELECT COUNT(*) FROM formes"),
        "nb_definitions": _count("SELECT COUNT(*) FROM sens"),
        "nb_synonymes": _count("SELECT COUNT(*) FROM lemme_synonymes"),
        "nb_antonymes": _count("SELECT COUNT(*) FROM lemme_antonymes"),
        "nb_derives": _count("SELECT COUNT(*) FROM lemme_derives"),
        "nb_hyperonymes": _count("SELECT COUNT(*) FROM lemme_hyperonymes"),
        "nb_apparentes": _count("SELECT COUNT(*) FROM lemme_apparentes_sem"),
    }
    return _stats_cache


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
async def accueil(request: Request):
    """Page d'accueil avec barre de recherche et stats."""
    stats = _get_stats()
    return templates.TemplateResponse(request, "lexique/accueil.html", stats)


@router.get("/rechercher", response_class=HTMLResponse)
async def rechercher(
    request: Request,
    q: str = Query("", max_length=_MAX_SEARCH),
    mode: str = Query("rechercher"),
    nb_phonemes: int = Query(2, ge=1, le=10),
    sous_ensemble: bool = Query(False),
    longueur_min: int = Query(2, ge=1, le=20),
    pe: int = Query(1, ge=1, le=10000),
    tri: str = Query("notoriete"),
    ordre: str = Query("desc"),
):
    """Resultats de recherche (multi-mode)."""
    q = q.strip()
    tri = tri if tri in ("label", "notoriete", "pageviews") else "notoriete"
    ordre = ordre if ordre in ("asc", "desc") else "desc"

    ctx: dict = {
        "q": q,
        "mode": mode,
        "nb_phonemes": nb_phonemes,
        "sous_ensemble": sous_ensemble,
        "longueur_min": longueur_min,
        "entites_exact": [],
        "entites_contient": [],
        "resultats": [],
        "total_exact": 0,
        "total_contient": 0,
        "page_contient": pe,
        "total_pages_contient": 1,
        "tri_entites": tri,
        "ordre_entites": ordre,
    }

    if not q:
        return templates.TemplateResponse(request, "lexique/recherche.html", ctx)

    lex = _get_lexique()

    if mode == "rimes":
        ctx["resultats"] = lex.rimes(q, nb_phonemes=nb_phonemes, limite=200)
    elif mode == "anagrammes":
        resultats = lex.anagrammes(q, limite=200)
        if sous_ensemble:
            resultats_courts = lex.anagrammes(q, limite=500)
            resultats = [
                r for r in resultats_courts
                if len(r.get("ortho", "")) >= longueur_min
            ]
        else:
            resultats = [
                r for r in resultats
                if len(r.get("ortho", "")) >= longueur_min
            ]
        ctx["resultats"] = resultats[:200]
    else:
        # Mode rechercher classique
        conn = lex._get_conn()
        exact = []
        try:
            cur = conn.execute(
                "SELECT * FROM entites WHERE label = ? COLLATE NOCASE", (q,),
            )
            exact = [dict(row) for row in cur.fetchall()]
        except Exception:
            pass

        fuzzy = lex.rechercher_entites(q, limite=5000)
        # Fusionner sans doublons
        seen_ids = {e["id"] for e in exact}
        merged = list(exact)
        for e in fuzzy:
            if e["id"] not in seen_ids:
                merged.append(e)
                seen_ids.add(e["id"])

        all_entites = _trier_entites_par_popularite(merged, limite=5000)

        # Separer match exact vs contient
        q_lower = q.lower()
        exact_ids = {e["id"] for e in exact}
        # Aussi inclure les entites dont le label match exactement
        for e in all_entites:
            if (e.get("label") or "").lower() == q_lower:
                exact_ids.add(e["id"])

        entites_exact = [e for e in all_entites if e["id"] in exact_ids]
        entites_contient = [e for e in all_entites if e["id"] not in exact_ids]

        # Fonction de tri
        def _sort_key(tri_col, rev):
            if tri_col == "label":
                return lambda e: (e.get("label") or "").lower()
            elif tri_col == "notoriete":
                return lambda e: e.get("notoriete", 0)
            else:
                return lambda e: e.get("pageviews_fr", 0)

        rev = ordre == "desc"
        entites_exact.sort(key=_sort_key(tri, rev), reverse=rev)
        entites_contient.sort(key=_sort_key(tri, rev), reverse=rev)

        # Exact : tous affiches (pas de pagination)
        ctx["entites_exact"] = entites_exact
        ctx["total_exact"] = len(entites_exact)

        # Contient : pagine
        total_contient = len(entites_contient)
        total_pages_contient = max(1, math.ceil(total_contient / _PER_PAGE))
        offset = (pe - 1) * _PER_PAGE
        ctx["entites_contient"] = entites_contient[offset:offset + _PER_PAGE]
        ctx["total_contient"] = total_contient
        ctx["page_contient"] = pe
        ctx["total_pages_contient"] = total_pages_contient

        pattern_exact = f"^{re.escape(q)}$"
        resultats = lex.rechercher(pattern_exact, champ="ortho", limite=100)

        if len(resultats) < 10:
            pattern_prefix = f"^{re.escape(q)}"
            resultats_prefix = lex.rechercher(pattern_prefix, champ="ortho", limite=100)
            seen = {r["ortho"].lower() for r in resultats}
            for r in resultats_prefix:
                if r["ortho"].lower() not in seen:
                    resultats.append(r)
                    seen.add(r["ortho"].lower())
        ctx["resultats"] = resultats[:100]

    return templates.TemplateResponse(request, "lexique/recherche.html", ctx)


@router.get("/mot/{lemme}/conjugaison", response_class=HTMLResponse)
async def page_conjugaison(request: Request, lemme: str):
    """Table de conjugaison d'un verbe."""
    lemme = _validate_lemme(lemme)
    lex = _get_lexique()
    conjugaison = lex.conjuguer(lemme)
    if not conjugaison:
        raise HTTPException(status_code=404, detail=f"Verbe '{lemme}' introuvable")
    return templates.TemplateResponse(request, "lexique/conjugaison.html", {
        "verbe": lemme,
        "conjugaison": conjugaison,
    })


def _detect_role(label: str, lemme: str, etype: str) -> str:
    """Detecte le role d'une entite par rapport a un lemme."""
    if etype == "homonyme":
        return "Homonyme"
    ll = lemme.lower()
    lab = label.lower()
    if " " in lab and lab.endswith(ll):
        return "Nom de famille"
    if " " in lab and lab.startswith(ll):
        return "Prénom"
    return "Lié"


def _entites_liees_paginee(
    lemme: str,
    page: int = 1,
    tri: str = "label",
    ordre: str = "asc",
    filtre: str = "tous",
) -> tuple[list[dict], int]:
    """Retourne (entites_page, total) pour les entites liees a un lemme."""
    lex = _get_lexique()
    conn = lex._get_conn()

    if lex._schema_version < 5:
        all_ent = lex.entites_associees(lemme, limite=5000)
        all_ent = _trier_entites_par_popularite(all_ent, limite=5000)
        return all_ent[:_PER_PAGE], len(all_ent)

    # Requete enrichie avec tri serveur
    sort_map = {
        "label": "e.label",
        "notoriete": "CAST(COALESCE(ep_n.valeur, '0') AS REAL)",
        "pageviews": "CAST(COALESCE(ep_p.valeur, '0') AS REAL)",
    }
    sort_col = sort_map.get(tri, "e.label")
    sort_dir = "DESC" if ordre == "desc" else "ASC"

    base_sql = (
        "FROM entites e "
        "JOIN entite_lemmes el ON el.entite_id = e.id "
        "LEFT JOIN entite_proprietes ep_n ON e.id = ep_n.entite_id AND ep_n.cle = 'notoriete' "
        "LEFT JOIN entite_proprietes ep_p ON e.id = ep_p.entite_id AND ep_p.cle = 'pageviews_fr' "
        "WHERE el.lemme_id IN (SELECT id FROM lemmes WHERE lemme = ? COLLATE NOCASE) "
    )
    params: list = [lemme]

    # Filtre par role
    if filtre == "prenom":
        base_sql += "AND el.type != 'homonyme' AND lower(e.label) LIKE lower(?) || ' %' "
        params.append(lemme)
    elif filtre == "nom":
        base_sql += "AND el.type != 'homonyme' AND lower(e.label) LIKE '% ' || lower(?) "
        params.append(lemme)
    elif filtre == "homonyme":
        base_sql += "AND el.type = 'homonyme' "
    elif filtre == "composant":
        base_sql += (
            "AND el.type != 'homonyme' "
            "AND NOT (lower(e.label) LIKE lower(?) || ' %') "
            "AND NOT (lower(e.label) LIKE '% ' || lower(?)) "
        )
        params.extend([lemme, lemme])

    # Count
    try:
        total = conn.execute(
            "SELECT COUNT(DISTINCT e.id) " + base_sql, params,
        ).fetchone()[0]
    except Exception:
        total = 0

    # Fetch page
    offset = (page - 1) * _PER_PAGE
    try:
        rows = conn.execute(
            "SELECT DISTINCT e.*, el.type AS _type, el.pertinence AS _pertinence, "
            "CAST(COALESCE(ep_n.valeur, '0') AS REAL) AS notoriete, "
            "CAST(COALESCE(ep_p.valeur, '0') AS REAL) AS pageviews_fr "
            + base_sql
            + f"ORDER BY {sort_col} {sort_dir} "
            "LIMIT ? OFFSET ?",
            params + [_PER_PAGE, offset],
        ).fetchall()
        entites = [dict(row) for row in rows]
    except Exception:
        entites = []

    return entites, total


@router.get("/mot/{lemme}", response_class=HTMLResponse)
async def page_mot(
    request: Request,
    lemme: str,
    pe: int = Query(1, ge=1, le=10000),
    tri: str = Query("label"),
    ordre: str = Query("asc"),
    filtre: str = Query("tous"),
):
    """Fiche lemme (tous les homographes)."""
    lemme = _validate_lemme(lemme)
    lex = _get_lexique()

    # Recuperer tous les homographes
    infos = lex.infos_lemmes(lemme)
    if not infos:
        raise HTTPException(status_code=404, detail=f"Mot '{lemme}' introuvable")

    entries = []
    first_def = ""
    for info in infos:
        lemme_id = info["id"]
        cgram = info.get("cgram", "")

        defs = lex.definitions(lemme, cgram)
        if not first_def and defs:
            d = defs[0]
            first_def = d.get("definition") or d.get("gloss") or d.get("label", "")

        # Recuperer le phone depuis les formes
        phone = None
        formes_list = lex.formes_de(lemme, cgram)
        for f in formes_list:
            if f.get("ortho", "").lower() == lemme.lower() and f.get("phone"):
                phone = f["phone"]
                break

        entries.append({
            **info,
            "phone": phone,
            "definitions": defs,
            "formes": formes_list,
            "synonymes": lex.synonymes_de(lemme_id),
            "antonymes": lex.antonymes_de(lemme_id),
            "derives": lex.derives_de(lemme_id),
            "hyperonymes": lex.hyperonymes_de(lemme_id),
            "apparentes": lex.apparentes_sem(lemme_id),
            "proverbes": lex.proverbes_de(lemme_id),
            "is_verb": cgram in ("VER", "AUX"),
        })

    # Entites associees paginées, triées, filtrées
    tri = tri if tri in ("label", "notoriete", "pageviews") else "label"
    ordre = ordre if ordre in ("asc", "desc") else "asc"
    filtre = filtre if filtre in ("tous", "prenom", "nom", "homonyme", "composant") else "tous"

    entites_liees, total_entites = _entites_liees_paginee(
        lemme, page=pe, tri=tri, ordre=ordre, filtre=filtre,
    )
    # Ajouter le role detecte pour chaque entite
    for e in entites_liees:
        e["_role"] = _detect_role(e.get("label", ""), lemme, e.get("_type", ""))

    total_pages_entites = max(1, math.ceil(total_entites / _PER_PAGE))

    # Query params pour la pagination
    qp_parts = []
    if tri != "label":
        qp_parts.append(f"tri={tri}")
    if ordre != "asc":
        qp_parts.append(f"ordre={ordre}")
    if filtre != "tous":
        qp_parts.append(f"filtre={filtre}")
    entites_query_params = "&" + "&".join(qp_parts) if qp_parts else ""

    return templates.TemplateResponse(request, "lexique/mot.html", {
        "lemme": lemme,
        "entries": entries,
        "entites_liees": entites_liees,
        "first_def": first_def,
        "total_entites": total_entites,
        "page_entites": pe,
        "total_pages_entites": total_pages_entites,
        "entites_base_url": f"/mot/{lemme}",
        "entites_query_params": entites_query_params,
        "tri_entites": tri,
        "ordre_entites": ordre,
        "filtre_entites": filtre,
    })


@router.get("/entite/{qid}/{slug}", response_class=HTMLResponse)
async def page_entite(request: Request, qid: str, slug: str = ""):
    """Fiche entite."""
    qid = _validate_qid(qid)
    lex = _get_lexique()

    entite = lex.entite_par_qid(qid)
    if entite is None:
        raise HTTPException(status_code=404, detail=f"Entité {qid} introuvable")

    # Enrichir les categories avec des labels si ce sont des dicts
    categories = entite.get("_categories", [])
    if categories and isinstance(categories[0], str):
        categories = [{"label": c} for c in categories]
        entite["_categories"] = categories

    # Resoudre l'URL image Wikimedia Commons
    image_url = ""
    image_name = (entite.get("_proprietes") or {}).get("image", "") or entite.get("image", "")
    if image_name:
        encoded = urlquote(image_name.replace(" ", "_"), safe="")
        image_url = (
            f"https://commons.wikimedia.org/w/index.php"
            f"?title=Special:Redirect/file/{encoded}&width=300"
        )

    return templates.TemplateResponse(request, "lexique/entite.html", {
        "entite": entite,
        "image_url": image_url,
    })


@router.get("/categorie/{slug}", response_class=HTMLResponse)
async def page_categorie(
    request: Request,
    slug: str,
    page: int = Query(1, ge=1, le=10000),
    sous_categories: bool = Query(False),
    tri: str = Query("label"),
    ordre: str = Query("asc"),
):
    """Liste d'entites dans une categorie, paginee."""
    label = _unslugify(slug)
    lex = _get_lexique()

    cat_info = lex.info_categorie(label)
    if cat_info is None:
        raise HTTPException(status_code=404, detail=f"Catégorie '{label}' introuvable")

    # Compter le total pour la pagination
    conn = lex._get_conn()
    try:
        if lex._schema_version >= 5:
            if sous_categories:
                total = conn.execute(
                    "SELECT COUNT(DISTINCT ec.entite_id) FROM entite_categories ec "
                    "JOIN categorie_hierarchie h ON ec.categorie_id = h.descendant_id "
                    "JOIN categories cat ON h.ancestor_id = cat.id "
                    "WHERE cat.label = ? COLLATE NOCASE",
                    (label,),
                ).fetchone()[0]
            else:
                total = conn.execute(
                    "SELECT COUNT(DISTINCT ec.entite_id) FROM entite_categories ec "
                    "JOIN categories cat ON ec.categorie_id = cat.id "
                    "WHERE cat.label = ? COLLATE NOCASE",
                    (label,),
                ).fetchone()[0]
        else:
            total = conn.execute(
                "SELECT COUNT(DISTINCT cc.concept_id) FROM concept_categories cc "
                "JOIN categories cat ON cc.categorie_id = cat.id "
                "WHERE cat.label = ? COLLATE NOCASE",
                (label,),
            ).fetchone()[0]
    except Exception:
        total = 0

    total_pages = max(1, math.ceil(total / _PER_PAGE))
    if page > total_pages:
        page = total_pages

    # Valider tri/ordre
    tri = tri if tri in ("label", "notoriete", "pageviews") else "label"
    ordre = ordre if ordre in ("asc", "desc") else "asc"

    # Requête triée avec enrichissement notoriete/pageviews
    sort_map = {
        "label": "e.label",
        "notoriete": "CAST(COALESCE(ep_n.valeur, '0') AS REAL)",
        "pageviews": "CAST(COALESCE(ep_p.valeur, '0') AS REAL)",
    }
    sort_col = sort_map[tri]
    sort_dir = "ASC" if ordre == "asc" else "DESC"

    offset = (page - 1) * _PER_PAGE

    try:
        if sous_categories and lex._schema_version >= 5:
            base_join = (
                "FROM entites e "
                "JOIN entite_categories ec ON e.id = ec.entite_id "
                "JOIN categorie_hierarchie h ON ec.categorie_id = h.descendant_id "
                "JOIN categories cat ON h.ancestor_id = cat.id "
            )
        elif lex._schema_version >= 5:
            base_join = (
                "FROM entites e "
                "JOIN entite_categories ec ON e.id = ec.entite_id "
                "JOIN categories cat ON ec.categorie_id = cat.id "
            )
        else:
            base_join = (
                "FROM entites e "
                "JOIN concept_categories ec ON e.id = ec.concept_id "
                "JOIN categories cat ON ec.categorie_id = cat.id "
            )

        sql = (
            "SELECT DISTINCT e.*, "
            "CAST(COALESCE(ep_n.valeur, '0') AS REAL) AS notoriete, "
            "CAST(COALESCE(ep_p.valeur, '0') AS REAL) AS pageviews_fr "
            + base_join +
            "LEFT JOIN entite_proprietes ep_n ON e.id = ep_n.entite_id AND ep_n.cle = 'notoriete' "
            "LEFT JOIN entite_proprietes ep_p ON e.id = ep_p.entite_id AND ep_p.cle = 'pageviews_fr' "
            f"WHERE cat.label = ? COLLATE NOCASE "
            f"ORDER BY {sort_col} {sort_dir} "
            "LIMIT ? OFFSET ?"
        )
        rows = conn.execute(sql, (label, _PER_PAGE, offset)).fetchall()
        entites = [dict(row) for row in rows]
    except Exception:
        entites = lex.entites_par_categorie(
            label, inclure_descendants=sous_categories,
            limite=_PER_PAGE, offset=offset,
        )
        for e in entites:
            e.setdefault("notoriete", 0)
            e.setdefault("pageviews_fr", 0)

    qp_parts = []
    if sous_categories:
        qp_parts.append("sous_categories=1")
    if tri != "label":
        qp_parts.append(f"tri={tri}")
    if ordre != "asc":
        qp_parts.append(f"ordre={ordre}")
    query_params = "&" + "&".join(qp_parts) if qp_parts else ""
    return templates.TemplateResponse(request, "lexique/categorie.html", {
        "categorie": cat_info,
        "entites": entites,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "base_url": f"/categorie/{slug}",
        "query_params": query_params,
        "sous_categories": sous_categories,
        "tri": tri,
        "ordre": ordre,
    })


@router.get("/documentation", response_class=HTMLResponse)
async def page_documentation(request: Request):
    """Page de documentation du projet Lexique Lectura."""
    stats = _get_stats()
    return templates.TemplateResponse(request, "lexique/documentation.html", {
        "stats": stats,
    })


@router.get("/categories", response_class=HTMLResponse)
async def page_categories(request: Request):
    """Explorateur hierarchique de categories."""
    return templates.TemplateResponse(request, "lexique/categories.html", {})


@router.get("/api/categories/arbre")
async def api_categories_arbre():
    """Arbre hierarchique des categories (JSON)."""
    lex = _get_lexique()
    conn = lex._get_conn()

    # Index id -> infos
    all_cats = lex.lister_categories()
    cat_info: dict[int, dict] = {}
    for cat in all_cats:
        cat_info[int(cat["id"])] = {
            "label": cat["label"],
            "type": cat.get("type", ""),
            "id": int(cat["id"]),
        }

    # Counts d'entites par categorie
    count_map: dict[int, int] = {}
    try:
        cur = conn.execute(
            "SELECT categorie_id, COUNT(*) FROM entite_categories "
            "GROUP BY categorie_id"
        )
        for row in cur.fetchall():
            count_map[int(row[0])] = int(row[1])
    except Exception:
        pass

    # Relations parent -> [child_ids] (depth=1)
    children_of: dict[int, list[int]] = {}
    child_ids: set[int] = set()
    try:
        cur = conn.execute(
            "SELECT ancestor_id, descendant_id FROM categorie_hierarchie "
            "WHERE depth = 1"
        )
        for row in cur.fetchall():
            parent_id, child_id = int(row[0]), int(row[1])
            if parent_id in cat_info and child_id in cat_info:
                child_ids.add(child_id)
                children_of.setdefault(parent_id, []).append(child_id)
    except Exception:
        pass

    # Counts recursifs via closure table
    descendants_of: dict[int, list[int]] = {}
    try:
        cur = conn.execute(
            "SELECT ancestor_id, descendant_id "
            "FROM categorie_hierarchie WHERE depth > 0"
        )
        for row in cur.fetchall():
            descendants_of.setdefault(int(row[0]), []).append(int(row[1]))
    except Exception:
        pass

    # Construction recursive (chaque noeud une seule fois)
    placed: set[int] = set()

    def _build(cid: int, ancestors: set[int]) -> dict | None:
        if cid in ancestors or cid in placed:
            return None
        info = cat_info.get(cid)
        if info is None:
            return None
        placed.add(cid)
        direct = count_map.get(cid, 0)
        rc = direct
        for desc_id in descendants_of.get(cid, []):
            rc += count_map.get(desc_id, 0)
        node = {
            "label": info["label"],
            "id": cid,
            "count": direct,
            "rc": rc,
            "children": [],
        }
        next_anc = ancestors | {cid}
        for child_id in sorted(
            children_of.get(cid, []),
            key=lambda x: cat_info.get(x, {}).get("label", ""),
        ):
            child_node = _build(child_id, next_anc)
            if child_node is not None:
                node["children"].append(child_node)
        return node

    root_ids = sorted(
        [cid for cid in cat_info if cid not in child_ids],
        key=lambda cid: cat_info[cid]["label"],
    )
    roots = []
    for cid in root_ids:
        node = _build(cid, set())
        if node is not None:
            roots.append(node)

    return roots


@router.get("/rimes", response_class=HTMLResponse)
async def page_rimes_accueil(request: Request):
    """Page d'accueil rimes (sans terminaison)."""
    return templates.TemplateResponse(request, "lexique/rimes.html", {
        "terminaison": "",
        "resultats": [],
    })


@router.get("/rimes/{terminaison:path}", response_class=HTMLResponse)
async def page_rimes(request: Request, terminaison: str):
    """Mots qui riment avec une terminaison phonetique."""
    if len(terminaison) > _MAX_LEMME:
        raise HTTPException(status_code=400, detail="Terminaison trop longue")

    lex = _get_lexique()
    resultats = lex.rimes_par_suffixe(terminaison, limite=200)

    return templates.TemplateResponse(request, "lexique/rimes.html", {
        "terminaison": terminaison,
        "resultats": resultats,
    })


@router.get("/anagrammes", response_class=HTMLResponse)
async def page_anagrammes(
    request: Request,
    mot: str = Query("", max_length=_MAX_LEMME),
):
    """Recherche d'anagrammes."""
    mot = mot.strip()
    resultats = []
    if mot:
        lex = _get_lexique()
        resultats = lex.anagrammes(mot, limite=100)
    return templates.TemplateResponse(request, "lexique/anagrammes.html", {
        "mot": mot,
        "resultats": resultats,
    })


# ── Robots.txt ───────────────────────────────────────────────────────


@router.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt():
    return (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /rechercher\n"
        "\n"
        "Sitemap: https://lexique.lec-tu-ra.com/sitemap.xml\n"
    )


# ── Sitemaps ─────────────────────────────────────────────────────────


@router.get("/sitemap.xml", response_class=Response)
async def sitemap_index():
    """Sitemap index avec sous-sitemaps."""
    lex = _get_lexique()
    conn = lex._get_conn()

    try:
        nb_lemmes = conn.execute("SELECT COUNT(*) FROM lemmes").fetchone()[0]
    except Exception:
        nb_lemmes = 0
    try:
        if lex._schema_version >= 5:
            nb_entites = conn.execute("SELECT COUNT(*) FROM entites").fetchone()[0]
        else:
            nb_entites = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    except Exception:
        nb_entites = 0

    nb_lemme_files = max(1, math.ceil(nb_lemmes / 50000))
    nb_entite_files = max(1, math.ceil(nb_entites / 50000))

    sitemaps = []
    for i in range(nb_lemme_files):
        sitemaps.append(f"  <sitemap><loc>https://lexique.lec-tu-ra.com/sitemap-lemmes-{i}.xml</loc></sitemap>")
    for i in range(nb_entite_files):
        sitemaps.append(f"  <sitemap><loc>https://lexique.lec-tu-ra.com/sitemap-entites-{i}.xml</loc></sitemap>")
    sitemaps.append("  <sitemap><loc>https://lexique.lec-tu-ra.com/sitemap-categories.xml</loc></sitemap>")

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(sitemaps) + "\n"
        '</sitemapindex>\n'
    )
    return Response(content=xml, media_type="application/xml")


@router.get("/sitemap-lemmes-{index}.xml", response_class=Response)
async def sitemap_lemmes(index: int):
    """Sous-sitemap des lemmes (50k par fichier)."""
    lex = _get_lexique()
    conn = lex._get_conn()

    offset = index * 50000
    try:
        rows = conn.execute(
            "SELECT lemme FROM lemmes ORDER BY id LIMIT 50000 OFFSET ?",
            (offset,),
        ).fetchall()
    except Exception:
        rows = []

    if not rows:
        raise HTTPException(status_code=404)

    urls = []
    for row in rows:
        lemme = row[0]
        urls.append(
            f"  <url><loc>https://lexique.lec-tu-ra.com/mot/{lemme}</loc>"
            f"<changefreq>monthly</changefreq><priority>0.6</priority></url>"
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls) + "\n"
        '</urlset>\n'
    )
    return Response(content=xml, media_type="application/xml")


@router.get("/sitemap-entites-{index}.xml", response_class=Response)
async def sitemap_entites(index: int):
    """Sous-sitemap des entites (50k par fichier)."""
    lex = _get_lexique()
    conn = lex._get_conn()

    offset = index * 50000
    table = "entites" if lex._schema_version >= 5 else "concepts"
    qid_col = "qid" if lex._schema_version >= 5 else "NULL"
    label_col = "label" if lex._schema_version >= 5 else "lemme"

    try:
        rows = conn.execute(
            f"SELECT id, {qid_col} AS qid, {label_col} AS label FROM {table} ORDER BY id LIMIT 50000 OFFSET ?",
            (offset,),
        ).fetchall()
    except Exception:
        rows = []

    if not rows:
        raise HTTPException(status_code=404)

    urls = []
    for row in rows:
        qid = row["qid"] if row["qid"] else str(row["id"])
        label = row["label"] or ""
        slug = _slugify(label)
        urls.append(
            f"  <url><loc>https://lexique.lec-tu-ra.com/entite/{qid}/{slug}</loc>"
            f"<changefreq>monthly</changefreq><priority>0.5</priority></url>"
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls) + "\n"
        '</urlset>\n'
    )
    return Response(content=xml, media_type="application/xml")


@router.get("/sitemap-categories.xml", response_class=Response)
async def sitemap_categories():
    """Sitemap des categories."""
    lex = _get_lexique()
    categories = lex.lister_categories()

    urls = []
    for cat in categories:
        label = cat.get("label", "")
        slug = _slugify(label)
        urls.append(
            f"  <url><loc>https://lexique.lec-tu-ra.com/categorie/{slug}</loc>"
            f"<changefreq>weekly</changefreq><priority>0.4</priority></url>"
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls) + "\n"
        '</urlset>\n'
    )
    return Response(content=xml, media_type="application/xml")
