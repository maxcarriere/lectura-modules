"""Router Aligneur — POST /aligneur/analyze, analyze_text, analyser_complet"""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Syllabeur charge une seule fois (singleton)
_syllabeur = None


def _get_syllabeur():
    global _syllabeur
    if _syllabeur is None:
        from lectura_aligneur.lectura_aligneur import LecturaSyllabeur
        _syllabeur = LecturaSyllabeur()
        logger.info("Aligneur LecturaSyllabeur charge (mode local)")
    return _syllabeur


# ── Modeles de requete ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    word: str
    phone: str | None = None


class AnalyzeTextRequest(BaseModel):
    text: str


class SyllabifyIpaRequest(BaseModel):
    phone: str


class MotAnalyseIn(BaseModel):
    text: str = ""
    phone: str = ""
    liaison: str = "none"
    pos: str = ""
    ponctuation_avant: bool = False
    elision_avant: bool = False
    est_formule: bool = False
    span: list[int] = [0, 0]


class EventFormuleIn(BaseModel):
    ortho: str
    phone: str
    span_source: list[int] = [0, 0]
    span_lecture: list[int] = [0, 0]


class LectureFormuleIn(BaseModel):
    display_fr: str
    events: list[EventFormuleIn] = []


class OptionsGroupesIn(BaseModel):
    gerer_elisions: bool = True
    gerer_liaisons: bool = True
    gerer_enchainement: bool = True
    ajouter_schwas_finaux: bool = False


class AnalyserCompletRequest(BaseModel):
    mots: list[MotAnalyseIn]
    lectures_formules: dict[str, LectureFormuleIn] | None = None
    options: OptionsGroupesIn | None = None


class ConstruireGroupesRequest(BaseModel):
    mots: list[MotAnalyseIn]
    options: OptionsGroupesIn | None = None


class SyllabifierGroupesRequest(BaseModel):
    groupes: list[dict]
    lectures_formules: dict[str, LectureFormuleIn] | None = None


# ── Helpers ────────────────────────────────────────────────────────────

class _MotProxy:
    """Objet duck-typed compatible avec construire_groupes_lecture et le Syllabeur."""

    def __init__(self, text, phone, liaison, pos, ponctuation_avant,
                 elision_avant, est_formule, span):
        self.text = text
        self.phone = phone
        self.liaison = liaison
        self.pos = pos
        self.ponctuation_avant = ponctuation_avant
        self.elision_avant = elision_avant
        self.est_formule = est_formule
        self.est_ponctuation = False
        self.span = span


def _mot_in_to_proxy(m: MotAnalyseIn) -> _MotProxy:
    """Convertit un MotAnalyseIn en objet duck-typed."""
    return _MotProxy(
        text=m.text,
        phone=m.phone,
        liaison=m.liaison,
        pos=m.pos,
        ponctuation_avant=m.ponctuation_avant,
        elision_avant=m.elision_avant,
        est_formule=m.est_formule,
        span=tuple(m.span),
    )


def _lecture_in_to_formule(lf: LectureFormuleIn):
    """Convertit une LectureFormuleIn en LectureFormule."""
    from lectura_aligneur._types import LectureFormule, EventFormule
    return LectureFormule(
        display_fr=lf.display_fr,
        events=[
            EventFormule(
                ortho=e.ortho,
                phone=e.phone,
                span_source=tuple(e.span_source),
                span_lecture=tuple(e.span_lecture),
            )
            for e in lf.events
        ],
    )


def _options_in_to_g2p_options(o: OptionsGroupesIn | None):
    """Convertit OptionsGroupesIn en OptionsGroupes du G2P."""
    if o is None:
        return None
    from lectura_phonemiseur.groupes_lecture import OptionsGroupes
    return OptionsGroupes(
        gerer_elisions=o.gerer_elisions,
        gerer_liaisons=o.gerer_liaisons,
        gerer_enchainement=o.gerer_enchainement,
        ajouter_schwas_finaux=o.ajouter_schwas_finaux,
    )


def _options_in_to_aligneur_options(o: OptionsGroupesIn | None):
    """Convertit OptionsGroupesIn en OptionsGroupes de l'aligneur."""
    if o is None:
        return None
    from lectura_aligneur._types import OptionsGroupes
    return OptionsGroupes(
        gerer_elisions=o.gerer_elisions,
        gerer_liaisons=o.gerer_liaisons,
        gerer_enchainement=o.gerer_enchainement,
        ajouter_schwas_finaux=o.ajouter_schwas_finaux,
    )


def _serialiser_resultat(obj) -> dict:
    """Serialise un objet dataclass ou duck-typed en dict JSON-safe."""
    try:
        return asdict(obj)
    except TypeError:
        # Fallback pour les objets non-dataclass
        result = {}
        for key in ("mots", "phone_groupe", "span", "jonctions",
                     "est_formule", "lecture", "text"):
            if hasattr(obj, key):
                val = getattr(obj, key)
                if isinstance(val, list):
                    val = [_serialiser_mot(m) if hasattr(m, "phone") else m for m in val]
                result[key] = val
        return result


def _serialiser_mot(m) -> dict:
    """Serialise un mot duck-typed en dict."""
    result = {}
    for key in ("text", "phone", "liaison", "pos", "ponctuation_avant",
                 "elision_avant", "est_formule", "est_ponctuation", "span"):
        if hasattr(m, key):
            result[key] = getattr(m, key)
    return result


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Analyse syllabique d'un mot."""
    syl = _get_syllabeur()
    result = syl.analyze(req.word, req.phone)
    return _serialiser_resultat(result)


@router.post("/analyze_text")
async def analyze_text(req: AnalyzeTextRequest):
    """Analyse syllabique de chaque mot d'un texte."""
    syl = _get_syllabeur()
    results = syl.analyze_text(req.text)
    return [_serialiser_resultat(r) for r in results]


@router.post("/syllabify_ipa")
async def syllabify_ipa(req: SyllabifyIpaRequest):
    """Decoupage syllabique sur de l'IPA brut."""
    syl = _get_syllabeur()
    return syl.syllabify_ipa(req.phone)


@router.post("/analyser_complet")
async def analyser_complet(req: AnalyserCompletRequest):
    """Analyse complete : groupes de lecture (G2P) + syllabation (Aligneur)."""
    syl = _get_syllabeur()
    mots = [_mot_in_to_proxy(m) for m in req.mots]
    lectures = None
    if req.lectures_formules:
        lectures = {
            int(k): _lecture_in_to_formule(v)
            for k, v in req.lectures_formules.items()
        }
    options = _options_in_to_aligneur_options(req.options)
    result = syl.analyser_complet(mots, lectures, options)
    return _serialiser_resultat(result)


@router.post("/construire_groupes")
async def construire_groupes_endpoint(req: ConstruireGroupesRequest):
    """E1 seul : construire les groupes de lecture (via G2P)."""
    from lectura_phonemiseur.groupes_lecture import construire_groupes_lecture

    mots = [_mot_in_to_proxy(m) for m in req.mots]
    options = _options_in_to_g2p_options(req.options)

    class _ResultProxy:
        def __init__(self, mots_list):
            self.mots = mots_list

    groupes = construire_groupes_lecture(_ResultProxy(mots), options)
    return [_serialiser_resultat(g) for g in groupes]


@router.post("/syllabifier_groupes")
async def syllabifier_groupes(req: SyllabifierGroupesRequest):
    """E2 seul : syllabifier des groupes de lecture."""
    syl = _get_syllabeur()
    # TODO : deserialiser les groupes depuis le JSON
    # Pour l'instant, retourner une erreur explicite
    return {"error": "not_implemented", "message": "syllabifier_groupes API a implementer"}
