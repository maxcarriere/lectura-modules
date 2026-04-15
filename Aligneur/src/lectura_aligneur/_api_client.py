"""Client API pour l'Aligneur-Syllabeur.

Meme interface que LecturaSyllabeur local, mais delegue l'execution
au serveur Lectura via HTTP. Utilise uniquement la stdlib (urllib).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error

from lectura_aligneur._types import (
    Phoneme,
    GroupePhonologique,
    Syllabe,
    ResultatAnalyse,
    MotAnalyse,
    EventFormule,
    LectureFormule,
    OptionsGroupes,
    GroupeLecture,
    ResultatGroupe,
    ResultatSyllabation,
    Span,
)

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.lec-tu-ra.com"
_TIMEOUT = 30


class LecturaApiError(Exception):
    """Erreur lors d'un appel a l'API Lectura."""


class LecturaSyllabeur:
    """Client API — meme interface que l'Aligneur local.

    Quand les donnees locales ne sont pas disponibles, toutes les
    methodes d'analyse delegent au serveur Lectura via HTTP.

    Parameters
    ----------
    api_url : str | None
        URL de base du serveur (defaut : LECTURA_API_URL ou https://api.lec-tu-ra.com)
    api_key : str | None
        Cle API (defaut : LECTURA_API_KEY ou vide pour le mode demo)
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        *,
        phonemizer: object | None = None,
    ) -> None:
        self._url = (
            api_url
            or os.environ.get("LECTURA_API_URL", "")
            or _DEFAULT_API_URL
        )
        self._key = api_key or os.environ.get("LECTURA_API_KEY", "")
        # phonemizer est accepte pour compatibilite de signature mais ignore
        # (le serveur a son propre phonemiseur)

    # ── API retrocompatible ──────────────────────────────────────────────

    def analyze(self, word: str, phone: str | None = None) -> ResultatAnalyse:
        """Analyse syllabique complete d'un mot."""
        data = self._post("/aligneur/analyze", {"word": word, "phone": phone})
        return _deserialiser_resultat_analyse(data)

    def analyze_text(self, text: str) -> list[ResultatAnalyse]:
        """Analyse syllabique de chaque mot d'un texte."""
        data = self._post("/aligneur/analyze_text", {"text": text})
        return [_deserialiser_resultat_analyse(r) for r in data]

    def syllabify_ipa(self, phone: str) -> list[str]:
        """Decoupage syllabique sur de l'IPA brut."""
        data = self._post("/aligneur/syllabify_ipa", {"phone": phone})
        return data

    # ── API complete avec groupes de lecture ──────────────────────────────

    def analyser_complet(
        self,
        mots: list[MotAnalyse],
        lectures_formules: dict[int, LectureFormule] | None = None,
        options: OptionsGroupes | None = None,
    ) -> ResultatSyllabation:
        """Analyse complete E1 + E2 : groupes de lecture puis syllabation."""
        payload: dict = {
            "mots": [_serialiser_mot_analyse(m) for m in mots],
        }
        if lectures_formules:
            payload["lectures_formules"] = {
                str(k): _serialiser_lecture_formule(v)
                for k, v in lectures_formules.items()
            }
        if options:
            payload["options"] = {
                "gerer_elisions": options.gerer_elisions,
                "gerer_liaisons": options.gerer_liaisons,
                "gerer_enchainement": options.gerer_enchainement,
                "ajouter_schwas_finaux": options.ajouter_schwas_finaux,
            }
        data = self._post("/aligneur/analyser_complet", payload)
        return _deserialiser_resultat_syllabation(data)

    def construire_groupes(
        self,
        mots: list[MotAnalyse],
        options: OptionsGroupes | None = None,
    ) -> list[GroupeLecture]:
        """E1 seul : construit les groupes de lecture."""
        payload: dict = {
            "mots": [_serialiser_mot_analyse(m) for m in mots],
        }
        if options:
            payload["options"] = {
                "gerer_elisions": options.gerer_elisions,
                "gerer_liaisons": options.gerer_liaisons,
                "gerer_enchainement": options.gerer_enchainement,
                "ajouter_schwas_finaux": options.ajouter_schwas_finaux,
            }
        data = self._post("/aligneur/construire_groupes", payload)
        return [_deserialiser_groupe_lecture(g) for g in data]

    def syllabifier_groupes(
        self,
        groupes: list[GroupeLecture],
        lectures_formules: dict[int, LectureFormule] | None = None,
    ) -> list[ResultatGroupe]:
        """E2 seul : syllabifie des groupes de lecture."""
        payload: dict = {
            "groupes": [_serialiser_groupe_lecture(g) for g in groupes],
        }
        if lectures_formules:
            payload["lectures_formules"] = {
                str(k): _serialiser_lecture_formule(v)
                for k, v in lectures_formules.items()
            }
        data = self._post("/aligneur/syllabifier_groupes", payload)
        return [_deserialiser_resultat_groupe(rg) for rg in data]

    # ── Transport HTTP ───────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: dict) -> object:
        """Envoie une requete POST JSON et retourne la reponse decodee."""
        url = f"{self._url.rstrip('/')}{endpoint}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="replace")
            raise LecturaApiError(
                f"Erreur API {exc.code} sur {endpoint} : {msg}"
            ) from None
        except urllib.error.URLError as exc:
            raise LecturaApiError(
                f"Impossible de contacter le serveur Lectura ({self._url}) : {exc.reason}"
            ) from None


# ══════════════════════════════════════════════════════════════════════════════
# Serialisation → JSON
# ══════════════════════════════════════════════════════════════════════════════


def _serialiser_mot_analyse(m: MotAnalyse) -> dict:
    return {
        "text": m.text,
        "phone": m.phone,
        "liaison": m.liaison,
        "pos": m.pos,
        "ponctuation_avant": m.ponctuation_avant,
        "elision_avant": m.elision_avant,
        "est_formule": m.est_formule,
        "span": list(m.span),
    }


def _serialiser_event_formule(e: EventFormule) -> dict:
    return {
        "ortho": e.ortho,
        "phone": e.phone,
        "span_source": list(e.span_source),
        "span_lecture": list(e.span_lecture),
    }


def _serialiser_lecture_formule(lf: LectureFormule) -> dict:
    return {
        "display_fr": lf.display_fr,
        "events": [_serialiser_event_formule(e) for e in lf.events],
    }


def _serialiser_groupe_lecture(g: GroupeLecture) -> dict:
    return {
        "mots": [_serialiser_mot_analyse(m) for m in g.mots],
        "phone_groupe": g.phone_groupe,
        "span": list(g.span),
        "jonctions": g.jonctions,
        "est_formule": g.est_formule,
        "lecture": _serialiser_lecture_formule(g.lecture) if g.lecture else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Deserialisation JSON → dataclasses
# ══════════════════════════════════════════════════════════════════════════════


def _span(raw: list | tuple | None) -> Span:
    if raw is None:
        return (0, 0)
    return (int(raw[0]), int(raw[1]))


def _deserialiser_phoneme(d: dict) -> Phoneme:
    return Phoneme(ipa=d.get("ipa", ""), grapheme=d.get("grapheme", ""))


def _deserialiser_groupe_phonologique(d: dict | None) -> GroupePhonologique:
    if d is None:
        return GroupePhonologique()
    phonemes = [_deserialiser_phoneme(p) for p in d.get("phonemes", [])]
    return GroupePhonologique(phonemes=phonemes)


def _deserialiser_syllabe(d: dict) -> Syllabe:
    return Syllabe(
        phone=d.get("phone", ""),
        ortho=d.get("ortho", ""),
        span=_span(d.get("span")),
        attaque=_deserialiser_groupe_phonologique(d.get("attaque")),
        noyau=_deserialiser_groupe_phonologique(d.get("noyau")),
        coda=_deserialiser_groupe_phonologique(d.get("coda")),
    )


def _deserialiser_resultat_analyse(d: dict) -> ResultatAnalyse:
    return ResultatAnalyse(
        mot=d.get("mot", ""),
        phone=d.get("phone", ""),
        syllabes=[_deserialiser_syllabe(s) for s in d.get("syllabes", [])],
    )


def _deserialiser_event_formule(d: dict) -> EventFormule:
    return EventFormule(
        ortho=d.get("ortho", ""),
        phone=d.get("phone", ""),
        span_source=_span(d.get("span_source")),
        span_lecture=_span(d.get("span_lecture")),
    )


def _deserialiser_lecture_formule(d: dict | None) -> LectureFormule | None:
    if d is None:
        return None
    return LectureFormule(
        display_fr=d.get("display_fr", ""),
        events=[_deserialiser_event_formule(e) for e in d.get("events", [])],
    )


def _deserialiser_mot_analyse(d: dict) -> MotAnalyse:
    return MotAnalyse(
        token=None,
        phone=d.get("phone", ""),
        liaison=d.get("liaison", "none"),
        pos=d.get("pos", ""),
        ponctuation_avant=d.get("ponctuation_avant", False),
        elision_avant=d.get("elision_avant", False),
        est_formule=d.get("est_formule", False),
    )


def _deserialiser_groupe_lecture(d: dict) -> GroupeLecture:
    return GroupeLecture(
        mots=[_deserialiser_mot_analyse(m) for m in d.get("mots", [])],
        phone_groupe=d.get("phone_groupe", ""),
        span=_span(d.get("span")),
        jonctions=d.get("jonctions", []),
        est_formule=d.get("est_formule", False),
        lecture=_deserialiser_lecture_formule(d.get("lecture")),
    )


def _deserialiser_resultat_groupe(d: dict) -> ResultatGroupe:
    return ResultatGroupe(
        groupe=_deserialiser_groupe_lecture(d.get("groupe", {})),
        syllabes=[_deserialiser_syllabe(s) for s in d.get("syllabes", [])],
    )


def _deserialiser_resultat_syllabation(d: dict) -> ResultatSyllabation:
    options_raw = d.get("options", {})
    return ResultatSyllabation(
        texte_original=d.get("texte_original", ""),
        groupes=[_deserialiser_resultat_groupe(rg) for rg in d.get("groupes", [])],
        options=OptionsGroupes(
            gerer_elisions=options_raw.get("gerer_elisions", True),
            gerer_liaisons=options_raw.get("gerer_liaisons", True),
            gerer_enchainement=options_raw.get("gerer_enchainement", True),
            ajouter_schwas_finaux=options_raw.get("ajouter_schwas_finaux", False),
        ),
    )
