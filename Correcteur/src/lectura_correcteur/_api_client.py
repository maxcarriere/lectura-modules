"""Client API pour le Correcteur.

Meme interface que Correcteur local, mais delegue l'execution
au serveur Lectura via HTTP. Utilise uniquement la stdlib (urllib).

Usage :
    from lectura_correcteur._api_client import CorrecteurAPI
    correcteur = CorrecteurAPI()
    result = correcteur.corriger("Les enfant mange.")
    print(result.phrase_corrigee)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error

from lectura_correcteur._types import (
    Correction,
    ResultatCorrection,
    TypeCorrection,
)

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.lec-tu-ra.com"
_TIMEOUT = 30


class LecturaApiError(Exception):
    """Erreur lors d'un appel a l'API Lectura."""


class CorrecteurAPI:
    """Client API — meme interface que Correcteur local.

    Quand le lexique local n'est pas disponible, delegue la correction
    au serveur Lectura via HTTP.

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
    ) -> None:
        self._url = (
            api_url
            or os.environ.get("LECTURA_API_URL", "")
            or _DEFAULT_API_URL
        )
        self._key = api_key or os.environ.get("LECTURA_API_KEY", "")

    def corriger(self, phrase: str) -> ResultatCorrection:
        """Corrige une phrase via l'API Lectura.

        Retourne un ResultatCorrection identique a celui du Correcteur local.
        """
        data = self._post("/correcteur/corriger", {"phrase": phrase})
        return _deserialiser_resultat(data)

    # ── Transport HTTP ───────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: dict) -> dict:
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
# Deserialisation JSON → dataclasses
# ══════════════════════════════════════════════════════════════════════════════

def _type_correction_from_str(s: str) -> TypeCorrection:
    """Convertit une string en TypeCorrection."""
    try:
        return TypeCorrection(s)
    except ValueError:
        return TypeCorrection.AUCUNE


def _deserialiser_resultat(data: dict) -> ResultatCorrection:
    """Convertit la reponse JSON en ResultatCorrection."""
    corrections = []
    for c in data.get("corrections", []):
        corrections.append(Correction(
            index=c.get("index", 0),
            original=c.get("original", ""),
            corrige=c.get("corrige", ""),
            type_correction=_type_correction_from_str(c.get("type_correction", "aucune")),
            regle=c.get("regle", ""),
            explication=c.get("explication", ""),
        ))
    return ResultatCorrection(
        phrase_originale=data.get("phrase_originale", ""),
        phrase_corrigee=data.get("phrase_corrigee", ""),
        corrections=corrections,
    )
