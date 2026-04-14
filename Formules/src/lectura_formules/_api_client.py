"""Client API pour les Formules — delegue la lecture au serveur Lectura.

Utilise uniquement la stdlib (urllib), zero dependance externe.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.lec-tu-ra.com"
_TIMEOUT = 30


def _get_api_config() -> tuple[str, str]:
    """Retourne (api_url, api_key) depuis l'environnement."""
    url = os.environ.get("LECTURA_API_URL", "") or _DEFAULT_API_URL
    key = os.environ.get("LECTURA_API_KEY", "")
    return url, key


def _post(endpoint: str, payload: dict) -> Any:
    """Envoie une requete POST JSON au serveur Lectura."""
    url_base, api_key = _get_api_config()
    url = f"{url_base.rstrip('/')}{endpoint}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Erreur API Formules {exc.code} sur {endpoint} : {msg}"
        ) from None
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Impossible de contacter le serveur Lectura : {exc.reason}"
        ) from None


def lire_formule_api(formule_type: str, text: str, **kwargs) -> dict:
    """Appelle POST /formules/lire sur le serveur."""
    payload = {"formule_type": formule_type, "text": text, **kwargs}
    return _post("/formules/lire", payload)


def lire_nombre_api(text: str, **kwargs) -> dict:
    """Appelle POST /formules/lire_nombre sur le serveur."""
    payload = {"text": text, **kwargs}
    return _post("/formules/lire_nombre", payload)
