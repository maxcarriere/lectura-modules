"""Backend API pour le G2P — delegue l'inference au serveur Lectura.

Meme interface que OnnxInferenceEngine / NumpyInferenceEngine / PureInferenceEngine.
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


class ApiInferenceEngine:
    """Backend API — meme interface que les engines locaux.

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

    def analyser(self, tokens: list[str]) -> dict[str, Any]:
        """Analyse une liste de tokens via l'API serveur.

        Returns
        -------
        dict
            {
                "tokens": [...],
                "g2p": [ipa_per_token],
                "pos": [pos_per_token],
                "liaison": [liaison_per_token],
                "morpho": {feat: [val_per_token]},
            }
        """
        if not tokens:
            return {"tokens": [], "g2p": [], "pos": [], "liaison": [], "morpho": {}}

        payload = json.dumps({"tokens": tokens}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        url = f"{self._url.rstrip('/')}/g2p/analyser"
        req = urllib.request.Request(url, data=payload, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Erreur API G2P {exc.code} : {msg}"
            ) from None
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Impossible de contacter le serveur Lectura ({self._url}) : {exc.reason}"
            ) from None
