"""Auth et rate limiting pour l'API Lectura.

Middleware FastAPI :
  - Sans cle API : 100 requetes/jour (demo)
  - Cle gratuite : 1000 requetes/jour (dev)
  - Cle payante : illimite

Les cles sont stockees dans un fichier JSON (api_keys.json).
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

_KEYS_FILE = Path(__file__).parent / "api_keys.json"

# Quotas par tier (requetes/jour)
_QUOTAS = {
    "demo": 100,
    "free": 1000,
    "paid": 100_000,
    "unlimited": float("inf"),
}

# Cache en memoire des compteurs (reset quotidien)
_counters: dict[str, dict] = defaultdict(lambda: {"count": 0, "day": 0})


def _load_keys() -> dict[str, dict]:
    """Charge les cles API depuis le fichier JSON."""
    if not _KEYS_FILE.exists():
        return {}
    try:
        return json.loads(_KEYS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Impossible de charger %s", _KEYS_FILE)
        return {}


def _get_tier(api_key: str | None) -> str:
    """Retourne le tier associe a une cle API."""
    if not api_key:
        return "demo"
    keys = _load_keys()
    info = keys.get(api_key)
    if info is None:
        return "demo"  # Cle inconnue → quota demo
    return info.get("tier", "free")


def _check_quota(identifier: str, tier: str) -> bool:
    """Verifie si le quota est respecte. Retourne True si OK."""
    quota = _QUOTAS.get(tier, 100)
    today = int(time.time() // 86400)
    entry = _counters[identifier]
    if entry["day"] != today:
        entry["count"] = 0
        entry["day"] = today
    if entry["count"] >= quota:
        return False
    entry["count"] += 1
    return True


# Chemins web exempts du rate limiting API (gere par nginx)
_WEB_EXEMPT_PREFIXES = (
    "/", "/mot/", "/entite/", "/categorie/", "/rimes/",
    "/rechercher", "/sitemap", "/robots.txt", "/static/",
)


def _is_web_path(path: str) -> bool:
    """Teste si un chemin est une page web (exempte du rate limiting API)."""
    # Les routes API commencent par un prefixe de module (/lexique/, /g2p/, etc.)
    if path in ("/", "/health", "/robots.txt"):
        return True
    if path.startswith(("/mot/", "/entite/", "/categorie/", "/rimes/",
                        "/rechercher", "/sitemap", "/static/")):
        return True
    return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting base sur la cle API ou l'IP."""

    async def dispatch(self, request: Request, call_next):
        # Pas de rate limit sur /health et les pages web
        # (le rate limiting web est gere par nginx)
        if _is_web_path(request.url.path):
            return await call_next(request)

        # Extraire la cle API du header Authorization
        auth_header = request.headers.get("Authorization", "")
        api_key = None
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:].strip()

        tier = _get_tier(api_key)
        identifier = api_key or (request.client.host if request.client else "unknown")

        if not _check_quota(identifier, tier):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Quota {tier} depasse. "
                    "Obtenez une cle API sur https://www.lec-tu-ra.com/api",
                },
            )

        response = await call_next(request)
        return response
