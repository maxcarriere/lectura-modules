"""Validation légère des credentials TTS cloud.

Chaque validateur fait un appel API GET léger (lister les voix, vérifier
l'utilisateur) — zéro crédit de synthèse consommé.

Usage :
    ok, message = validate_credentials("cloud_google", {"api_key": "..."})
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


def validate_credentials(engine_key: str, params: dict) -> tuple[bool, str]:
    """Valide les credentials d'un moteur cloud via un appel API léger.

    Returns:
        (True, "") en cas de succès, (False, "message") en cas d'échec.
    """
    validators = {
        "cloud_google": _validate_google,
        "cloud_azure": _validate_azure,
        "elevenlabs": _validate_elevenlabs,
        "cloud_aws": _validate_aws,
    }
    validator = validators.get(engine_key)
    if validator is None:
        return False, f"Moteur inconnu : {engine_key}"
    try:
        return validator(params)
    except Exception as exc:
        return False, str(exc)


# ── Google Cloud TTS ─────────────────────────────────────────────────

def _validate_google(params: dict) -> tuple[bool, str]:
    api_key = params.get("api_key", "")
    credentials_file = params.get("credentials_file", "")

    if credentials_file:
        return _validate_google_service_account(credentials_file)
    if api_key:
        return _validate_google_api_key(api_key)
    return False, "Aucune clé API ni fichier service account renseigné."


def _validate_google_api_key(api_key: str) -> tuple[bool, str]:
    url = f"https://texttospeech.googleapis.com/v1/voices?key={api_key}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if "voices" in data:
                return True, ""
            return False, "Réponse inattendue de l'API Google."
    except urllib.error.HTTPError as exc:
        return _format_http_error("Google Cloud TTS", exc)
    except urllib.error.URLError as exc:
        return False, f"Erreur réseau : {exc.reason}"


def _validate_google_service_account(credentials_file: str) -> tuple[bool, str]:
    if not os.path.isfile(credentials_file):
        return False, f"Fichier introuvable : {credentials_file}"

    try:
        with open(credentials_file, encoding="utf-8") as f:
            creds = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"Fichier JSON invalide : {exc}"

    if "client_email" not in creds or "private_key" not in creds:
        return False, "Le fichier JSON ne contient pas client_email / private_key."

    # Tenter l'échange OAuth2 puis lister les voix
    try:
        from lectura_tts.engines.cloud_google import _get_access_token
        token = _get_access_token(creds)
    except Exception as exc:
        return False, f"Échec authentification OAuth2 : {exc}"

    url = "https://texttospeech.googleapis.com/v1/voices"
    req = urllib.request.Request(
        url, method="GET",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if "voices" in data:
                return True, ""
            return False, "Réponse inattendue de l'API Google."
    except urllib.error.HTTPError as exc:
        return _format_http_error("Google Cloud TTS", exc)
    except urllib.error.URLError as exc:
        return False, f"Erreur réseau : {exc.reason}"


# ── Azure TTS ────────────────────────────────────────────────────────

def _validate_azure(params: dict) -> tuple[bool, str]:
    subscription_key = params.get("subscription_key", "")
    region = params.get("region") or "westeurope"

    if not subscription_key:
        return False, "Clé d'abonnement non renseignée."

    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
    req = urllib.request.Request(
        url, method="GET",
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if isinstance(data, list) and len(data) > 0:
                return True, ""
            return False, "Réponse inattendue de l'API Azure."
    except urllib.error.HTTPError as exc:
        return _format_http_error("Azure TTS", exc)
    except urllib.error.URLError as exc:
        return False, f"Erreur réseau : {exc.reason}"


# ── Elevenlabs ───────────────────────────────────────────────────────

def _validate_elevenlabs(params: dict) -> tuple[bool, str]:
    api_key = params.get("api_key", "")

    if not api_key:
        return False, "Clé API non renseignée."

    url = "https://api.elevenlabs.io/v1/voices"
    req = urllib.request.Request(
        url, method="GET",
        headers={"xi-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if "voices" in data:
                return True, ""
            return False, "Réponse inattendue de l'API Elevenlabs."
    except urllib.error.HTTPError as exc:
        return _format_http_error("Elevenlabs", exc)
    except urllib.error.URLError as exc:
        return False, f"Erreur réseau : {exc.reason}"


# ── AWS Polly ────────────────────────────────────────────────────────

def _validate_aws(params: dict) -> tuple[bool, str]:
    access_key = params.get("access_key", "")
    secret_key = params.get("secret_key", "")
    region = params.get("region") or "eu-west-3"

    if not access_key or not secret_key:
        return False, "Access Key et Secret Key requis."

    from lectura_tts.engines.cloud_aws import _build_sigv4_headers

    host = f"polly.{region}.amazonaws.com"
    path = "/v1/voices"
    url = f"https://{host}{path}"

    signed_headers = _build_sigv4_headers(
        method="GET",
        host=host,
        path=path,
        headers={"Content-Type": "application/json"},
        payload=b"",
        access_key=access_key,
        secret_key=secret_key,
        region=region,
    )

    req = urllib.request.Request(url, method="GET", headers=signed_headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if "Voices" in data:
                return True, ""
            return False, "Réponse inattendue de l'API AWS Polly."
    except urllib.error.HTTPError as exc:
        return _format_http_error("AWS Polly", exc)
    except urllib.error.URLError as exc:
        return False, f"Erreur réseau : {exc.reason}"


# ── Helpers ──────────────────────────────────────────────────────────

def _format_http_error(engine_name: str, exc: urllib.error.HTTPError) -> tuple[bool, str]:
    """Extrait un message lisible d'une erreur HTTP."""
    status = exc.code
    try:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
            # Google : {"error": {"message": "..."}}
            msg = detail.get("error", {}).get("message", "")
            if not msg:
                # Elevenlabs : {"detail": {"message": "..."}}
                d = detail.get("detail", "")
                msg = d.get("message", str(d)) if isinstance(d, dict) else str(d)
            if not msg:
                msg = body[:200]
        except (json.JSONDecodeError, AttributeError):
            msg = body[:200]
    except Exception:
        msg = str(exc)
    return False, f"{engine_name} — HTTP {status} : {msg}"
