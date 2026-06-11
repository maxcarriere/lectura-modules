"""Lectura API — Serveur FastAPI pour les modules Lectura.

Point d'entree de l'application. Monte les routers et le middleware d'auth.

Usage :
    uvicorn serveur.app:app --host 0.0.0.0 --port 8000
    # ou en dev :
    uvicorn serveur.app:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from serveur.auth import RateLimitMiddleware
from serveur.routers import g2p, p2g, aligneur, formules, tts, tts_multi, tts_diphone, correcteur, vc, lexique, ctc, stt
from serveur.routers import lexique_web

app = FastAPI(
    title="Lectura API",
    description="API pour les modules Lectura — G2P, P2G, Aligneur, Formules, TTS, TTS Diphone, Correcteur, VC, CTC, STT, Lexique",
    version="1.0.0",
)

# Fichiers statiques (CSS, images)
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# CORS — autoriser les appels depuis le site et les demos Pyodide
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.lec-tu-ra.com",
        "https://lec-tu-ra.com",
        "https://lexique.lec-tu-ra.com",
        "http://localhost:*",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Rate limiting
app.add_middleware(RateLimitMiddleware)

# Routers API
app.include_router(g2p.router, prefix="/g2p", tags=["G2P"])
app.include_router(p2g.router, prefix="/p2g", tags=["P2G"])
app.include_router(aligneur.router, prefix="/aligneur", tags=["Aligneur"])
app.include_router(formules.router, prefix="/formules", tags=["Formules"])
app.include_router(tts.router, prefix="/tts", tags=["TTS"])
app.include_router(tts_multi.router, prefix="/tts-multi", tags=["TTS Multi-Speaker"])
app.include_router(tts_diphone.router, prefix="/tts-diphone", tags=["TTS Diphone"])
app.include_router(correcteur.router, prefix="/correcteur", tags=["Correcteur"])
app.include_router(vc.router, prefix="/vc", tags=["VC"])
app.include_router(ctc.router, prefix="/ctc", tags=["CTC"])
app.include_router(stt.router, prefix="/stt", tags=["STT"])
app.include_router(lexique.router, prefix="/lexique", tags=["Lexique"])

# Router Web (pages HTML publiques)
app.include_router(lexique_web.router, tags=["Lexique Web"])


@app.get("/health")
async def health():
    """Verification de sante du serveur."""
    return {"status": "ok", "version": "1.0.0"}


# ── 404 handler pour les pages web ──────────────────────────────────

_templates_dir = Path(__file__).resolve().parent / "templates"
_templates_404 = Jinja2Templates(directory=str(_templates_dir))


@app.exception_handler(404)
async def custom_404(request: Request, exc):
    """Page 404 HTML pour les navigateurs, JSON pour les appels API."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return _templates_404.TemplateResponse(
            request,
            "lexique/404.html",
            status_code=404,
        )
    return JSONResponse(status_code=404, content={"detail": "Not found"})
