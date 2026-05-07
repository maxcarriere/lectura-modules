"""Lectura API — Serveur FastAPI pour les modules Lectura.

Point d'entree de l'application. Monte les routers et le middleware d'auth.

Usage :
    uvicorn serveur.app:app --host 0.0.0.0 --port 8000
    # ou en dev :
    uvicorn serveur.app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serveur.auth import RateLimitMiddleware
from serveur.routers import g2p, p2g, aligneur, formules, tts, tts_multi, tts_diphone, correcteur, vc

app = FastAPI(
    title="Lectura API",
    description="API pour les modules Lectura — G2P, P2G, Aligneur, Formules, TTS, TTS Diphone, Correcteur, VC",
    version="1.0.0",
)

# CORS — autoriser les appels depuis le site et les demos Pyodide
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lec-tu-ra.com", "https://lec-tu-ra.com", "http://localhost:*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Rate limiting
app.add_middleware(RateLimitMiddleware)

# Routers
app.include_router(g2p.router, prefix="/g2p", tags=["G2P"])
app.include_router(p2g.router, prefix="/p2g", tags=["P2G"])
app.include_router(aligneur.router, prefix="/aligneur", tags=["Aligneur"])
app.include_router(formules.router, prefix="/formules", tags=["Formules"])
app.include_router(tts.router, prefix="/tts", tags=["TTS"])
app.include_router(tts_multi.router, prefix="/tts-multi", tags=["TTS Multi-Speaker"])
app.include_router(tts_diphone.router, prefix="/tts-diphone", tags=["TTS Diphone"])
app.include_router(correcteur.router, prefix="/correcteur", tags=["Correcteur"])
app.include_router(vc.router, prefix="/vc", tags=["VC"])


@app.get("/health")
async def health():
    """Verification de sante du serveur."""
    return {"status": "ok", "version": "1.0.0"}
