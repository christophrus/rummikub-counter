"""
Rummikub Stein-Erkennung – FastAPI Backend

Hauptanwendung mit CORS-Konfiguration und Router-Einbindung.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze
from app.services.ocr_service import get_reader

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/Shutdown Events.
    Beim Start wird das EasyOCR-Modell vorgeladen.
    """
    logger.info("🚀 Starte Rummikub-Erkennung Backend...")
    logger.info("📦 Lade EasyOCR Deep-Learning-Modell (CNN + LSTM)...")
    get_reader()  # Modell vorladen
    logger.info("✅ Modell geladen. Backend bereit!")
    yield
    logger.info("👋 Backend wird beendet.")


app = FastAPI(
    title="Rummikub Stein-Erkennung",
    description=(
        "API zur Erkennung von Rummikub-Steinen in Bildern. "
        "Nutzt EasyOCR (Deep Learning: CNN + LSTM) für die Zahlenerkennung "
        "und OpenCV für die Stein-Segmentierung."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS für React-Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite Dev Server
        "http://localhost:3000",  # Alternative
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router einbinden
app.include_router(analyze.router, prefix="/api", tags=["Analyse"])


@app.get("/")
async def root():
    return {
        "message": "Rummikub Stein-Erkennung API",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
