"""
Rummikub Stein-Erkennung – FastAPI Backend

Hauptanwendung mit CORS-Konfiguration und Router-Einbindung.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze
from app.services.cnn_classifier import load_model as load_cnn
from app.services.yolo_detector import load_yolo_model, MODEL_PATH as YOLO_MODEL_PATH

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
    Lädt YOLO-Modell (bevorzugt) oder CNN-Modell als Fallback.
    """
    logger.info("🚀 Starte Rummikub-Erkennung Backend...")

    if YOLO_MODEL_PATH.exists():
        logger.info("📦 Lade YOLOv8-Modell (Detection + Klassifikation)...")
        load_yolo_model()
        app.state.detection_mode = "yolo"
        logger.info("✅ YOLO-Modell geladen. Backend bereit!")
    else:
        logger.info("📦 Lade CNN-Modell für Stein-Erkennung...")
        load_cnn()
        app.state.detection_mode = "cnn"
        logger.info("✅ CNN-Modell geladen. Backend bereit (OpenCV + CNN Modus).")

    yield
    logger.info("👋 Backend wird beendet.")


app = FastAPI(
    title="Rummikub Stein-Erkennung",
    description=(
        "API zur Erkennung von Rummikub-Steinen in Bildern. "
        "Nutzt YOLOv8 oder CNN+OpenCV für Erkennung und Klassifikation."
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
        "https://christophrus.github.io",  # GitHub Pages PWA
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
