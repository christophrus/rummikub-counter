"""
API-Router für die Bildanalyse.
"""

import time
import logging
import base64

from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2

from app.models.schemas import AnalysisResult, DetectedTile
from app.services.tile_detector import detect_tiles, draw_detections
from app.services.ocr_service import recognize_number
from app.utils.image_processing import (
    load_image_from_bytes,
    preprocess_image,
    extract_tile_region,
    encode_image_to_bytes,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analysiert ein hochgeladenes Bild von Rummikub-Steinen.

    Pipeline:
    1. Bild laden und vorverarbeiten
    2. Steine im Bild lokalisieren (OpenCV)
    3. Für jeden Stein: Zahl erkennen (EasyOCR Deep Learning)
    4. Punkte berechnen und zurückgeben
    """
    start_time = time.time()

    # Dateivalidierung
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Nur Bilddateien sind erlaubt.")

    # 1. Bild laden
    try:
        image_bytes = await file.read()
        image = load_image_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bild konnte nicht geladen werden: {e}")

    # 2. Vorverarbeitung
    processed = preprocess_image(image)
    img_h, img_w = processed.shape[:2]

    # 3. Steine erkennen
    tile_regions = detect_tiles(processed)
    logger.info(f"{len(tile_regions)} Steine erkannt.")

    # 4. Jeden Stein analysieren
    detected_tiles = []
    total_score = 0

    for tile_info in tile_regions:
        x, y, w, h = tile_info["x"], tile_info["y"], tile_info["w"], tile_info["h"]

        # Steinbereich ausschneiden
        tile_image = extract_tile_region(processed, x, y, w, h)

        if tile_image.size == 0:
            continue

        # Zahlenerkennung (Deep Learning mit EasyOCR)
        ocr_result = recognize_number(tile_image)

        number = ocr_result.get("number")
        confidence = ocr_result.get("confidence", 0.0)

        detected_tiles.append(DetectedTile(
            number=number,
            confidence=confidence,
            is_joker=False,
            x=x, y=y, width=w, height=h,
        ))

        if number is not None:
            total_score += number

    processing_time = (time.time() - start_time) * 1000

    return AnalysisResult(
        tiles=detected_tiles,
        total_score=total_score,
        tile_count=len(detected_tiles),
        processing_time_ms=round(processing_time, 2),
        image_width=img_w,
        image_height=img_h,
    )


@router.post("/analyze-debug")
async def analyze_image_debug(file: UploadFile = File(...)):
    """
    Debug-Endpoint: Gibt das Bild mit eingezeichneten Erkennungen zurück.
    Nützlich zum Feintuning der Erkennung.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Nur Bilddateien sind erlaubt.")

    image_bytes = await file.read()
    image = load_image_from_bytes(image_bytes)
    processed = preprocess_image(image)

    tile_regions = detect_tiles(processed)

    # Steine analysieren
    results = []
    for tile_info in tile_regions:
        x, y, w, h = tile_info["x"], tile_info["y"], tile_info["w"], tile_info["h"]
        tile_image = extract_tile_region(processed, x, y, w, h)
        if tile_image.size == 0:
            results.append({"number": None})
            continue
        ocr_result = recognize_number(tile_image)
        results.append({
            "number": ocr_result.get("number"),
        })

    # Debug-Bild erzeugen
    debug_image = draw_detections(processed, tile_regions, results)
    debug_bytes = encode_image_to_bytes(debug_image)
    debug_base64 = base64.b64encode(debug_bytes).decode("utf-8")

    return {
        "debug_image": f"data:image/png;base64,{debug_base64}",
        "tile_count": len(tile_regions),
        "results": results,
    }
