"""
API-Router für die Bildanalyse.
"""

import time
import logging
import base64

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import cv2

from app.models.schemas import AnalysisResult, DetectedTile
from app.services.tile_detector import detect_tiles, draw_detections
from app.services.cnn_classifier import classify_tile
from app.services.yolo_detector import detect_and_classify
from app.utils.image_processing import (
    load_image_from_bytes,
    resize_image,
    preprocess_image,
    extract_tile_region,
    encode_image_to_bytes,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    """
    Analysiert ein hochgeladenes Bild von Rummikub-Steinen.

    Pipeline (YOLO-Modus):
    1. Bild laden
    2. YOLO erkennt und klassifiziert alle Steine in einem Schritt

    Pipeline (CNN-Fallback):
    1. Bild laden und vorverarbeiten
    2. Steine im Bild lokalisieren (OpenCV)
    3. Für jeden Stein: Zahl erkennen (CNN)
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

    detection_mode = getattr(request.app.state, "detection_mode", "cnn")

    if detection_mode == "yolo":
        detected_tiles, total_score, img_h, img_w = _analyze_yolo(image)
    else:
        detected_tiles, total_score, img_h, img_w = _analyze_cnn(image)

    processing_time = (time.time() - start_time) * 1000

    return AnalysisResult(
        tiles=detected_tiles,
        total_score=total_score,
        tile_count=len(detected_tiles),
        processing_time_ms=round(processing_time, 2),
        image_width=img_w,
        image_height=img_h,
    )


def _analyze_yolo(image):
    """YOLO-Pipeline: Erkennung + Klassifikation in einem Schritt."""
    resized = resize_image(image)
    img_h, img_w = resized.shape[:2]

    detections = detect_and_classify(resized)
    logger.info(f"YOLO: {len(detections)} Steine erkannt.")

    detected_tiles = []
    total_score = 0

    for det in detections:
        detected_tiles.append(DetectedTile(
            number=det["number"],
            confidence=det["confidence"],
            is_joker=det["is_joker"],
            x=det["x"], y=det["y"],
            width=det["w"], height=det["h"],
        ))
        if det["is_joker"]:
            total_score += 20
        elif det["number"] is not None:
            total_score += det["number"]

    return detected_tiles, total_score, img_h, img_w


def _analyze_cnn(image):
    """CNN-Pipeline: OpenCV-Erkennung + CNN-Klassifikation."""
    resized = resize_image(image)
    enhanced = preprocess_image(image)
    img_h, img_w = resized.shape[:2]

    tile_regions = detect_tiles(resized)
    logger.info(f"OpenCV+CNN: {len(tile_regions)} Steine erkannt.")

    detected_tiles = []
    total_score = 0

    for tile_info in tile_regions:
        x, y, w, h = tile_info["x"], tile_info["y"], tile_info["w"], tile_info["h"]
        tile_image = extract_tile_region(enhanced, x, y, w, h)

        if tile_image.size == 0:
            continue

        result = classify_tile(tile_image)
        number = result["number"]
        confidence = result["confidence"]
        is_joker_tile = result["is_joker"]

        detected_tiles.append(DetectedTile(
            number=number,
            confidence=confidence,
            is_joker=is_joker_tile,
            x=x, y=y, width=w, height=h,
        ))

        if is_joker_tile:
            total_score += 20
        elif number is not None:
            total_score += number

    return detected_tiles, total_score, img_h, img_w


@router.post("/analyze-debug")
async def analyze_image_debug(request: Request, file: UploadFile = File(...)):
    """
    Debug-Endpoint: Gibt das Bild mit eingezeichneten Erkennungen zurück.
    Nützlich zum Feintuning der Erkennung.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Nur Bilddateien sind erlaubt.")

    image_bytes = await file.read()
    image = load_image_from_bytes(image_bytes)
    resized = resize_image(image)

    detection_mode = getattr(request.app.state, "detection_mode", "cnn")

    if detection_mode == "yolo":
        detections = detect_and_classify(resized)
        tile_regions = [{"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"]} for d in detections]
        results = [{"number": d["number"]} for d in detections]
    else:
        enhanced = preprocess_image(image)
        tile_regions = detect_tiles(resized)
        results = []
        for tile_info in tile_regions:
            x, y, w, h = tile_info["x"], tile_info["y"], tile_info["w"], tile_info["h"]
            tile_image = extract_tile_region(enhanced, x, y, w, h)
            if tile_image.size == 0:
                results.append({"number": None})
                continue
            cnn_result = classify_tile(tile_image)
            results.append({"number": cnn_result["number"]})

    # Debug-Bild erzeugen
    debug_image = draw_detections(resized, tile_regions, results)
    debug_bytes = encode_image_to_bytes(debug_image)
    debug_base64 = base64.b64encode(debug_bytes).decode("utf-8")

    return {
        "debug_image": f"data:image/png;base64,{debug_base64}",
        "tile_count": len(tile_regions),
        "results": results,
    }
