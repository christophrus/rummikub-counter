"""
YOLO-Detector: Erkennt und klassifiziert Rummikub-Steine in einem Schritt.

Ersetzt sowohl den OpenCV-Tile-Detector als auch den CNN-Klassifikator.
Ein einziger Forward Pass findet alle Steine und erkennt ihre Zahlen.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "rummikub_yolo.pt"

# YOLO-Klassen-Index → Rummikub-Wert
YOLO_CLASS_MAP = {i: str(i + 1) for i in range(13)}
YOLO_CLASS_MAP[13] = "joker"

_model = None


def load_yolo_model() -> None:
    """Lädt das trainierte YOLOv8-Modell."""
    global _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO-Modell nicht gefunden: {MODEL_PATH}")

    from ultralytics import YOLO
    _model = YOLO(str(MODEL_PATH))

    logger.info(f"YOLO-Modell geladen: {MODEL_PATH.name}")


def detect_and_classify(image: np.ndarray, confidence_threshold: float = 0.25) -> list[dict]:
    """
    Erkennt alle Rummikub-Steine im Bild und klassifiziert sie.

    Args:
        image: BGR-Bild (OpenCV-Format)
        confidence_threshold: Minimum Confidence für Detektionen

    Returns:
        Liste von Dicts mit:
            number: int|None (1-13 oder None bei Joker)
            confidence: float
            is_joker: bool
            x, y, w, h: Bounding-Box-Koordinaten
    """
    if _model is None:
        raise RuntimeError("YOLO-Modell nicht geladen. Zuerst load_yolo_model() aufrufen.")

    results = _model(image, conf=confidence_threshold, verbose=False)

    detected = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_idx = int(box.cls[0])

            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

            class_name = YOLO_CLASS_MAP.get(class_idx, "unknown")
            is_joker = class_name == "joker"

            try:
                number = None if is_joker else int(class_name)
            except ValueError:
                number = None

            detected.append({
                "number": number,
                "confidence": confidence,
                "is_joker": is_joker,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })

    # Nach x-Position sortieren (links → rechts)
    detected.sort(key=lambda d: d["x"])

    return detected
