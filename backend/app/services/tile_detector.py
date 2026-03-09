"""
Steinerkennung und -segmentierung mit OpenCV.

Findet einzelne Rummikub-Steine im Bild durch Konturfindung.
"""

import cv2
import numpy as np


def detect_tiles(image: np.ndarray) -> list[dict]:
    """
    Erkennt Rummikub-Steine im Bild durch Konturfindung.

    Strategie:
    1. Bild in Graustufen umwandeln
    2. Kanten erkennen (Canny Edge Detection)
    3. Konturen finden
    4. Rechteckige Konturen filtern (Steine sind rechteckig)
    5. Bounding Boxes zurückgeben

    Returns:
        Liste von Dicts mit {x, y, w, h, contour} für jeden erkannten Stein.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rauschen reduzieren
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kanten erkennen
    edges = cv2.Canny(blurred, 50, 150)

    # Kanten verbinden (Morphologie)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Konturen finden
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tiles = []
    img_area = image.shape[0] * image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # Flächen-Filter: Steine sollten eine bestimmte Mindestgröße haben
        # aber nicht zu groß sein (nicht das ganze Bild)
        min_area = img_area * 0.001  # Mindestens 0.1% des Bildes
        max_area = img_area * 0.05  # Maximal 5% des Bildes

        if area < min_area or area > max_area:
            continue

        # Bounding Rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Seitenverhältnis prüfen: Rummikub-Steine sind ~2:3 (Höhe > Breite)
        aspect_ratio = w / h if h > 0 else 0

        # Akzeptiere Seitenverhältnisse zwischen 0.3 und 1.5
        # (Steine können hochkant oder quer liegen)
        if aspect_ratio < 0.3 or aspect_ratio > 1.5:
            continue

        # Konturfüllung prüfen (Solidity) – Steine sind recht ausgefüllt
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if solidity < 0.7:
            continue

        tiles.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "contour": contour,
        })

    # Überlappende Boxen entfernen (Non-Maximum Suppression)
    tiles = _non_max_suppression(tiles, overlap_thresh=0.3)

    # Nach Position sortieren (oben-links nach unten-rechts)
    tiles.sort(key=lambda t: (t["y"] // 50, t["x"]))

    return tiles


def _non_max_suppression(tiles: list[dict], overlap_thresh: float = 0.3) -> list[dict]:
    """
    Entfernt überlappende Detektionen (Non-Maximum Suppression).
    Behält die größere Detektion bei Überlappung.
    """
    if not tiles:
        return []

    # Nach Fläche sortieren (größte zuerst)
    tiles_sorted = sorted(tiles, key=lambda t: t["area"], reverse=True)
    keep = []

    for tile in tiles_sorted:
        is_overlap = False
        for kept in keep:
            iou = _compute_iou(tile, kept)
            if iou > overlap_thresh:
                is_overlap = True
                break
        if not is_overlap:
            keep.append(tile)

    return keep


def _compute_iou(box1: dict, box2: dict) -> float:
    """Berechnet Intersection over Union (IoU) zweier Boxen."""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["w"], box2["x"] + box2["w"])
    y2 = min(box1["y"] + box1["h"], box2["y"] + box2["h"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def draw_detections(image: np.ndarray, tiles: list[dict], results: list[dict] | None = None) -> np.ndarray:
    """
    Zeichnet erkannte Steine und deren Werte ins Bild (Debug-Visualisierung).
    """
    output = image.copy()

    color_map = {
        "rot": (0, 0, 255),
        "blau": (255, 0, 0),
        "schwarz": (0, 0, 0),
        "orange": (0, 165, 255),
        None: (0, 255, 0),
    }

    for i, tile in enumerate(tiles):
        color = (0, 255, 0)  # Standard: Grün
        label = f"#{i + 1}"

        if results and i < len(results):
            result = results[i]
            tile_color = result.get("color")
            number = result.get("number")
            color = color_map.get(tile_color, (0, 255, 0))
            if number is not None:
                label = f"{number}"

        cv2.rectangle(output, (tile["x"], tile["y"]),
                      (tile["x"] + tile["w"], tile["y"] + tile["h"]),
                      color, 2)
        cv2.putText(output, label, (tile["x"], tile["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output
