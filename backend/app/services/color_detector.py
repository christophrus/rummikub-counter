"""
Farberkennung für Rummikub-Steine.

Rummikub hat 4 Farben:
- Schwarz
- Rot
- Blau
- Orange

Die Farberkennung nutzt den HSV-Farbraum, der besser für
Farbunterscheidung geeignet ist als RGB.
"""

import cv2
import numpy as np


# HSV-Bereiche für Rummikub-Farben
# H: 0-180 (OpenCV), S: 0-255, V: 0-255
COLOR_RANGES = {
    "rot": [
        # Rot liegt an beiden Enden des H-Spektrums
        {"lower": np.array([0, 80, 80]), "upper": np.array([10, 255, 255])},
        {"lower": np.array([160, 80, 80]), "upper": np.array([180, 255, 255])},
    ],
    "blau": [
        {"lower": np.array([90, 60, 60]), "upper": np.array([130, 255, 255])},
    ],
    "orange": [
        {"lower": np.array([10, 80, 80]), "upper": np.array([25, 255, 255])},
    ],
    "schwarz": [
        # Schwarz = niedrige Sättigung und niedriger Wert
        {"lower": np.array([0, 0, 0]), "upper": np.array([180, 80, 80])},
    ],
}


def detect_color(tile_image: np.ndarray) -> dict:
    """
    Erkennt die Farbe der Zahl auf einem Rummikub-Stein.

    Strategie:
    1. Stein-Hintergrund entfernen (cremefarbig/beige)
    2. Farbige Pixel im HSV-Raum analysieren
    3. Farbe mit den meisten Treffern zurückgeben

    Args:
        tile_image: Ausgeschnittenes Bild eines einzelnen Steins (BGR)

    Returns:
        Dict mit {color, confidence}
    """
    hsv = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)

    # Stein-Hintergrund maskieren (cremefarbig)
    # Der Hintergrund hat hohe Helligkeit und niedrige Sättigung
    bg_lower = np.array([0, 0, 160])
    bg_upper = np.array([180, 60, 255])
    bg_mask = cv2.inRange(hsv, bg_lower, bg_upper)

    # Invertierte Maske = nur die farbigen Bereiche (Zahlen)
    fg_mask = cv2.bitwise_not(bg_mask)

    # Optional: Kleine Artefakte entfernen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    total_fg_pixels = cv2.countNonZero(fg_mask)

    if total_fg_pixels == 0:
        return {"color": "schwarz", "confidence": 0.5}

    # Für jede Farbe: Anzahl passender Pixel zählen
    color_scores = {}

    for color_name, ranges in COLOR_RANGES.items():
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for r in ranges:
            range_mask = cv2.inRange(hsv, r["lower"], r["upper"])
            color_mask = cv2.bitwise_or(color_mask, range_mask)

        # Nur Vordergrund-Pixel berücksichtigen
        combined = cv2.bitwise_and(color_mask, fg_mask)
        pixel_count = cv2.countNonZero(combined)
        color_scores[color_name] = pixel_count

    # Farbe mit den meisten Treffern
    if not color_scores or max(color_scores.values()) == 0:
        return {"color": "schwarz", "confidence": 0.5}

    best_color = max(color_scores, key=color_scores.get)
    best_count = color_scores[best_color]
    confidence = best_count / total_fg_pixels if total_fg_pixels > 0 else 0

    return {
        "color": best_color,
        "confidence": min(float(confidence), 1.0),
    }


def is_joker(tile_image: np.ndarray) -> bool:
    """
    Erkennt ob ein Stein ein Joker ist.

    Joker-Heuristiken:
    - Joker haben ein buntes Gesicht/Symbol statt einer Zahl
    - Deutlich mehr Farbvielfalt als normale Steine
    - Kein klar erkennbarer Zahlenwert
    """
    hsv = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)

    # Farbvielfalt messen (Standardabweichung des H-Kanals)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]

    # Nur gesättigte Pixel betrachten
    saturated_mask = s_channel > 50
    if cv2.countNonZero(saturated_mask.astype(np.uint8)) < 20:
        return False

    h_values = h_channel[saturated_mask]
    h_std = np.std(h_values)

    # Joker haben typischerweise eine hohe Farbvielfalt
    return h_std > 40
