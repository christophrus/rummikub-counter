"""
Bildvorverarbeitung für die Steinerkennung.
"""

import cv2
import numpy as np
from PIL import Image
import io


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Lädt ein Bild aus Bytes in ein OpenCV-Array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Bild konnte nicht geladen werden.")
    return image


def resize_image(image: np.ndarray, max_dimension: int = 1920) -> np.ndarray:
    """Größe anpassen ohne Kontrastveränderung. Für Steinerkennung."""
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Vorverarbeitung: Größe anpassen, Kontrast verbessern.
    Für OCR-Erkennung auf einzelnen Steinen.
    """
    image = resize_image(image)

    # Kontrast mit CLAHE verbessern (Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Bild schärfen für bessere OCR-Ergebnisse."""
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, kernel)


def extract_tile_region(image: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 5) -> np.ndarray:
    """Extrahiert einen Steinbereich aus dem Bild mit optionalem Padding."""
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    return image[y1:y2, x1:x2]


def prepare_tile_for_ocr(tile_image: np.ndarray) -> np.ndarray:
    """
    Bereitet ein einzelnes Stein-Bild für OCR vor.
    - Graustufen
    - Binarisierung
    - Rauschunterdrückung
    """
    # In Graustufen umwandeln
    if len(tile_image.shape) == 3:
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = tile_image.copy()

    # Größe normalisieren
    target_height = 80
    scale = target_height / gray.shape[0]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Rauschen entfernen
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive Binarisierung
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return binary


def encode_image_to_bytes(image: np.ndarray, format: str = ".png") -> bytes:
    """Kodiert ein OpenCV-Bild in Bytes."""
    success, buffer = cv2.imencode(format, image)
    if not success:
        raise ValueError("Bild konnte nicht kodiert werden.")
    return buffer.tobytes()
