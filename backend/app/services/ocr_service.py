"""
OCR-Service mit EasyOCR für die Zahlenerkennung auf Rummikub-Steinen.

EasyOCR nutzt Deep Learning (CNN + LSTM) unter der Haube:
- Feature Extraction: ResNet / VGG (CNN)
- Sequence Modeling: BiLSTM
- Prediction: CTC (Connectionist Temporal Classification)
"""

import easyocr
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# EasyOCR Reader als Singleton (Modell-Laden dauert einige Sekunden)
_reader: easyocr.Reader | None = None


def get_reader() -> easyocr.Reader:
    """
    Gibt den EasyOCR Reader zurück (Singleton-Pattern).
    Das Laden des Modells geschieht nur beim ersten Aufruf.
    """
    global _reader
    if _reader is None:
        logger.info("Lade EasyOCR-Modell (Deep Learning: CNN + LSTM)...")
        _reader = easyocr.Reader(
            ["en"],  # Englisch reicht für Zahlen
            gpu=False,  # Auf True setzen wenn GPU verfügbar
        )
        logger.info("EasyOCR-Modell geladen.")
    return _reader


def recognize_number(tile_image: np.ndarray) -> dict:
    """
    Erkennt die Zahl auf einem einzelnen Rummikub-Stein.

    Deep Learning Pipeline:
    1. Bildvorverarbeitung (Graustufen, Binarisierung)
    2. CNN extrahiert visuelle Features
    3. LSTM verarbeitet die Feature-Sequenz
    4. CTC-Decoder gibt den erkannten Text aus

    Args:
        tile_image: Ausgeschnittenes Bild eines einzelnen Steins (BGR)

    Returns:
        Dict mit {number, confidence} oder {number: None} bei Fehler
    """
    reader = get_reader()

    # Mehrere Vorverarbeitungs-Varianten probieren für bessere Erkennung
    candidates = []

    # Variante 1: Originalbild
    candidates.append(tile_image)

    # Variante 2: Graustufen mit Kontrastverstärkung
    gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    candidates.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

    # Variante 3: Binarisiert (Schwarz auf Weiß)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

    # Variante 4: Invertiert binarisiert
    binary_inv = cv2.bitwise_not(binary)
    candidates.append(cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR))

    best_result = {"number": None, "confidence": 0.0}

    for candidate in candidates:
        # Größe skalieren für bessere OCR-Ergebnisse
        h, w = candidate.shape[:2]
        if h < 50:
            scale = 50 / h
            candidate = cv2.resize(candidate, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

        try:
            results = reader.readtext(
                candidate,
                allowlist="0123456789",  # Nur Zahlen erkennen
                paragraph=False,
                min_size=5,
                text_threshold=0.3,
                low_text=0.3,
            )

            for (bbox, text, confidence) in results:
                text = text.strip()
                if not text:
                    continue

                try:
                    number = int(text)
                    # Rummikub hat Zahlen von 1-13
                    if 1 <= number <= 13 and confidence > best_result["confidence"]:
                        best_result = {
                            "number": number,
                            "confidence": float(confidence),
                        }
                except ValueError:
                    continue

        except Exception as e:
            logger.warning(f"OCR-Fehler: {e}")
            continue

    return best_result


def is_joker(tile_image: np.ndarray) -> bool:
    """
    Erkennt ob ein Stein ein Joker ist.

    Joker haben ein buntes Symbol statt einer Zahl.
    Erkennung über hohe Farbvielfalt (Standardabweichung im H-Kanal).
    """
    hsv = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    # Nur gesättigte Pixel betrachten (nicht den cremefarbigen Hintergrund)
    saturated_mask = s_channel > 50
    if cv2.countNonZero(saturated_mask.astype(np.uint8)) < 20:
        return False

    h_values = hsv[:, :, 0][saturated_mask]
    h_std = np.std(h_values)

    # Joker haben typischerweise eine hohe Farbvielfalt
    return h_std > 40


def recognize_batch(tile_images: list[np.ndarray]) -> list[dict]:
    """
    Erkennt Zahlen auf mehreren Steinen.

    Args:
        tile_images: Liste von Stein-Bildern

    Returns:
        Liste von Ergebnissen {number, confidence}
    """
    results = []
    for tile_image in tile_images:
        result = recognize_number(tile_image)
        results.append(result)
    return results
