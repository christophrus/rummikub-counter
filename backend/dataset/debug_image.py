"""
Debug-Tool: Analysiert ein Bild und zeigt detaillierte Ergebnisse.

Nutzung:
    python debug_image.py pfad/zum/bild.jpg

Erzeugt:
  - debug_detections.jpg  → Bild mit markierten Steinpositionen + Labels
  - debug_tiles/          → Einzelne ausgeschnittene Steine mit YOLO-Ergebnis
  - Konsolen-Output mit allen Detektions-Details
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Backend-Module importierbar machen
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.yolo_detector import load_yolo_model, detect_and_classify


def main():
    if len(sys.argv) < 2:
        print("Nutzung: python debug_image.py <bild.jpg>")
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Fehler: Konnte {image_path} nicht laden.")
        return

    print("=" * 60)
    print("  Rummikub Debug-Analyse (YOLO)")
    print("=" * 60)
    print(f"Bild: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # YOLO laden
    print("\nLade YOLO-Modell...")
    load_yolo_model()

    # Steine erkennen + klassifizieren
    detections = detect_and_classify(image)
    print(f"\n{len(detections)} Steine erkannt.")

    # Debug-Ordner für einzelne Tiles
    debug_dir = Path("debug_tiles")
    debug_dir.mkdir(exist_ok=True)

    # Jeden Stein analysieren
    print(f"\n{'#':>3} | {'Klasse':>8} | {'Conf':>6} | {'Joker':>5} | {'Position'}")
    print("-" * 55)

    total_score = 0

    for i, det in enumerate(detections):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        number = det["number"]
        confidence = det["confidence"]
        is_joker = det["is_joker"]

        label = "Joker" if is_joker else str(number) if number else "?"
        print(f"{i+1:>3} | {label:>8} | {confidence:>5.1%} | {'Ja' if is_joker else 'Nein':>5} | ({x},{y}) {w}x{h}")

        if is_joker:
            total_score += 20
        elif number:
            total_score += number

        # Tile-Bild ausschneiden und speichern
        tile_image = image[max(0,y):y+h, max(0,x):x+w]
        if tile_image.size > 0:
            tile_filename = f"{i+1:02d}_{label}_conf{confidence:.0%}.jpg"
            cv2.imwrite(str(debug_dir / tile_filename), tile_image)

    print(f"\nGesamtpunktzahl: {total_score}")
    print(f"\nEinzelne Tiles gespeichert in: {debug_dir}/")

    # Debug-Bild mit Markierungen erzeugen
    debug_image = image.copy()
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        label = "Joker" if det["is_joker"] else str(det["number"]) if det["number"] else "?"
        conf = det["confidence"]

        color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 3)
        text = f"{label} {conf:.0%}"
        cv2.putText(debug_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    debug_path = "debug_detections.jpg"
    cv2.imwrite(debug_path, debug_image)
    print(f"Markiertes Bild gespeichert: {debug_path}")


if __name__ == "__main__":
    main()
