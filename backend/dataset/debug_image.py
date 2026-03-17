"""
Debug-Tool: Analysiert ein Bild und zeigt detaillierte Ergebnisse.

Nutzung:
    python debug_image.py pfad/zum/bild.jpg

Erzeugt:
  - debug_detections.jpg  → Bild mit markierten Steinpositionen
  - debug_tiles/          → Einzelne ausgeschnittene Steine mit CNN-Ergebnis
  - Konsolen-Output mit allen Klassifikations-Details
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Backend-Module importierbar machen
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.cnn_classifier import load_model, classify_tile
from app.services.tile_detector import detect_tiles, draw_detections
from app.utils.image_processing import resize_image, preprocess_image, extract_tile_region


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
    print("  Rummikub Debug-Analyse")
    print("=" * 60)
    print(f"Bild: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # CNN laden
    print("\nLade CNN-Modell...")
    load_model()

    # Pipeline ausführen
    resized = resize_image(image)
    enhanced = preprocess_image(image)
    print(f"Resized: {resized.shape[1]}x{resized.shape[0]}")

    # Steine erkennen
    tile_regions = detect_tiles(resized)
    print(f"\n{len(tile_regions)} Steine erkannt.")

    # Debug-Ordner für einzelne Tiles
    debug_dir = Path("debug_tiles")
    debug_dir.mkdir(exist_ok=True)

    # Jeden Stein analysieren
    print(f"\n{'#':>3} | {'Klasse':>8} | {'Conf':>6} | {'Joker':>5} | {'Position'}")
    print("-" * 55)

    results = []
    total_score = 0

    for i, tile_info in enumerate(tile_regions):
        x, y, w, h = tile_info["x"], tile_info["y"], tile_info["w"], tile_info["h"]
        tile_image = extract_tile_region(enhanced, x, y, w, h)

        if tile_image.size == 0:
            print(f"{i+1:>3} | {'LEER':>8} | {'---':>6} | {'---':>5} | ({x},{y}) {w}x{h}")
            results.append({"number": None})
            continue

        result = classify_tile(tile_image)
        number = result["number"]
        confidence = result["confidence"]
        is_joker = result["is_joker"]

        label = "Joker" if is_joker else str(number) if number else "?"
        print(f"{i+1:>3} | {label:>8} | {confidence:>5.1%} | {'Ja' if is_joker else 'Nein':>5} | ({x},{y}) {w}x{h}")

        results.append({"number": number})

        if is_joker:
            total_score += 20
        elif number:
            total_score += number

        # Tile-Bild speichern mit Label
        tile_filename = f"{i+1:02d}_{label}_conf{confidence:.0%}.jpg"
        cv2.imwrite(str(debug_dir / tile_filename), tile_image)

    print(f"\nGesamtpunktzahl: {total_score}")
    print(f"\nEinzelne Tiles gespeichert in: {debug_dir}/")

    # Debug-Bild mit Markierungen erzeugen
    debug_image = draw_detections(resized, tile_regions, results)
    debug_path = "debug_detections.jpg"
    cv2.imwrite(debug_path, debug_image)
    print(f"Markiertes Bild gespeichert: {debug_path}")


if __name__ == "__main__":
    main()
