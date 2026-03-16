"""
Tile-Extractor: Schneidet einzelne Rummikub-Steine aus Gesamtbildern aus.

Nutzung:
    python extract_tiles.py

Das Script:
1. Liest alle Bilder aus dataset/raw/
2. Erkennt Steine mit OpenCV (gleiche Logik wie die App)
3. Schneidet jeden Stein aus
4. Zeigt jeden Stein an – du gibst die Zahl ein (1-13, j=Joker, s=Skip)
5. Speichert den Stein im richtigen Ordner (dataset/tiles/<zahl>/)
"""

import sys
import os

# Projektpfad hinzufügen damit app-Module importiert werden können
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from app.services.tile_detector import detect_tiles
from app.utils.image_processing import resize_image, preprocess_image, extract_tile_region

# Pfade
SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "raw"
TILES_DIR = SCRIPT_DIR / "tiles"

# Gültige Labels
VALID_LABELS = [str(i) for i in range(1, 14)] + ["j"]  # 1-13 + j(oker)

# Maus-Interaktionsstatus
HANDLE_NONE = -1
HANDLE_MOVE = 0
HANDLE_TOP = 1
HANDLE_BOTTOM = 2
HANDLE_LEFT = 3
HANDLE_RIGHT = 4
HANDLE_TL = 5
HANDLE_TR = 6
HANDLE_BL = 7
HANDLE_BR = 8
EDGE_GRAB = 10  # Pixel-Toleranz für Kantengriff


def _adjust_all_boxes(image: np.ndarray, tile_regions: list[dict]) -> str | list[dict]:
    """
    Zeigt alle erkannten Bounding Boxes auf dem Gesamtbild.
    Jede Box kann per Maus verschoben und an Kanten/Ecken skaliert werden.

    Steuerung:
        Kanten/Ecken ziehen  – Box-Größe anpassen
        Innenfläche ziehen   – Box verschieben
        R                    – Alle Boxen zurücksetzen
        Enter                – Übernehmen, weiter zum Labeln
        Q / Esc              – Komplett beenden

    Returns:
        Liste der (ggf. angepassten) tile_regions, oder 'quit'.
    """
    img_h, img_w = image.shape[:2]

    # Skalierung für Anzeige (Bild auf max 1200px)
    scale = min(1.0, 1200 / max(img_h, img_w))
    disp_w = int(img_w * scale)
    disp_h = int(img_h * scale)

    window_name = "Boxen anpassen: Ziehen=Bewegen/Skalieren | Enter=Weiter | R=Reset | Q=Beenden"

    # Arbeitskopie der Boxen
    boxes = [[t["x"], t["y"], t["w"], t["h"]] for t in tile_regions]
    orig_boxes = [b[:] for b in boxes]

    # Zustand
    state = {
        'active_idx': -1,     # welche Box wird gezogen
        'selected_idx': -1,   # zuletzt angeklickte Box (bleibt nach Loslassen)
        'handle': HANDLE_NONE,
        'drag_start': (0, 0),
        'box_at_drag_start': [0, 0, 0, 0],
        'drawing': False,     # Rechtsklick-Zeichnen aktiv
        'draw_start': (0, 0), # Startposition (Bildkoordinaten)
        'draw_end': (0, 0),   # Aktuelle Endposition (Bildkoordinaten)
    }

    def _hit_test(mx, my):
        """Prüft alle Boxen, gibt (box_index, handle) zurück."""
        g = max(6, int(EDGE_GRAB * scale))
        # Rückwärts iterieren damit obere (später gezeichnete) Boxen Priorität haben
        for idx in range(len(boxes) - 1, -1, -1):
            bx, by, bw, bh = boxes[idx]
            rx = int(bx * scale)
            ry = int(by * scale)
            rr = int((bx + bw) * scale)
            rb = int((by + bh) * scale)

            on_left = abs(mx - rx) < g
            on_right = abs(mx - rr) < g
            on_top = abs(my - ry) < g
            on_bottom = abs(my - rb) < g
            in_x = rx - g < mx < rr + g
            in_y = ry - g < my < rb + g

            if on_top and on_left:   return idx, HANDLE_TL
            if on_top and on_right:  return idx, HANDLE_TR
            if on_bottom and on_left:  return idx, HANDLE_BL
            if on_bottom and on_right: return idx, HANDLE_BR
            if on_top and in_x:      return idx, HANDLE_TOP
            if on_bottom and in_x:   return idx, HANDLE_BOTTOM
            if on_left and in_y:     return idx, HANDLE_LEFT
            if on_right and in_y:    return idx, HANDLE_RIGHT
            if rx < mx < rr and ry < my < rb:
                return idx, HANDLE_MOVE

        return -1, HANDLE_NONE

    def _apply_drag(idx, handle, dx, dy):
        sx, sy, sw, sh = state['box_at_drag_start']
        if handle == HANDLE_MOVE:
            boxes[idx] = [max(0, min(img_w - sw, sx + dx)),
                          max(0, min(img_h - sh, sy + dy)), sw, sh]
        elif handle == HANDLE_LEFT:
            nx = max(0, min(sx + sw - 20, sx + dx))
            boxes[idx] = [nx, sy, sw - (nx - sx), sh]
        elif handle == HANDLE_RIGHT:
            boxes[idx] = [sx, sy, max(20, min(img_w - sx, sw + dx)), sh]
        elif handle == HANDLE_TOP:
            ny = max(0, min(sy + sh - 20, sy + dy))
            boxes[idx] = [sx, ny, sw, sh - (ny - sy)]
        elif handle == HANDLE_BOTTOM:
            boxes[idx] = [sx, sy, sw, max(20, min(img_h - sy, sh + dy))]
        elif handle == HANDLE_TL:
            nx = max(0, min(sx + sw - 20, sx + dx))
            ny = max(0, min(sy + sh - 20, sy + dy))
            boxes[idx] = [nx, ny, sw - (nx - sx), sh - (ny - sy)]
        elif handle == HANDLE_TR:
            ny = max(0, min(sy + sh - 20, sy + dy))
            boxes[idx] = [sx, ny, max(20, min(img_w - sx, sw + dx)), sh - (ny - sy)]
        elif handle == HANDLE_BL:
            nx = max(0, min(sx + sw - 20, sx + dx))
            boxes[idx] = [nx, sy, sw - (nx - sx), max(20, min(img_h - sy, sh + dy))]
        elif handle == HANDLE_BR:
            boxes[idx] = [sx, sy, max(20, min(img_w - sx, sw + dx)),
                          max(20, min(img_h - sy, sh + dy))]

    def _mouse_cb(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx, handle = _hit_test(mx, my)
            if idx >= 0:
                state['active_idx'] = idx
                state['selected_idx'] = idx
                state['handle'] = handle
                state['drag_start'] = (mx, my)
                state['box_at_drag_start'] = boxes[idx][:]
            else:
                state['selected_idx'] = -1

        elif event == cv2.EVENT_MOUSEMOVE:
            if state['active_idx'] >= 0:
                dx = int((mx - state['drag_start'][0]) / scale)
                dy = int((my - state['drag_start'][1]) / scale)
                _apply_drag(state['active_idx'], state['handle'], dx, dy)
            elif state['drawing']:
                state['draw_end'] = (int(mx / scale), int(my / scale))

        elif event == cv2.EVENT_LBUTTONUP:
            state['active_idx'] = -1
            state['handle'] = HANDLE_NONE

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Neue Box zeichnen starten
            state['drawing'] = True
            ix, iy = int(mx / scale), int(my / scale)
            state['draw_start'] = (ix, iy)
            state['draw_end'] = (ix, iy)

        elif event == cv2.EVENT_RBUTTONUP:
            if state['drawing']:
                state['drawing'] = False
                x1, y1 = state['draw_start']
                x2, y2 = state['draw_end']
                bx = max(0, min(x1, x2))
                by = max(0, min(y1, y2))
                bw = min(img_w - bx, abs(x2 - x1))
                bh = min(img_h - by, abs(y2 - y1))
                if bw >= 10 and bh >= 10:
                    boxes.append([bx, by, bw, bh])
                    state['selected_idx'] = len(boxes) - 1

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse_cb)

    while True:
        frame = image.copy()
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            if idx == state['active_idx']:
                color = (0, 200, 255)     # orange: wird gerade gezogen
            elif idx == state['selected_idx']:
                color = (255, 200, 0)     # cyan: ausgewählt
            else:
                color = (0, 255, 0)       # grün: normal
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
            # Eckgriffe
            for px, py in [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]:
                cv2.circle(frame, (px, py), 4, (0, 200, 255), -1)
            cv2.putText(frame, f"#{idx+1}", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Vorschau der neuen Box beim Zeichnen
        if state['drawing']:
            x1, y1 = state['draw_start']
            x2, y2 = state['draw_end']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        display = cv2.resize(frame, (disp_w, disp_h),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # Enter
            break
        elif key in (27, ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            return 'quit'
        elif key in (ord('r'), ord('R')):
            boxes = [b[:] for b in orig_boxes]
            state['selected_idx'] = -1
        elif key in (ord('d'), ord('D'), 0):
            # Ausgewählte Box löschen (D oder Delete)
            si = state['selected_idx']
            if 0 <= si < len(boxes):
                boxes.pop(si)
                state['selected_idx'] = -1
                state['active_idx'] = -1

    cv2.destroyWindow(window_name)

    # Angepasste Regionen zurückgeben
    result = []
    for idx, (bx, by, bw, bh) in enumerate(boxes):
        if idx < len(tile_regions):
            t = dict(tile_regions[idx])
        else:
            t = {}  # manuell hinzugefügte Box
        t["x"], t["y"], t["w"], t["h"] = bx, by, bw, bh
        result.append(t)
    return result


def extract_tiles_from_image(image_path: Path, tile_counter: int) -> int:
    """
    Extrahiert Steine aus einem einzelnen Bild.
    Zeigt jeden Stein an und fragt nach dem Label.

    Returns:
        Aktualisierter tile_counter
    """
    print(f"\n{'='*60}")
    print(f"Bild: {image_path.name}")
    print(f"{'='*60}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  FEHLER: Bild konnte nicht geladen werden.")
        return tile_counter

    processed = resize_image(image)
    tile_regions = detect_tiles(processed)

    print(f"  {len(tile_regions)} Steine erkannt.")

    if not tile_regions:
        # Gesamtbild anzeigen zur Kontrolle
        display = cv2.resize(processed, (800, 600)) if max(processed.shape[:2]) > 800 else processed
        cv2.imshow("Keine Steine erkannt - Taste druecken", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return tile_counter

    # Alle Boxen auf dem Gesamtbild anpassen (Drag & Drop)
    adjusted = _adjust_all_boxes(processed, tile_regions)
    if adjusted == 'quit':
        print("Abgebrochen.")
        return tile_counter
    tile_regions = adjusted

    # Jeden Stein einzeln labeln
    for i, tile in enumerate(tile_regions):
        x, y, w, h = tile["x"], tile["y"], tile["w"], tile["h"]
        tile_image = extract_tile_region(processed, x, y, w, h, padding=3)

        if tile_image.size == 0:
            continue

        # Phase 2: Stein vergrößert anzeigen und Label abfragen
        display_tile = cv2.resize(tile_image, (200, 300), interpolation=cv2.INTER_CUBIC)
        cv2.imshow(f"Stein #{i+1}/{len(tile_regions)} - Label eingeben", display_tile)
        cv2.waitKey(1)  # Fenster aktualisieren

        while True:
            label = input(f"  Stein #{i+1}: Welche Zahl? (1-13 / j=Joker / s=Skip / q=Quit): ").strip().lower()

            if label == "q":
                cv2.destroyAllWindows()
                print("Abgebrochen.")
                return tile_counter

            if label == "s":
                print("    → Übersprungen")
                break

            if label in VALID_LABELS:
                folder_name = "joker" if label == "j" else label
                target_dir = TILES_DIR / folder_name
                target_dir.mkdir(parents=True, exist_ok=True)

                tile_counter += 1
                filename = f"tile_{tile_counter:04d}.png"
                target_path = target_dir / filename

                cv2.imwrite(str(target_path), tile_image)
                print(f"    → Gespeichert: {folder_name}/{filename}")
                break
            else:
                print(f"    Ungültig! Bitte 1-13, j, s oder q eingeben.")

        cv2.destroyAllWindows()

    return tile_counter


def count_existing_tiles() -> int:
    """Zählt vorhandene Tiles um den Counter fortzusetzen."""
    count = 0
    for folder in TILES_DIR.iterdir():
        if folder.is_dir():
            count += len(list(folder.glob("*.png")))
    return count


def print_stats():
    """Zeigt Statistiken über das bisherige Dataset."""
    print(f"\n{'='*60}")
    print("Dataset-Statistiken:")
    print(f"{'='*60}")
    total = 0
    for label in [str(i) for i in range(1, 14)] + ["joker"]:
        folder = TILES_DIR / label
        if folder.exists():
            count = len(list(folder.glob("*.png")))
        else:
            count = 0
        total += count
        bar = "█" * count + "░" * max(0, 20 - count)
        status = "✓" if count >= 20 else " "
        print(f"  {status} {label:>5}: {bar} {count}")

    print(f"\n  Gesamt: {total} Steine")
    missing = [l for l in ([str(i) for i in range(1, 14)] + ["joker"])
               if not (TILES_DIR / l).exists() or len(list((TILES_DIR / l).glob("*.png"))) < 20]
    if missing:
        print(f"  Noch fehlend (< 20 Bilder): {', '.join(missing)}")
    else:
        print("  ✓ Alle Klassen haben mindestens 20 Bilder!")


def main():
    print("=" * 60)
    print("  Rummikub Tile-Extractor")
    print("  Schneidet Steine aus deinen Fotos aus")
    print("=" * 60)

    # Prüfen ob raw-Bilder vorhanden
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True)
        print(f"\nBitte lege deine Fotos in diesen Ordner:")
        print(f"  {RAW_DIR}")
        return

    image_files = sorted([
        f for f in RAW_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    ])

    if not image_files:
        print(f"\nKeine Bilder in {RAW_DIR} gefunden!")
        print("Unterstützte Formate: JPG, PNG, WebP, BMP")
        return

    print(f"\n{len(image_files)} Bilder gefunden in {RAW_DIR}")

    # Bisherigen Counter ermitteln
    tile_counter = count_existing_tiles()
    if tile_counter > 0:
        print(f"{tile_counter} Steine bereits vorhanden (werden fortgesetzt)")

    print("\nSteuerung:")
    print("  Schritt 1 – Boxen anpassen (OpenCV-Fenster):")
    print("    Linksklick ziehen   = Box verschieben / Kanten+Ecken skalieren")
    print("    Rechtsklick ziehen  = Neue Box zeichnen")
    print("    D / Delete          = Ausgewählte Box löschen")
    print("    R                   = Alle Boxen zurücksetzen")
    print("    Enter               = Übernehmen, weiter zum Labeln")
    print("    Q / Esc             = Beenden")
    print("  Schritt 2 – Label vergeben (Terminal):")
    print("    1-13  = Zahlenwert des Steins")
    print("    j     = Joker")
    print("    s     = Stein überspringen")
    print("    q     = Beenden")
    print()

    input("Enter drücken zum Starten...")

    for image_path in image_files:
        tile_counter = extract_tiles_from_image(image_path, tile_counter)

    print_stats()
    print("\nFertig! Nächster Schritt: python augment_dataset.py")


if __name__ == "__main__":
    main()
