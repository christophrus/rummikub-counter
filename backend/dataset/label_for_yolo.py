"""
YOLO Labeling-Tool: Annotiert Rummikub-Gesamtbilder mit Bounding Boxes.

Öffnet jedes Bild und nutzt das vorhandene CNN + OpenCV-Detector
um automatisch Vorschläge zu generieren. Du korrigierst dann nur noch.

Nutzung:
    python label_for_yolo.py --images pfad/zu/bildern/

Steuerung:
    Rechte Maustaste    → Bounding Box zeichnen (Klick + Ziehen)
    Linke Maustaste     → Ecke resizen / Box auswählen
    1-9                 → Klasse der ausgewählten Box auf 1-9 setzen
    F1=10, F2=11, F3=12, F4=13
    j                   → Klasse auf Joker setzen
    x / Entf            → Ausgewählte Box löschen (Linksklick auf Box zum Auswählen)
    c                   → Alle Boxen löschen
    a                   → Auto-Detect (YOLO oder CNN Vorschläge)
    t                   → Quick-Train YOLO mit bisherigen Labels
    s / Enter           → Speichern und nächstes Bild
    d                   → Bild überspringen (nicht speichern)
    q                   → Beenden
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.tile_detector import detect_tiles
from app.services.cnn_classifier import load_model, classify_tile
from app.utils.image_processing import resize_image, preprocess_image, extract_tile_region

# YOLO-Klassen: Index → Label
CLASS_NAMES = {i: str(i + 1) for i in range(13)}
CLASS_NAMES[13] = "joker"

# Label → YOLO-Index
LABEL_TO_IDX = {str(i + 1): i for i in range(13)}
LABEL_TO_IDX["joker"] = 13

SCRIPT_DIR = Path(__file__).parent
YOLO_DIR = SCRIPT_DIR.parent / "yolo_dataset"
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
    (200, 200, 0), (128, 128, 255),
]


class BBoxAnnotator:
    HANDLE_RADIUS = 8  # Pixel-Radius für Eck-Anfasser

    def __init__(self, image: np.ndarray, boxes: list):
        self.image = image.copy()
        self.display = image.copy()
        self.boxes = boxes  # [(x1, y1, x2, y2, class_idx), ...]
        self.drawing = False
        self.resizing = False
        self.resize_idx = -1     # Index der Box die resized wird
        self.resize_corner = -1  # 0=TL, 1=TR, 2=BR, 3=BL
        self.start_x = 0
        self.start_y = 0
        self.current_class = 0

    def _find_handle(self, x, y):
        """Prüft ob (x,y) auf einem Eck-Anfasser liegt. Gibt (box_idx, corner) oder None zurück."""
        r = self.HANDLE_RADIUS
        for i in range(len(self.boxes) - 1, -1, -1):
            bx1, by1, bx2, by2, _ = self.boxes[i]
            corners = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
            for ci, (cx, cy) in enumerate(corners):
                if abs(x - cx) <= r and abs(y - cy) <= r:
                    return i, ci
        return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            handle = self._find_handle(x, y)
            if handle:
                self.resizing = True
                self.resize_idx, self.resize_corner = handle
            else:
                # Klick auf bestehende Box → auswählen (für Klassen-Änderung)
                for i in range(len(self.boxes) - 1, -1, -1):
                    bx1, by1, bx2, by2, _ = self.boxes[i]
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        # Box ans Ende verschieben (= "ausgewählt")
                        self.boxes.append(self.boxes.pop(i))
                        self._redraw()
                        return

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.resizing:
                bx1, by1, bx2, by2, cls = self.boxes[self.resize_idx]
                if self.resize_corner == 0:    # Top-Left
                    bx1, by1 = x, y
                elif self.resize_corner == 1:  # Top-Right
                    bx2, by1 = x, y
                elif self.resize_corner == 2:  # Bottom-Right
                    bx2, by2 = x, y
                elif self.resize_corner == 3:  # Bottom-Left
                    bx1, by2 = x, y
                self.boxes[self.resize_idx] = (bx1, by1, bx2, by2, cls)
                self._redraw()
            elif self.drawing:
                self.display = self.image.copy()
                self._draw_all_boxes()
                cv2.rectangle(self.display, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.resizing:
                # Koordinaten normalisieren (x1<x2, y1<y2)
                bx1, by1, bx2, by2, cls = self.boxes[self.resize_idx]
                self.boxes[self.resize_idx] = (min(bx1, bx2), min(by1, by2),
                                                max(bx1, bx2), max(by1, by2), cls)
                self.resizing = False
                self.resize_idx = -1
                self._redraw()

        elif event == cv2.EVENT_RBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1 = min(self.start_x, x)
                y1 = min(self.start_y, y)
                x2 = max(self.start_x, x)
                y2 = max(self.start_y, y)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.boxes.append((x1, y1, x2, y2, self.current_class))
            self._redraw()

    def _draw_all_boxes(self):
        r = self.HANDLE_RADIUS
        for idx, (x1, y1, x2, y2, cls) in enumerate(self.boxes):
            color = COLORS[cls % len(COLORS)]
            is_selected = (idx == len(self.boxes) - 1) and len(self.boxes) > 0
            thickness = 3 if is_selected else 2
            cv2.rectangle(self.display, (x1, y1), (x2, y2), color, thickness)
            if is_selected:
                # Gestrichelte Umrandung als Auswahl-Indikator
                cv2.rectangle(self.display, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 1)
            label = CLASS_NAMES.get(cls, "?")
            cv2.putText(self.display, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Eck-Anfasser zeichnen
            for (cx, cy) in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                cv2.circle(self.display, (cx, cy), r, color, -1)

    def _redraw(self):
        self.display = self.image.copy()
        self._draw_all_boxes()
        # Status-Leiste
        h = self.display.shape[0]
        status = f"Klasse: {CLASS_NAMES.get(self.current_class, '?')} | Boxen: {len(self.boxes)} | s=Speichern d=Skip q=Quit a=AutoDetect"
        cv2.putText(self.display, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def set_class(self, cls_idx):
        self.current_class = cls_idx
        if self.boxes:
            # Letzte Box aktualisieren
            x1, y1, x2, y2, _ = self.boxes[-1]
            self.boxes[-1] = (x1, y1, x2, y2, cls_idx)
        self._redraw()

    def get_display(self):
        return self.display


YOLO_MODEL = YOLO_DIR.parent / "models" / "rummikub_yolo.pt"

_yolo_model = None


def _load_yolo_if_available():
    """Lädt YOLO-Modell falls vorhanden, für bessere Auto-Detect-Vorschläge."""
    global _yolo_model
    if YOLO_MODEL.exists():
        from ultralytics import YOLO as UltralyticsYOLO
        _yolo_model = UltralyticsYOLO(str(YOLO_MODEL))
        print(f"  YOLO-Modell geladen: {YOLO_MODEL.name}")
        return True
    return False


def auto_detect_yolo(image: np.ndarray) -> list:
    """Nutzt YOLO für automatische Vorschläge."""
    results = _yolo_model(image, conf=0.15, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_idx = int(box.cls[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), cls_idx))
    return boxes


def auto_detect(image: np.ndarray) -> list:
    """Nutzt YOLO (wenn vorhanden) oder OpenCV + CNN für automatische Vorschläge."""
    if _yolo_model is not None:
        return auto_detect_yolo(image)

    resized = resize_image(image)
    enhanced = preprocess_image(image)

    scale_x = image.shape[1] / resized.shape[1]
    scale_y = image.shape[0] / resized.shape[0]

    tile_regions = detect_tiles(resized)
    boxes = []

    for tile_info in tile_regions:
        x, y, w, h = tile_info["x"], tile_info["y"], tile_info["w"], tile_info["h"]
        tile_img = extract_tile_region(enhanced, x, y, w, h)
        if tile_img.size == 0:
            continue

        result = classify_tile(tile_img)

        # Koordinaten zurückskalieren auf Originalgröße
        ox1 = int(x * scale_x)
        oy1 = int(y * scale_y)
        ox2 = int((x + w) * scale_x)
        oy2 = int((y + h) * scale_y)

        if result["is_joker"]:
            cls_idx = 13
        elif result["number"]:
            cls_idx = result["number"] - 1
        else:
            cls_idx = 0

        boxes.append((ox1, oy1, ox2, oy2, cls_idx))

    return boxes


def to_yolo_format(boxes: list, img_w: int, img_h: int) -> str:
    """Konvertiert Bounding Boxes in YOLO-Format."""
    lines = []
    for (x1, y1, x2, y2, cls) in boxes:
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="YOLO Labeling Tool für Rummikub")
    parser.add_argument("--images", type=str, required=True, help="Ordner mit Gesamtbildern")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="Ziel-Split (Standard: train)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"Fehler: {images_dir} existiert nicht.")
        return

    image_files = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.png"))
    )

    if not image_files:
        print(f"Keine Bilder in {images_dir} gefunden.")
        return

    print(f"{len(image_files)} Bilder gefunden.")

    # YOLO-Modell bevorzugen, CNN als Fallback
    if not _load_yolo_if_available():
        print("Kein YOLO-Modell gefunden, lade CNN für Auto-Detect...")
        load_model()
    print("  Tipp: Drücke 't' um YOLO mit bisherigen Labels zu trainieren.")

    out_images = YOLO_DIR / args.split / "images"
    out_labels = YOLO_DIR / args.split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Bereits gelabelte Bilder überspringen
    existing = {p.stem for p in out_images.glob("*")}

    total_labeled = 0
    window_name = "Rummikub YOLO Labeling"

    for i, img_path in enumerate(image_files):
        if img_path.stem in existing:
            print(f"  Überspringe {img_path.name} (bereits gelabelt)")
            continue

        original = cv2.imread(str(img_path))
        if original is None:
            continue

        # Display-Größe an Bildschirm anpassen
        screen_w, screen_h = 1600, 900  # Sichere Maximalwerte (mit Platz für Taskbar)
        h, w = original.shape[:2]
        display_scale = min(screen_w / w, screen_h / h, 1.0)
        if display_scale < 1.0:
            display_img = cv2.resize(original, (int(w * display_scale), int(h * display_scale)))
        else:
            display_img = original.copy()

        print(f"\n[{i+1}/{len(image_files)}] {img_path.name} ({w}x{h}, Anzeige: {display_img.shape[1]}x{display_img.shape[0]})")

        annotator = BBoxAnnotator(display_img, [])
        annotator._redraw()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_img.shape[1], display_img.shape[0])
        cv2.setMouseCallback(window_name, annotator.mouse_callback)

        while True:
            cv2.imshow(window_name, annotator.get_display())
            key = cv2.waitKeyEx(20)

            if key == -1:
                continue

            if key == ord('q'):
                cv2.destroyAllWindows()
                print(f"\nBeendet. {total_labeled} Bilder gelabelt.")
                return

            elif key == ord('a'):
                print("  Auto-Detecting...")
                boxes = auto_detect(original)
                # Skalierung auf Display-Größe
                annotator.boxes = [
                    (int(x1*display_scale), int(y1*display_scale), int(x2*display_scale), int(y2*display_scale), c)
                    for (x1, y1, x2, y2, c) in boxes
                ]
                annotator._redraw()
                print(f"  {len(annotator.boxes)} Steine vorgeschlagen.")

            elif key in (ord('s'), 13):  # s oder Enter
                if annotator.boxes:
                    # Boxen zurückskalieren auf Originalgröße für YOLO-Format
                    orig_h, orig_w = original.shape[:2]
                    inv_scale = 1.0 / display_scale
                    orig_boxes = [
                        (int(x1*inv_scale), int(y1*inv_scale), int(x2*inv_scale), int(y2*inv_scale), c)
                        for (x1, y1, x2, y2, c) in annotator.boxes
                    ]
                    yolo_txt = to_yolo_format(orig_boxes, orig_w, orig_h)

                    # Originalbild in voller Auflösung speichern
                    cv2.imwrite(str(out_images / img_path.name), original)
                    label_path = out_labels / (img_path.stem + ".txt")
                    label_path.write_text(yolo_txt, encoding="utf-8")

                    total_labeled += 1
                    print(f"  Gespeichert: {len(annotator.boxes)} Boxen → {args.split}/")
                else:
                    print("  Keine Boxen – übersprungen.")
                break

            elif key == ord('d'):
                print("  Übersprungen.")
                break

            # Klassen-Shortcuts
            elif key == ord('j'):
                annotator.set_class(13)
                print("  → Joker")
            elif ord('1') <= key <= ord('9'):
                cls = key - ord('1')
                annotator.set_class(cls)
                print(f"  → Klasse {cls + 1}")
            elif key == 7340032:  # F1 → Klasse 10
                annotator.set_class(9)
                print("  → Klasse 10")
            elif key == 7405568:  # F2 → Klasse 11
                annotator.set_class(10)
                print("  → Klasse 11")
            elif key == 7471104:  # F3 → Klasse 12
                annotator.set_class(11)
                print("  → Klasse 12")
            elif key == 7536640:  # F4 → Klasse 13
                annotator.set_class(12)
                print("  → Klasse 13")
            elif key in (ord('x'), 3014656):  # x oder Entf → ausgewählte Box löschen
                if annotator.boxes:
                    removed = annotator.boxes.pop()
                    annotator._redraw()
                    print(f"  Box gelöscht (Klasse {CLASS_NAMES.get(removed[4], '?')})")
            elif key == ord('c'):  # c → alle Boxen löschen
                count = len(annotator.boxes)
                annotator.boxes.clear()
                annotator._redraw()
                print(f"  Alle {count} Boxen gelöscht.")

            elif key == ord('t'):  # t → Quick-Train YOLO
                global _yolo_model
                train_imgs = list((YOLO_DIR / "train" / "images").glob("*"))
                if len(train_imgs) < 5:
                    print(f"  Zu wenige Labels ({len(train_imgs)}). Mindestens 5 zum Trainieren.")
                else:
                    print(f"  Quick-Training mit {len(train_imgs)} Bildern (20 Epochen)...")
                    cv2.destroyAllWindows()
                    try:
                        from ultralytics import YOLO as UltralyticsYOLO
                        data_yaml = str((YOLO_DIR / "data.yaml").resolve())
                        import torch
                        device = "0" if torch.cuda.is_available() else "cpu"
                        yolo = UltralyticsYOLO("yolov8n.pt")
                        val_imgs = list((YOLO_DIR / "val" / "images").glob("*"))
                        yolo.train(
                            data=data_yaml, epochs=20, imgsz=640, batch=16,
                            device=device, project=str(SCRIPT_DIR / "runs" / "detect"),
                            name="quick", exist_ok=True, verbose=False,
                            val=len(val_imgs) > 0,
                        )
                        best = SCRIPT_DIR / "runs" / "detect" / "quick" / "weights" / "best.pt"
                        if best.exists():
                            import shutil
                            YOLO_MODEL.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(best), str(YOLO_MODEL))
                            _yolo_model = UltralyticsYOLO(str(YOLO_MODEL))
                            print(f"  ✅ YOLO-Modell trainiert und geladen! Auto-Detect verbessert.")
                    except Exception as e:
                        print(f"  Fehler beim Training: {e}")
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
                    cv2.setMouseCallback(window_name, annotator.mouse_callback)

    cv2.destroyAllWindows()
    print(f"\nFertig! {total_labeled} Bilder gelabelt.")
    print(f"Bilder: {out_images}")
    print(f"Labels: {out_labels}")


if __name__ == "__main__":
    main()
