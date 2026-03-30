"""
YOLO Labeling-Tool: Annotiert Rummikub-Gesamtbilder mit Bounding Boxes.

Öffnet jedes Bild und nutzt das vorhandene CNN + OpenCV-Detector
um automatisch Vorschläge zu generieren. Du korrigierst dann nur noch.

Nutzung:
    python label_for_yolo.py --images pfad/zu/bildern/
    python label_for_yolo.py --edit-mode                  # Bereits gelabelte Bilder bearbeiten
    python label_for_yolo.py --edit-mode --split val      # Val-Labels bearbeiten
    python label_for_yolo.py --edit-mode 20260313_174707.jpg  # Einzelnes Bild bearbeiten

Steuerung:
    Rechte Maustaste    → Bounding Box zeichnen (Klick + Ziehen)
    Linke Maustaste     → Ecke resizen / Box auswählen
    1-9                 → Klasse der ausgewählten Box auf 1-9 setzen
    F1=10, F2=11, F3=12, F4=13
    j                   → Klasse auf Joker setzen
    r                   → Bild um 90° drehen (im Uhrzeigersinn)
    x / Entf            → Ausgewählte Box löschen (Linksklick auf Box zum Auswählen)
    Strg+Z              → Letztes Löschen rückgängig machen
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
from PIL import Image, ImageOps

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


def _boxes_display_to_image(boxes: list, scale: float) -> list:
    """Skaliert Box-Koordinaten von der Anzeigegröße zurück auf die Bildgröße."""
    if scale <= 0:
        return boxes
    inv = 1.0 / scale
    return [
        (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv), cls)
        for (x1, y1, x2, y2, cls) in boxes
    ]


def _boxes_image_to_display(boxes: list, scale: float) -> list:
    """Skaliert Box-Koordinaten von der Bildgröße auf die Anzeigegröße."""
    return [
        (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale), cls)
        for (x1, y1, x2, y2, cls) in boxes
    ]


def _rotate_boxes_90_cw(boxes: list, img_w: int, img_h: int) -> list:
    """Dreht alle Boxen um 90° im Uhrzeigersinn (in Bildkoordinaten)."""
    rotated = []
    if not boxes:
        return rotated

    for (x1, y1, x2, y2, cls) in boxes:
        # Vier Ecken des Rechtecks
        corners = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]
        # Mapping für 90° CW: (x, y) -> (h - 1 - y, x)
        transformed = [
            (img_h - 1 - y, x) for (x, y) in corners
        ]
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]
        nx1, nx2 = int(min(xs)), int(max(xs))
        ny1, ny2 = int(min(ys)), int(max(ys))
        rotated.append((nx1, ny1, nx2, ny2, cls))

    return rotated


def _load_image_exif(path: Path) -> np.ndarray | None:
    """Lädt ein Bild mit korrekter EXIF-Orientierung."""
    try:
        pil_img = Image.open(str(path))
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


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
        self.undo_stack = []     # Stack für Rückgängig-Funktion
        self.box_confidences = {}  # (x1,y1,x2,y2,cls) → confidence

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
            conf = self.box_confidences.get((x1, y1, x2, y2, cls))
            if conf is not None:
                conf_text = f"{conf:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                (tw, th), _ = cv2.getTextSize(conf_text, font, font_scale, 1)
                tx = x1 + (x2 - x1 - tw) // 2
                ty = y1 + (y2 - y1 + th) // 2
                cv2.putText(self.display, conf_text, (tx, ty), font, font_scale, color, 1)
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


def _compute_iou(a, b) -> float:
    """Berechnet Intersection over Union zweier Boxen (x1,y1,x2,y2,...)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _remove_overlapping(boxes: list, iou_threshold: float = 0.3) -> list:
    """Entfernt stark überlappende Boxen (behält die kleinere / präzisere)."""
    if not boxes:
        return boxes
    # Sortiere nach Fläche (kleinste zuerst → wird bevorzugt behalten)
    sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    keep = []
    for box in sorted_boxes:
        if all(_compute_iou(box, kept) < iou_threshold for kept in keep):
            keep.append(box)
    return keep


def auto_detect_yolo(image: np.ndarray) -> list:
    """Nutzt YOLO für automatische Vorschläge. Gibt (x1,y1,x2,y2,cls,conf) zurück."""
    results = _yolo_model(image, conf=0.15, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), cls_idx, conf))
    return _remove_overlapping(boxes)


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

        conf = result.get("confidence", 0.0)
        boxes.append((ox1, oy1, ox2, oy2, cls_idx, conf))

    return _remove_overlapping(boxes)


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


def from_yolo_format(yolo_txt: str, img_w: int, img_h: int) -> list:
    """Liest YOLO-Format zurück in Bounding Boxes (x1, y1, x2, y2, cls)."""
    boxes = []
    for line in yolo_txt.strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        boxes.append((x1, y1, x2, y2, cls))
    return boxes


def main():
    parser = argparse.ArgumentParser(description="YOLO Labeling Tool für Rummikub")
    parser.add_argument("--images", type=str, required=False, help="Ordner mit Gesamtbildern")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="Ziel-Split (Standard: train)")
    parser.add_argument("--edit-mode", nargs="?", const=True, default=False,
                        metavar="FILENAME",
                        help="Bereits gelabelte Bilder bearbeiten (optional: einzelnes Bild angeben, z.B. --edit-mode 20260313_174707.jpg)")
    args = parser.parse_args()

    edit_mode = bool(args.edit_mode)

    out_images = YOLO_DIR / args.split / "images"
    out_labels = YOLO_DIR / args.split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    if edit_mode:
        # Im Edit-Modus: bereits gelabelte Bilder laden
        if isinstance(args.edit_mode, str):
            # Einzelnes Bild angegeben
            target = out_images / args.edit_mode
            if not target.exists():
                # Auch ohne Extension versuchen
                matches = list(out_images.glob(f"{Path(args.edit_mode).stem}.*"))
                if matches:
                    target = matches[0]
                else:
                    print(f"Bild '{args.edit_mode}' nicht in {out_images} gefunden.")
                    return
            image_files = [target]
            print(f"Edit-Modus: Bearbeite {target.name}")
        else:
            image_files = sorted(list(out_images.glob("*.jpg")) + list(out_images.glob("*.jpeg")) + list(out_images.glob("*.png")))
            if not image_files:
                print(f"Keine gelabelten Bilder in {out_images} gefunden.")
                return
            print(f"Edit-Modus: {len(image_files)} gelabelte Bilder gefunden.")
    else:
        if not args.images:
            print("Fehler: --images ist erforderlich (oder --edit-mode verwenden).")
            return
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

    # Bereits gelabelte Bilder überspringen (nur im Normal-Modus)
    existing = {p.stem for p in out_images.glob("*")} if not edit_mode else set()

    total_labeled = 0
    window_name = "Rummikub YOLO Labeling"

    for i, img_path in enumerate(image_files):
        if img_path.stem in existing:
            print(f"  Überspringe {img_path.name} (bereits gelabelt)")
            continue

        original = _load_image_exif(img_path)
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

        # Im Edit-Modus: bestehende Labels laden
        initial_boxes = []
        if edit_mode:
            label_path = out_labels / (img_path.stem + ".txt")
            if label_path.exists():
                yolo_txt = label_path.read_text(encoding="utf-8")
                orig_boxes = from_yolo_format(yolo_txt, w, h)
                # Auf Display-Größe skalieren
                initial_boxes = [
                    (int(x1*display_scale), int(y1*display_scale), int(x2*display_scale), int(y2*display_scale), c)
                    for (x1, y1, x2, y2, c) in orig_boxes
                ]
                print(f"  {len(initial_boxes)} bestehende Boxen geladen.")

        annotator = BBoxAnnotator(display_img, initial_boxes)
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
                detections = auto_detect(original)
                # Skalierung auf Display-Größe + Confidence speichern
                annotator.boxes = []
                annotator.box_confidences = {}
                for det in detections:
                    x1, y1, x2, y2, c = det[:5]
                    conf = det[5] if len(det) > 5 else None
                    sx1 = int(x1 * display_scale)
                    sy1 = int(y1 * display_scale)
                    sx2 = int(x2 * display_scale)
                    sy2 = int(y2 * display_scale)
                    annotator.boxes.append((sx1, sy1, sx2, sy2, c))
                    if conf is not None:
                        annotator.box_confidences[(sx1, sy1, sx2, sy2, c)] = conf
                annotator._redraw()
                print(f"  {len(annotator.boxes)} Steine vorgeschlagen.")

            elif key == ord('r'):
                # Bild und Boxen um 90° im Uhrzeigersinn drehen
                print("  Bild um 90° drehen (CW)...")
                # Aktuelle Boxen von Display- in Bildkoordinaten zurückskalieren
                img_h, img_w = original.shape[:2]
                img_boxes = _boxes_display_to_image(annotator.boxes, display_scale)

                # Bild drehen
                original = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)

                # Boxen in Bildkoordinaten drehen (unter Verwendung der alten Dimensionen)
                rotated_img_boxes = _rotate_boxes_90_cw(img_boxes, img_w, img_h)

                # Neue Anzeigegröße berechnen
                h, w = original.shape[:2]
                display_scale = min(screen_w / w, screen_h / h, 1.0)
                if display_scale < 1.0:
                    display_img = cv2.resize(original, (int(w * display_scale), int(h * display_scale)))
                else:
                    display_img = original.copy()

                # Boxen wieder auf Anzeigegröße skalieren
                annotator.boxes = _boxes_image_to_display(rotated_img_boxes, display_scale)
                annotator.image = display_img.copy()
                annotator.undo_stack.clear()  # Historie zurücksetzen, da Koordinaten geändert wurden
                cv2.resizeWindow(window_name, display_img.shape[1], display_img.shape[0])
                annotator._redraw()

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
                    annotator.undo_stack.append([removed])
                    annotator._redraw()
                    print(f"  Box gelöscht (Klasse {CLASS_NAMES.get(removed[4], '?')}) – Strg+Z zum Wiederherstellen")
            elif key == 26:  # Strg+Z → Rückgängig
                if annotator.undo_stack:
                    restored = annotator.undo_stack.pop()
                    annotator.boxes.extend(restored)
                    annotator._redraw()
                    print(f"  {len(restored)} Box(en) wiederhergestellt.")
                else:
                    print("  Nichts zum Rückgängig machen.")
            elif key == ord('c'):  # c → alle Boxen löschen
                count = len(annotator.boxes)
                if count > 0:
                    annotator.undo_stack.append(list(annotator.boxes))
                annotator.boxes.clear()
                annotator._redraw()
                print(f"  Alle {count} Boxen gelöscht. – Strg+Z zum Wiederherstellen")

            elif key == ord('t'):  # t → Quick-Train YOLO
                global _yolo_model
                train_imgs = list((YOLO_DIR / "train" / "images").glob("*"))
                if len(train_imgs) < 5:
                    print(f"  Zu wenige Labels ({len(train_imgs)}). Mindestens 5 zum Trainieren.")
                else:
                    epochs = 50 if len(train_imgs) < 20 else 30
                    print(f"  Quick-Training mit {len(train_imgs)} Bildern ({epochs} Epochen)...")
                    cv2.destroyAllWindows()
                    try:
                        from ultralytics import YOLO as UltralyticsYOLO
                        import torch
                        import yaml

                        device = "0" if torch.cuda.is_available() else "cpu"
                        # Vom vorherigen Modell weitertrainieren wenn vorhanden
                        base_model = str(YOLO_MODEL) if YOLO_MODEL.exists() else "yolov8n.pt"
                        yolo = UltralyticsYOLO(base_model)
                        val_imgs = list((YOLO_DIR / "val" / "images").glob("*"))
                        has_val = len(val_imgs) > 0

                        # Temporäre data.yaml ohne val wenn keine Val-Bilder vorhanden
                        orig_yaml = YOLO_DIR / "data.yaml"
                        with open(orig_yaml, "r", encoding="utf-8") as f:
                            data_cfg = yaml.safe_load(f)
                        if not has_val:
                            data_cfg["val"] = data_cfg["train"]  # YOLO braucht val-Eintrag
                        tmp_yaml = YOLO_DIR / "data_quick.yaml"
                        with open(tmp_yaml, "w", encoding="utf-8") as f:
                            yaml.dump(data_cfg, f, default_flow_style=False)

                        yolo.train(
                            data=str(tmp_yaml.resolve()), epochs=epochs, imgsz=640, batch=16,
                            device=device, project=str(SCRIPT_DIR / "runs" / "detect"),
                            name="quick", exist_ok=True, verbose=False,
                            val=has_val, workers=0,
                        )
                        # best.pt nur wenn Validierung aktiv, sonst last.pt
                        weights_dir = SCRIPT_DIR / "runs" / "detect" / "quick" / "weights"
                        best = weights_dir / ("best.pt" if has_val else "last.pt")
                        if best.exists():
                            import shutil
                            YOLO_MODEL.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(best), str(YOLO_MODEL))
                            _yolo_model = UltralyticsYOLO(str(YOLO_MODEL))
                            print(f"  ✅ YOLO-Modell trainiert und geladen! Auto-Detect verbessert.")
                    except Exception as e:
                        print(f"  Fehler beim Training: {e}")
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, display_img.shape[1], display_img.shape[0])
                    cv2.setMouseCallback(window_name, annotator.mouse_callback)

    cv2.destroyAllWindows()
    print(f"\nFertig! {total_labeled} Bilder gelabelt.")
    print(f"Bilder: {out_images}")
    print(f"Labels: {out_labels}")


if __name__ == "__main__":
    main()
