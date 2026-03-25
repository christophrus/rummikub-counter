"""
Korrigiert die Orientierung von Portrait-Bildern im YOLO-Dataset.

Zeigt jedes Portrait-Bild an und laesst den Benutzer die Rotation waehlen.
Rotiert dann Bild UND YOLO-Labels zusammen.

Tasten:
    r  = 90° CW drehen (Bild ist nach rechts gekippt)
    l  = 90° CCW drehen (Bild ist nach links gekippt)
    s  = Ueberspringen (Bild ist OK)
    q  = Beenden

Nutzung:
    python fix_portrait_orientation.py
    python fix_portrait_orientation.py --dry-run   # Nur anzeigen, nichts aendern
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).parent
YOLO_DIR = SCRIPT_DIR.parent / "yolo_dataset"


def rotate_yolo_labels_cw90(labels: list[tuple]) -> list[tuple]:
    """Rotiert YOLO-Labels um 90° im Uhrzeigersinn.
    
    Bei 90° CW: (xc, yc) -> (1 - yc, xc), (w, h) -> (h, w)
    """
    rotated = []
    for cls, xc, yc, w, h in labels:
        new_xc = 1.0 - yc
        new_yc = xc
        new_w = h
        new_h = w
        rotated.append((cls, new_xc, new_yc, new_w, new_h))
    return rotated


def rotate_yolo_labels_ccw90(labels: list[tuple]) -> list[tuple]:
    """Rotiert YOLO-Labels um 90° gegen den Uhrzeigersinn (= 270° CW).
    
    Bei 270° CW: (xc, yc) -> (yc, 1 - xc), (w, h) -> (h, w)
    """
    rotated = []
    for cls, xc, yc, w, h in labels:
        new_xc = yc
        new_yc = 1.0 - xc
        new_w = h
        new_h = w
        rotated.append((cls, new_xc, new_yc, new_w, new_h))
    return rotated


def read_yolo_labels(label_path: Path) -> list[tuple]:
    """Liest YOLO-Label-Datei."""
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        cls = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        labels.append((cls, xc, yc, w, h))
    return labels


def write_yolo_labels(label_path: Path, labels: list[tuple]):
    """Schreibt YOLO-Label-Datei."""
    lines = []
    for cls, xc, yc, w, h in labels:
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_boxes(image: np.ndarray, labels: list[tuple]) -> np.ndarray:
    """Zeichnet YOLO-Boxes auf ein Bild zur Vorschau."""
    vis = image.copy()
    h, w = vis.shape[:2]
    for cls, xc, yc, bw, bh in labels:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    return vis


def resize_for_display(image: np.ndarray, max_h: int = 900) -> np.ndarray:
    """Skaliert Bild fuer Bildschirmanzeige."""
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def main():
    parser = argparse.ArgumentParser(description="Portrait-Bilder im YOLO-Dataset aufrichten")
    parser.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts aendern")
    parser.add_argument("--split", default="train", help="Split (train/val/test, Standard: train)")
    args = parser.parse_args()

    img_dir = YOLO_DIR / args.split / "images"
    lbl_dir = YOLO_DIR / args.split / "labels"

    if not img_dir.exists():
        print(f"Verzeichnis nicht gefunden: {img_dir}")
        return

    # Alle Portrait-Bilder finden
    portrait_images = []
    for f in sorted(img_dir.glob("*.jpg")):
        img = cv2.imread(str(f))
        if img is not None and img.shape[0] > img.shape[1]:
            portrait_images.append(f)

    if not portrait_images:
        print("Keine Portrait-Bilder gefunden.")
        return

    print(f"\n{len(portrait_images)} Portrait-Bilder gefunden in {args.split}/")
    print("=" * 50)
    print("Tasten:  r = 90° CW  |  l = 90° CCW  |  s = Skip  |  q = Quit")
    print("=" * 50)

    stats = {"cw": 0, "ccw": 0, "skip": 0}

    for i, img_path in enumerate(portrait_images):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        label_path = lbl_dir / (img_path.stem + ".txt")
        labels = read_yolo_labels(label_path)

        # Vorschau: Original mit Boxes
        vis = draw_boxes(img, labels)
        vis = resize_for_display(vis)
        window_name = f"[{i+1}/{len(portrait_images)}] {img_path.name} ({w}x{h}) - r/l/s/q"
        cv2.imshow(window_name, vis)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("r"):
                # 90° CW
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                rotated_labels = rotate_yolo_labels_cw90(labels)

                if not args.dry_run:
                    cv2.imwrite(str(img_path), rotated_img)
                    if labels:
                        write_yolo_labels(label_path, rotated_labels)

                stats["cw"] += 1
                action = "90° CW" + (" (dry-run)" if args.dry_run else "")
                print(f"  [{i+1}/{len(portrait_images)}] {img_path.name}: {action}")

                # Kurze Vorschau des Ergebnisses
                result_vis = draw_boxes(rotated_img, rotated_labels)
                result_vis = resize_for_display(result_vis)
                cv2.imshow("Ergebnis", result_vis)
                cv2.waitKey(800)
                cv2.destroyWindow("Ergebnis")
                break

            elif key == ord("l"):
                # 90° CCW (= 270° CW)
                rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_labels = rotate_yolo_labels_ccw90(labels)

                if not args.dry_run:
                    cv2.imwrite(str(img_path), rotated_img)
                    if labels:
                        write_yolo_labels(label_path, rotated_labels)

                stats["ccw"] += 1
                action = "90° CCW" + (" (dry-run)" if args.dry_run else "")
                print(f"  [{i+1}/{len(portrait_images)}] {img_path.name}: {action}")

                result_vis = draw_boxes(rotated_img, rotated_labels)
                result_vis = resize_for_display(result_vis)
                cv2.imshow("Ergebnis", result_vis)
                cv2.waitKey(800)
                cv2.destroyWindow("Ergebnis")
                break

            elif key == ord("s"):
                stats["skip"] += 1
                print(f"  [{i+1}/{len(portrait_images)}] {img_path.name}: Uebersprungen")
                break

            elif key == ord("q"):
                print("\nAbgebrochen.")
                cv2.destroyAllWindows()
                print(f"\nErgebnis: {stats['cw']} CW, {stats['ccw']} CCW, {stats['skip']} Skip")
                return

        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()
    print(f"\nFertig! {stats['cw']} x 90° CW, {stats['ccw']} x 90° CCW, {stats['skip']} x Skip")
    if args.dry_run:
        print("(Dry-Run: Keine Dateien geaendert)")


if __name__ == "__main__":
    main()
