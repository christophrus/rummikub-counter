"""Visualisiert YOLO-Labels auf den Originalbildern zur Überprüfung."""

import argparse
from pathlib import Path
import cv2

YOLO_DIR = Path(__file__).parent.parent / "yolo_dataset"
CLASS_NAMES = {i: str(i + 1) for i in range(13)}
CLASS_NAMES[13] = "joker"
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
    (200, 200, 0), (128, 128, 255),
]


def main():
    parser = argparse.ArgumentParser(description="YOLO Labels auf Bildern visualisieren")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    img_dir = YOLO_DIR / args.split / "images"
    lbl_dir = YOLO_DIR / args.split / "labels"

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        print(f"Keine Bilder in {img_dir}")
        return

    print(f"{len(images)} Bilder. Pfeiltasten: vor/zurück, q: beenden")

    idx = 0
    while 0 <= idx < len(images):
        img_path = images[idx]
        label_path = lbl_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Labels einlesen und zeichnen
        box_count = 0
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                color = COLORS[cls % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                label = CLASS_NAMES.get(cls, "?")
                cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                box_count += 1

        # Für Anzeige skalieren
        scale = min(1600 / w, 900 / h, 1.0)
        display = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

        title = f"[{idx+1}/{len(images)}] {img_path.name} ({w}x{h}) - {box_count} Boxen"
        print(f"  {title}")

        cv2.namedWindow("Verify Labels", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verify Labels", display.shape[1], display.shape[0])
        cv2.imshow("Verify Labels", display)

        key = cv2.waitKey(0) & 0xFFFF
        if key == ord('q'):
            break
        elif key in (ord('d'), 2555904, 32):  # d, Pfeil rechts, Leertaste
            idx += 1
        elif key in (ord('a'), 2424832):  # a, Pfeil links
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
