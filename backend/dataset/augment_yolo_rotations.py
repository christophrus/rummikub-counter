"""
Erweitert YOLO-Datensaetze um exakte 90- und 270-Grad-Rotationen.

180 Grad wird bewusst weggelassen, um 6/9-Verwechslungen zu vermeiden.
Bei Hochformat-Bildern (h > w) wird nur 270 Grad erzeugt, da 90 Grad
einer 180-Grad-Drehung der Originalszene entspricht.

Nutzung:
    python augment_yolo_rotations.py
    python augment_yolo_rotations.py --splits train val
    python augment_yolo_rotations.py --overwrite

Vorhandene Rotationsdateien werden standardmaessig uebersprungen.
Mit --overwrite werden sie neu erzeugt.
"""

import argparse
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).parent
YOLO_DIR = SCRIPT_DIR.parent / "yolo_dataset"
ROTATION_SUFFIXES = ("_rot90", "_rot180", "_rot270")
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def load_yolo_labels(label_path: Path) -> list:
    """Laedt YOLO-Labels als (cls, xc, yc, w, h)."""
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        boxes.append((cls, xc, yc, width, height))

    return boxes


def save_yolo_labels(label_path: Path, boxes: list) -> None:
    """Speichert YOLO-Labels mit 6 Nachkommastellen."""
    lines = [
        f"{cls} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"
        for cls, xc, yc, width, height in boxes
    ]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def rotate_boxes(boxes: list, angle: int) -> list:
    """Rotiert YOLO-Labels fuer exakte 90er-Schritte."""
    rotated = []

    for cls, xc, yc, width, height in boxes:
        if angle == 90:
            new_xc = 1.0 - yc
            new_yc = xc
            new_width = height
            new_height = width
        elif angle == 180:
            new_xc = 1.0 - xc
            new_yc = 1.0 - yc
            new_width = width
            new_height = height
        elif angle == 270:
            new_xc = yc
            new_yc = 1.0 - xc
            new_width = height
            new_height = width
        else:
            raise ValueError(f"Nicht unterstuetzter Winkel: {angle}")

        rotated.append(
            (
                cls,
                min(max(new_xc, 0.0), 1.0),
                min(max(new_yc, 0.0), 1.0),
                min(max(new_width, 0.0), 1.0),
                min(max(new_height, 0.0), 1.0),
            )
        )

    return rotated


def rotate_image(image, angle: int):
    """Rotiert ein Bild im Uhrzeigersinn um 90/180/270 Grad."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Nicht unterstuetzter Winkel: {angle}")


def iter_split_images(images_dir: Path) -> list:
    """Liefert nur Originalbilder ohne bereits erzeugte Rotationssuffixe."""
    image_paths = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(images_dir.glob(pattern))

    originals = []
    for image_path in sorted(image_paths):
        if image_path.stem.endswith(ROTATION_SUFFIXES):
            continue
        originals.append(image_path)

    return originals


def augment_split(split: str, overwrite: bool) -> tuple[int, int]:
    """Erzeugt Rotationsvarianten fuer einen Split."""
    images_dir = YOLO_DIR / split / "images"
    labels_dir = YOLO_DIR / split / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"{split}: Ordner nicht gefunden, uebersprungen.")
        return 0, 0

    created = 0
    skipped = 0

    originals = iter_split_images(images_dir)
    total = len(originals)

    for idx, image_path in enumerate(originals, 1):
        print(f"\r  {split}: [{idx}/{total}] {image_path.name}", end="", flush=True)

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"\r  {split}: Label fehlt fuer {image_path.name}, uebersprungen.")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"\r  {split}: Bild konnte nicht geladen werden: {image_path.name}")
            skipped += 1
            continue

        boxes = load_yolo_labels(label_path)

        h, w = image.shape[:2]
        is_portrait = h > w

        # Querformat: 90 + 270. Hochformat: nur 270 (90 waere effektiv 180 der Szene).
        angles = [270] if is_portrait else [90, 270]

        for angle in angles:
            suffix = f"_rot{angle}"
            out_image = images_dir / f"{image_path.stem}{suffix}{image_path.suffix}"
            out_label = labels_dir / f"{image_path.stem}{suffix}.txt"

            if not overwrite and (out_image.exists() or out_label.exists()):
                skipped += 1
                continue

            rotated_image = rotate_image(image, angle)
            rotated_boxes = rotate_boxes(boxes, angle)

            cv2.imwrite(str(out_image), rotated_image)
            save_yolo_labels(out_label, rotated_boxes)
            created += 1

    print()  # Zeilenumbruch nach Progress
    return created, skipped


def main():
    parser = argparse.ArgumentParser(description="YOLO 90/180/270 Grad Rotations-Augmentierung")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="Welche Splits erweitert werden sollen (Standard: train)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Bestehende *_rot90/*_rot180/*_rot270 Dateien ueberschreiben",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO Rotations-Augmentierung")
    print("=" * 60)

    total_created = 0
    total_skipped = 0

    for split in args.splits:
        created, skipped = augment_split(split, args.overwrite)
        total_created += created
        total_skipped += skipped
        print(f"{split}: {created} Dateien erzeugt, {skipped} uebersprungen")

    print("\nFertig.")
    print(f"Erzeugt: {total_created}")
    print(f"Uebersprungen: {total_skipped}")
    print("\nNaechster Schritt: train_yolo.py ohne starke Rotations-Augmentierung starten")


if __name__ == "__main__":
    main()