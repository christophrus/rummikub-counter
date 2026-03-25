"""
Splittet YOLO-Trainingsdaten in train/val/test.

Nutzung:
    1. Alle Bilder+Labels in yolo_dataset/train/ legen
    2. python split_yolo_val.py
    3. Ergebnis: train (~70%), val (~15%), test (~15%)

Optional:
    python split_yolo_val.py --val 20 --test 10
"""

import argparse
import random
import shutil
from pathlib import Path

YOLO_DIR = Path(__file__).parent.parent / "yolo_dataset"


def main():
    parser = argparse.ArgumentParser(description="YOLO Train/Val/Test Split")
    parser.add_argument("--val", type=int, default=20, help="Val-Anteil in %% (Standard: 20)")
    parser.add_argument("--test", type=int, default=0, help="Test-Anteil in %% (Standard: 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed (Standard: 42)")
    parser.add_argument("--reverse", action="store_true", help="Val/Test zurueck nach Train verschieben und beenden")
    args = parser.parse_args()

    train_images = YOLO_DIR / "train" / "images"
    train_labels = YOLO_DIR / "train" / "labels"
    val_images = YOLO_DIR / "val" / "images"
    val_labels = YOLO_DIR / "val" / "labels"
    test_images = YOLO_DIR / "test" / "images"
    test_labels = YOLO_DIR / "test" / "labels"

    # Bestehende val/test-Daten zurück nach train verschieben
    for src_imgs, src_lbls in [(val_images, val_labels), (test_images, test_labels)]:
        if src_imgs.exists():
            for img in src_imgs.glob("*"):
                shutil.move(str(img), str(train_images / img.name))
            for lbl in src_lbls.glob("*"):
                shutil.move(str(lbl), str(train_labels / lbl.name))

    if args.reverse:
        total = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
        print(f"Alles zurueck nach train/ verschoben. {total} Bilder in train/")
        return

    # Ordner erstellen
    for d in [val_images, val_labels, test_images, test_labels]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(train_images.glob("*.jpg")) + sorted(train_images.glob("*.png"))
    if not images:
        print("Keine Trainingsbilder gefunden.")
        return

    random.seed(args.seed)
    random.shuffle(images)

    val_count = max(1, len(images) * args.val // 100) if args.val > 0 else 0
    test_count = max(1, len(images) * args.test // 100) if args.test > 0 else 0

    val_selection = images[:val_count]
    test_selection = images[val_count:val_count + test_count]

    print(f"{len(images)} Bilder gesamt")
    print(f"  → Val:   {val_count} ({args.val}%)")
    print(f"  → Test:  {test_count} ({args.test}%)")
    print(f"  → Train: {len(images) - val_count - test_count}")
    print()

    def move_files(selection, dst_imgs, dst_lbls, label):
        for img_path in selection:
            label_path = train_labels / (img_path.stem + ".txt")
            shutil.move(str(img_path), str(dst_imgs / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(dst_lbls / label_path.name))
        print(f"{label}: {len(selection)} Bilder verschoben")

    move_files(val_selection, val_images, val_labels, "Val")
    move_files(test_selection, test_images, test_labels, "Test")

    remaining = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
    print(f"\nFertig! Train: {remaining}, Val: {val_count}, Test: {test_count}")


if __name__ == "__main__":
    main()
