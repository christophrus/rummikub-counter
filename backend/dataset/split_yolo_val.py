"""Verschiebt zufällig ~20% der YOLO-Trainingsdaten nach val/."""

import random
import shutil
from pathlib import Path

YOLO_DIR = Path(__file__).parent.parent / "yolo_dataset"

train_images = YOLO_DIR / "train" / "images"
train_labels = YOLO_DIR / "train" / "labels"
val_images = YOLO_DIR / "val" / "images"
val_labels = YOLO_DIR / "val" / "labels"

val_images.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

images = sorted(train_images.glob("*.jpg")) + sorted(train_images.glob("*.png"))
if not images:
    print("Keine Trainingsbilder gefunden.")
    exit()

random.seed(42)
val_count = max(1, len(images) // 5)  # ~20%
val_selection = random.sample(images, val_count)

print(f"{len(images)} Trainingsbilder, verschiebe {val_count} nach val/")

for img_path in val_selection:
    label_path = train_labels / (img_path.stem + ".txt")

    shutil.move(str(img_path), str(val_images / img_path.name))
    if label_path.exists():
        shutil.move(str(label_path), str(val_labels / label_path.name))

    print(f"  → {img_path.name}")

print(f"\nFertig! Train: {len(images) - val_count}, Val: {val_count}")
