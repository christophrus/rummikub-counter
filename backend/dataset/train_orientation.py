"""
Trainiert ein kleines CNN zur Erkennung der Bildorientierung (0/90/180/270 Grad).

Das Trainingsset wird automatisch aus den vorhandenen YOLO-Bildern erzeugt:
Jedes Bild wird in allen 4 Orientierungen gespeichert.

Nutzung:
    python train_orientation.py
    python train_orientation.py --epochs 30 --imgsz 224

Das Modell wird nach models/orientation_cnn.pth gespeichert.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

SCRIPT_DIR = Path(__file__).parent
YOLO_DIR = SCRIPT_DIR.parent / "yolo_dataset"
MODEL_OUT = SCRIPT_DIR.parent / "models" / "orientation_cnn.pth"

# Klassen: 0=aufrecht, 1=90° CW, 2=180°, 3=270° CW
ORIENTATIONS = [0, 90, 180, 270]
ORI_TO_IDX = {0: 0, 90: 1, 180: 2, 270: 3}


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotiert ein Bild um exakt 0/90/180/270 Grad im Uhrzeigersinn."""
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Ungueltiger Winkel: {angle}")


def collect_images() -> list[Path]:
    """Sammelt alle Originalbilder aus train/ und val/ (ohne _rot Suffixe)."""
    images = []
    rot_suffixes = ("_rot90", "_rot180", "_rot270")
    for split in ["train", "val"]:
        img_dir = YOLO_DIR / split / "images"
        if not img_dir.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in img_dir.glob(ext):
                if not p.stem.endswith(rot_suffixes):
                    images.append(p)
    return sorted(images)


class OrientationDataset(Dataset):
    """Dataset das jedes Bild in allen 4 Orientierungen liefert."""

    def __init__(self, image_paths: list[Path], imgsz: int, augment: bool = False):
        self.samples = []  # (path, angle)
        for p in image_paths:
            for angle in ORIENTATIONS:
                self.samples.append((p, angle))
        self.imgsz = imgsz
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, angle = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Fallback: schwarzes Bild
            img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)

        img = rotate_image(img, angle)

        # Leichte Augmentierung (nur Training)
        if self.augment:
            # Helligkeitsvariation
            if random.random() < 0.5:
                factor = random.uniform(0.7, 1.3)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
            # Leichter Blur
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        label = ORI_TO_IDX[angle]
        return tensor, label


def main():
    parser = argparse.ArgumentParser(description="Orientierungs-CNN Training")
    parser.add_argument("--epochs", type=int, default=20, help="Anzahl Epochen (Standard: 20)")
    parser.add_argument("--imgsz", type=int, default=224, help="Bildgroesse (Standard: 224)")
    parser.add_argument("--batch", type=int, default=32, help="Batch Size (Standard: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate (Standard: 0.001)")
    parser.add_argument("--max-images", type=int, default=100, help="Max. Anzahl Bilder (Standard: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Orientierungs-CNN Training")
    print("=" * 60)

    # Bilder sammeln
    all_images = collect_images()
    if len(all_images) < 10:
        print(f"Zu wenige Bilder ({len(all_images)}). Mindestens 10 benoetigt.")
        return

    # Auf max. Anzahl begrenzen (zufaellige Auswahl)
    random.seed(42)
    random.shuffle(all_images)
    if len(all_images) > args.max_images:
        print(f"Verwende {args.max_images} von {len(all_images)} Bildern.")
        all_images = all_images[:args.max_images]
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"\nBilder: {len(all_images)} (x4 Orientierungen = {len(all_images) * 4} Samples)")
    print(f"Train: {len(train_images)} Bilder ({len(train_images) * 4} Samples)")
    print(f"Val:   {len(val_images)} Bilder ({len(val_images) * 4} Samples)")

    train_dataset = OrientationDataset(train_images, args.imgsz, augment=True)
    val_dataset = OrientationDataset(val_images, args.imgsz, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    # Modell: ResNet-18 (klein und schnell)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 Klassen
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training
    best_val_acc = 0.0
    print(f"\nStarte Training: {args.epochs} Epochen")
    print("-" * 60)

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        scheduler.step(val_loss / val_total)

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"Train Loss: {train_loss/train_total:.4f}  Acc: {train_acc:.1%}  |  "
              f"Val Loss: {val_loss/val_total:.4f}  Acc: {val_acc:.1%}")

        # Bestes Modell speichern
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "imgsz": args.imgsz,
                "orientations": ORIENTATIONS,
            }, str(MODEL_OUT))
            print(f"         -> Neues bestes Modell gespeichert ({val_acc:.1%})")

    print(f"\nTraining abgeschlossen!")
    print(f"Beste Val-Accuracy: {best_val_acc:.1%}")
    print(f"Modell: {MODEL_OUT}")


if __name__ == "__main__":
    main()
