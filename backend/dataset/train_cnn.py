"""
CNN Training: Trainiert ein Custom CNN zur Rummikub-Stein-Erkennung.

Nutzung:
    python train_cnn.py
    python train_cnn.py --epochs 50 --batch-size 32 --lr 0.001

Das Modell erkennt 14 Klassen: Zahlen 1-13 + Joker.
Nach dem Training wird das beste Modell als 'rummikub_cnn.pth' gespeichert.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SCRIPT_DIR = Path(__file__).parent
TRAIN_DIR = SCRIPT_DIR / "train"
VAL_DIR = SCRIPT_DIR / "val"
TEST_DIR = SCRIPT_DIR / "test"
MODEL_DIR = SCRIPT_DIR.parent / "models"

# 14 Klassen: 1-13 + Joker
NUM_CLASSES = 14
IMG_WIDTH = 64
IMG_HEIGHT = 96


class RummikubCNN(nn.Module):
    """
    Custom CNN für Rummikub-Stein-Erkennung.

    Architektur:
    - 4 Convolutional Blocks (Conv2D → BatchNorm → ReLU → MaxPool)
    - Dropout zur Regularisierung
    - 2 Fully Connected Layers
    - Softmax Output (14 Klassen)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 32 Kanäle
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x96 → 32x48

            # Block 2: 32 → 64 Kanäle
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x48 → 16x24

            # Block 3: 64 → 128 Kanäle
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x24 → 8x12

            # Block 4: 128 → 256 Kanäle
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x12 → 4x6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def get_data_loaders(batch_size: int):
    """Erstellt Train/Val/Test DataLoaders mit Transforms."""

    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=eval_transform)

    # Klassen-Mapping ausgeben
    print(f"\nKlassen-Mapping (Index → Label):")
    for cls_name, idx in sorted(train_dataset.class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx:>2} → {cls_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    test_loader = None
    if TEST_DIR.exists():
        test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Trainiert eine Epoche."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluiert auf Validation/Test Set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Rummikub CNN Training")
    parser.add_argument("--epochs", type=int, default=30, help="Anzahl Epochen (Standard: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size (Standard: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate (Standard: 0.001)")
    parser.add_argument("--patience", type=int, default=7, help="Early Stopping Patience (Standard: 7)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Rummikub CNN Training")
    print("=" * 60)

    # Prüfen ob Daten vorhanden
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("Fehler: train/ und val/ Ordner nicht gefunden.")
        print("Zuerst split_dataset.py ausführen!")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # DataLoaders
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(args.batch_size)
    print(f"\nTrain: {len(train_loader.dataset)} Bilder")
    print(f"Val:   {len(val_loader.dataset)} Bilder")
    if test_loader:
        print(f"Test:  {len(test_loader.dataset)} Bilder")

    # Modell
    model = RummikubCNN(NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModell-Parameter: {total_params:,}")

    # Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=3)

    # Training Loop
    best_val_acc = 0.0
    patience_counter = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_DIR / "rummikub_cnn.pth"

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 65)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>8.4f} | {val_acc:>6.1%} | {current_lr:>8.6f}")

        # Bestes Modell speichern
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "val_acc": val_acc,
                "epoch": epoch,
                "img_size": (IMG_WIDTH, IMG_HEIGHT),
            }, str(best_model_path))
            print(f"        → Bestes Modell gespeichert! (Val Acc: {val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly Stopping nach {epoch} Epochen (keine Verbesserung seit {args.patience} Epochen)")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining abgeschlossen in {elapsed:.1f}s")
    print(f"Bestes Modell: Val Acc = {best_val_acc:.1%}")

    # Test-Evaluation
    if test_loader:
        print(f"\n{'='*40}")
        print("Test-Evaluation:")
        checkpoint = torch.load(str(best_model_path), weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Test Accuracy: {test_acc:.1%}")
        print(f"  Test Loss:     {test_loss:.4f}")

        # Confusion Details pro Klasse
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        model.eval()
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)

                for label, pred in zip(labels, predicted):
                    cls = idx_to_class[label.item()]
                    class_total[cls] = class_total.get(cls, 0) + 1
                    if label == pred:
                        class_correct[cls] = class_correct.get(cls, 0) + 1

        print(f"\nPro Klasse:")
        for cls in sorted(class_total.keys()):
            correct = class_correct.get(cls, 0)
            total = class_total[cls]
            acc = correct / total if total > 0 else 0
            print(f"  {cls:>5}: {acc:.0%} ({correct}/{total})")

    print(f"\nModell gespeichert: {best_model_path}")
    print("Nächster Schritt: Das Modell in die App integrieren!")


if __name__ == "__main__":
    main()
