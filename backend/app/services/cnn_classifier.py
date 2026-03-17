"""
CNN-Klassifikator: Erkennt Rummikub-Steinzahlen mit dem eigenen CNN-Modell.

Ersetzt EasyOCR durch ein selbst trainiertes CNN.
Klassen: 1-13 + Joker (14 Klassen).
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "rummikub_cnn.pth"

NUM_CLASSES = 14
IMG_WIDTH = 64
IMG_HEIGHT = 96


class RummikubCNN(nn.Module):
    """Identische Architektur wie beim Training."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Globaler State
_model: RummikubCNN | None = None
_idx_to_class: dict[int, str] = {}
_device: torch.device = torch.device("cpu")

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model() -> None:
    """Lädt das trainierte CNN-Modell."""
    global _model, _idx_to_class, _device

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"CNN Device: {_device}")

    checkpoint = torch.load(str(MODEL_PATH), map_location=_device, weights_only=False)

    _model = RummikubCNN(NUM_CLASSES).to(_device)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    _idx_to_class = {v: k for k, v in class_to_idx.items()}

    val_acc = checkpoint.get("val_acc", 0)
    epoch = checkpoint.get("epoch", "?")
    logger.info(f"CNN geladen (Epoche {epoch}, Val Acc: {val_acc:.1%})")


def classify_tile(tile_image: np.ndarray) -> dict:
    """
    Klassifiziert ein einzelnes Tile-Bild.

    Args:
        tile_image: BGR-Bild (OpenCV-Format) des ausgeschnittenen Steins.

    Returns:
        {"number": int|None, "confidence": float, "is_joker": bool}
    """
    if _model is None:
        raise RuntimeError("CNN-Modell nicht geladen. Zuerst load_model() aufrufen.")

    # BGR → RGB für torchvision
    rgb_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)

    # Transform und Batch-Dimension hinzufügen
    tensor = _transform(rgb_image).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    confidence_val = confidence.item()
    class_name = _idx_to_class.get(predicted_idx.item(), "unknown")

    if class_name == "joker":
        return {"number": None, "confidence": confidence_val, "is_joker": True}

    try:
        number = int(class_name)
    except ValueError:
        return {"number": None, "confidence": confidence_val, "is_joker": False}

    return {"number": number, "confidence": confidence_val, "is_joker": False}
