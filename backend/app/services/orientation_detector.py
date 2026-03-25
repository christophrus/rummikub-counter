"""Service zur Erkennung der Bildorientierung (0/90/180/270 Grad).

Nutzt ein vortrainiertes ResNet-18 das auf Rummikub-Spielfotos trainiert wurde.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "orientation_cnn.pth"

ORIENTATIONS = [0, 90, 180, 270]

# Gegendrehung um das Bild aufzurichten
CORRECTION = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}

_model = None
_device = None
_imgsz = 224


def load_orientation_model() -> bool:
    """Laedt das Orientierungsmodell. Gibt True zurueck bei Erfolg."""
    global _model, _device, _imgsz

    if not MODEL_PATH.exists():
        print(f"[Orientierung] Modell nicht gefunden: {MODEL_PATH}")
        return False

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(str(MODEL_PATH), map_location=_device, weights_only=True)
    _imgsz = checkpoint.get("imgsz", 224)

    _model = models.resnet18(weights=None)
    _model.fc = nn.Linear(_model.fc.in_features, 4)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device)
    _model.eval()

    val_acc = checkpoint.get("val_acc", 0)
    print(f"[Orientierung] Modell geladen (Val-Acc: {val_acc:.1%}, Device: {_device})")
    return True


def detect_orientation(image: np.ndarray) -> int:
    """Erkennt die Orientierung eines Bildes.

    Args:
        image: BGR OpenCV Bild

    Returns:
        Erkannte Orientierung in Grad (0, 90, 180, 270).
        0 bedeutet das Bild ist aufrecht.
    """
    if _model is None:
        return 0  # Fallback: kein Modell -> aufrecht annehmen

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((_imgsz, _imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _model(tensor)
        _, predicted = outputs.max(1)

    angle = ORIENTATIONS[predicted.item()]
    return angle


def correct_orientation(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Erkennt und korrigiert die Orientierung eines Bildes.

    Args:
        image: BGR OpenCV Bild

    Returns:
        Tuple aus (korrigiertes_bild, erkannter_winkel).
        Das Bild wird so gedreht, dass es aufrecht ist.
    """
    angle = detect_orientation(image)

    rotation = CORRECTION[angle]
    if rotation is not None:
        image = cv2.rotate(image, rotation)

    return image, angle
