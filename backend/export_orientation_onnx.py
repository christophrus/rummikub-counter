"""Convert orientation_cnn.pth to ONNX format."""
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

MODEL_DIR = Path(__file__).parent / "models"
PTH_PATH = MODEL_DIR / "orientation_cnn.pth"
ONNX_PATH = MODEL_DIR / "orientation_cnn.onnx"

# Load checkpoint
checkpoint = torch.load(str(PTH_PATH), map_location="cpu", weights_only=True)
imgsz = checkpoint.get("imgsz", 224)

# Build model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_acc = checkpoint.get("val_acc", 0)
print(f"Model loaded (Val-Acc: {val_acc:.1%}, imgsz: {imgsz})")

# Export to ONNX
dummy_input = torch.randn(1, 3, imgsz, imgsz)
torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_PATH),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
    dynamo=False,
)
print(f"Exported to {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
