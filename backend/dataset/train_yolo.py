"""
YOLOv8 Training: Trainiert ein Object-Detection-Modell für Rummikub.

Nutzung:
    python train_yolo.py
    python train_yolo.py --epochs 100 --model yolov8n.pt --imgsz 640

Das Modell erkennt und lokalisiert Rummikub-Steine in einem Schritt.
Ergebnisse werden unter runs/detect/rummikub/ gespeichert.
"""

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
YOLO_DIR = SCRIPT_DIR.parent / "yolo_dataset"
DATA_YAML = YOLO_DIR / "data.yaml"


def check_dataset():
    """Prüft ob genügend Trainingsdaten vorhanden sind."""
    train_images = list((YOLO_DIR / "train" / "images").glob("*"))
    train_labels = list((YOLO_DIR / "train" / "labels").glob("*.txt"))
    val_images = list((YOLO_DIR / "val" / "images").glob("*"))
    val_labels = list((YOLO_DIR / "val" / "labels").glob("*.txt"))

    print(f"Train: {len(train_images)} Bilder, {len(train_labels)} Labels")
    print(f"Val:   {len(val_images)} Bilder, {len(val_labels)} Labels")

    if len(train_images) < 10:
        print("\n⚠️  Zu wenige Trainingsbilder! Mindestens 50-100 empfohlen.")
        print("   Zuerst labeln: python label_for_yolo.py --images pfad/zu/bildern/")
        return False

    if len(val_images) < 5:
        print("\n⚠️  Zu wenige Validierungsbilder! Mindestens 10-20 empfohlen.")
        print("   Labeln mit: python label_for_yolo.py --images pfad/ --split val")
        return False

    if len(train_labels) != len(train_images):
        print(f"\n⚠️  Anzahl Bilder ({len(train_images)}) und Labels ({len(train_labels)}) stimmen nicht überein!")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Rummikub YOLOv8 Training")
    parser.add_argument("--epochs", type=int, default=100, help="Anzahl Epochen (Standard: 100)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Basis-Modell (Standard: yolov8n.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Bildgröße (Standard: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch Size (Standard: 16)")
    parser.add_argument("--device", type=str, default=None, help="Device: 0 für GPU, cpu für CPU")
    args = parser.parse_args()

    print("=" * 60)
    print("  Rummikub YOLOv8 Training")
    print("=" * 60)

    if not DATA_YAML.exists():
        print(f"Fehler: {DATA_YAML} nicht gefunden!")
        return

    if not check_dataset():
        return

    # Ultralytics importieren (erst hier, damit Fehler oben schnell kommen)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nFehler: ultralytics nicht installiert!")
        print("Installieren mit: pip install ultralytics")
        return

    # Device bestimmen
    import torch
    if args.device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\nDevice: {'GPU' if device == '0' else 'CPU'}")
    if device == "0":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Modell laden (vortrainiert auf COCO → Transfer Learning)
    print(f"\nLade Basis-Modell: {args.model}")
    model = YOLO(args.model)

    # Training starten
    print(f"\nStarte Training: {args.epochs} Epochen, Bildgröße {args.imgsz}")
    print("-" * 60)

    results = model.train(
        data=str(DATA_YAML.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        project=str(SCRIPT_DIR / "runs" / "detect"),
        name="rummikub",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    # Bestes Modell kopieren
    best_model = SCRIPT_DIR / "runs" / "detect" / "rummikub" / "weights" / "best.pt"
    target = SCRIPT_DIR.parent / "models" / "rummikub_yolo.pt"

    if best_model.exists():
        import shutil
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_model), str(target))
        print(f"\n✅ Bestes Modell kopiert nach: {target}")
    else:
        print(f"\n⚠️  best.pt nicht gefunden unter {best_model}")

    # Evaluation
    print("\n" + "=" * 60)
    print("  Evaluation auf Validierungsdaten")
    print("=" * 60)

    model_best = YOLO(str(best_model)) if best_model.exists() else model
    metrics = model_best.val(data=str(DATA_YAML.resolve()), device=device)

    print(f"\nmAP50:    {metrics.box.map50:.1%}")
    print(f"mAP50-95: {metrics.box.map:.1%}")

    print(f"\nTraining abgeschlossen!")
    print(f"Ergebnisse: {SCRIPT_DIR / 'runs' / 'detect' / 'rummikub'}")
    print(f"Nächster Schritt: YOLO in die App integrieren")


if __name__ == "__main__":
    main()
