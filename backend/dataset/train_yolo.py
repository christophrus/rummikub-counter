"""
YOLO26 Training: Trainiert ein Object-Detection-Modell für Rummikub.

Nutzung:
    python train_yolo.py
    python train_yolo.py --epochs 100 --model yolo26n.pt --imgsz 1280
    python train_yolo.py --hard-weight 3   # Hard Examples 3x duplizieren

Das Modell erkennt und lokalisiert Rummikub-Steine in einem Schritt.
Ergebnisse werden unter runs/detect/rummikub/ gespeichert.
"""

import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
YOLO_DIR = SCRIPT_DIR.parent / "yolo_dataset"
DATA_YAML = YOLO_DIR / "data.yaml"
HARD_EXAMPLES_YOLO_PATH = SCRIPT_DIR / "hard_examples_yolo.txt"


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
        label_stems = {p.stem for p in train_labels}
        for img in sorted(train_images):
            if img.stem not in label_stems:
                print(f"   Kein Label: {img.name}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Rummikub YOLO26 Training")
    parser.add_argument("--epochs", type=int, default=200, help="Anzahl Epochen (Standard: 200)")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Basis-Modell (Standard: yolo26n.pt)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Bildgröße (Standard: 1280)")
    parser.add_argument("--batch", type=int, default=16, help="Batch Size (Standard: 16)")
    parser.add_argument("--device", type=str, default=None, help="Device: 0 für GPU, cpu für CPU")
    parser.add_argument("--hard-weight", type=int, default=1,
                        help="Hard Examples N-fach duplizieren (Standard: 1 = keine Duplikation, z.B. 3 = 3 Kopien)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Rummikub YOLO26 Training")
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

    # Hard Examples duplizieren (Oversampling)
    hard_copies = []
    if args.hard_weight > 1:
        hard_copies = _create_hard_copies(args.hard_weight)

    # Training starten
    print(f"\nStarte Training: {args.epochs} Epochen, Bildgröße {args.imgsz}")
    print("-" * 60)

    results = model.train(
        data=str(DATA_YAML.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=4,
        project=str(SCRIPT_DIR / "runs" / "detect"),
        name="rummikub",
        exist_ok=True,
        patience=50,
        save=True,
        plots=True,
        verbose=True,
        # Keine Rotation: Orientierungs-CNN korrigiert vorher
        degrees=0.0,
        flipud=0.0,
        # Farb-Augmentierung: Modell ignoriert Steinfarben
        hsv_h=0.5,   # Volle Hue-Rotation (alle Farben ↔ alle Farben)
        hsv_s=0.9,   # Starke Sättigungsvariation (bunt ↔ grau)
        hsv_v=0.5,   # Helligkeitsvariation
    )

    # Hard-Example-Kopien aufräumen
    if hard_copies:
        _cleanup_hard_copies(hard_copies)

    # Bestes Modell kopieren
    best_model = SCRIPT_DIR / "runs" / "detect" / "rummikub" / "weights" / "best.pt"
    target = SCRIPT_DIR.parent / "models" / "rummikub_yolo.pt"

    if best_model.exists():
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

    # Test-Evaluation (unabhängiger Datensatz)
    test_images = list((YOLO_DIR / "test" / "images").glob("*"))
    if len(test_images) >= 1:
        print("\n" + "=" * 60)
        print("  Evaluation auf Testdaten (unabhängig)")
        print("=" * 60)

        test_metrics = model_best.val(
            data=str(DATA_YAML.resolve()), split="test", device=device
        )

        print(f"\nTest mAP50:    {test_metrics.box.map50:.1%}")
        print(f"Test mAP50-95: {test_metrics.box.map:.1%}")

        if test_metrics.box.map50 < metrics.box.map50 - 0.1:
            print("\n⚠️  Test-mAP50 deutlich schlechter als Val → mögliches Overfitting!")
    else:
        print("\nℹ️  Keine Testbilder vorhanden, Test-Evaluation übersprungen.")

    print(f"\nTraining abgeschlossen!")
    print(f"Ergebnisse: {SCRIPT_DIR / 'runs' / 'detect' / 'rummikub'}")
    print(f"Nächster Schritt: YOLO in die App integrieren")


def _create_hard_copies(weight: int) -> list[Path]:
    """Dupliziert Hard-Example-Bilder und Labels für Oversampling."""
    if not HARD_EXAMPLES_YOLO_PATH.exists():
        print(f"\nKeine Hard Examples: {HARD_EXAMPLES_YOLO_PATH.name} nicht gefunden.")
        return []

    hard_paths = []
    for line in HARD_EXAMPLES_YOLO_PATH.read_text(encoding="utf-8").strip().splitlines():
        p = Path(line.strip())
        if p.exists():
            hard_paths.append(p)

    if not hard_paths:
        print(f"\nKeine gültigen Hard Examples in {HARD_EXAMPLES_YOLO_PATH.name}.")
        return []

    copies = []
    for img_path in hard_paths:
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        for i in range(1, weight):
            img_copy = img_path.parent / f"{img_path.stem}_hard{i}{img_path.suffix}"
            lbl_copy = label_path.parent / f"{label_path.stem}_hard{i}.txt"
            shutil.copy2(str(img_path), str(img_copy))
            shutil.copy2(str(label_path), str(lbl_copy))
            copies.extend([img_copy, lbl_copy])

    print(f"\nHard Examples: {len(hard_paths)} Bilder je {weight-1}x dupliziert ({len(copies)//2} Kopien erstellt)")
    return copies


def _cleanup_hard_copies(copies: list[Path]):
    """Entfernt die temporären Hard-Example-Kopien nach dem Training."""
    removed = 0
    for p in copies:
        if p.exists():
            p.unlink()
            removed += 1
    print(f"\nHard-Example-Kopien aufgeräumt ({removed} Dateien entfernt)")


if __name__ == "__main__":
    main()
