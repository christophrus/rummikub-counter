"""
Mislabel-Finder: Durchsucht Trainingsdaten nach falsch gelabelten Steinen.

Nutzt das trainierte CNN-Modell, um alle Bilder in train/val/test zu prüfen.
Bilder, bei denen das Modell mit hoher Konfidenz eine andere Klasse vorhersagt
als der Ordnername vorgibt, werden als verdächtig markiert und zur manuellen
Korrektur angezeigt.

Nutzung:
    python find_mislabeled.py                    # Alle Splits prüfen
    python find_mislabeled.py --split train      # Nur Trainingsbilder prüfen
    python find_mislabeled.py --threshold 0.5    # Niedrigere Schwelle = mehr Ergebnisse
    python find_mislabeled.py --yolo             # YOLO-Labels prüfen

Steuerung (CNN-Modus):
    1-9         → Bild in Klasse 1-9 verschieben
    F1=10, F2=11, F3=12, F4=13
    j           → Bild als Joker markieren
    k / Enter   → Label ist korrekt, weiter (wird als Hard Example gespeichert)
    d / Entf    → Bild löschen
    q           → Beenden

Steuerung (YOLO-Modus):
    e           → Bild im Label-Editor öffnen (label_for_yolo.py --edit-mode)
    k / Enter   → Labels stimmen, weiter
    q           → Beenden
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR.parent / "models" / "rummikub_cnn.pth"
YOLO_MODEL_PATH = SCRIPT_DIR.parent / "models" / "rummikub_yolo.pt"
YOLO_DIR = SCRIPT_DIR.parent / "yolo_dataset"
HARD_EXAMPLES_PATH = SCRIPT_DIR / "hard_examples.txt"
HARD_EXAMPLES_YOLO_PATH = SCRIPT_DIR / "hard_examples_yolo.txt"

NUM_CLASSES = 14
IMG_WIDTH = 64
IMG_HEIGHT = 96

CLASS_NAMES = {i: str(i + 1) for i in range(13)}
CLASS_NAMES[13] = "joker"

LABEL_TO_IDX = {str(i + 1): i for i in range(13)}
LABEL_TO_IDX["joker"] = 13

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
    (200, 200, 0), (128, 128, 255),
]

# --- CNN Model (identisch zu train_cnn.py) ---

class RummikubCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 4 * 6, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_cnn_model():
    """Lädt das trainierte CNN-Modell."""
    if not MODEL_PATH.exists():
        print(f"CNN-Modell nicht gefunden: {MODEL_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)

    model = RummikubCNN(NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"CNN geladen auf {device} (Val-Acc: {checkpoint.get('val_acc', '?'):.1%})")
    return model, idx_to_class, class_to_idx, transform, device


def predict_image(model, img_bgr, transform, device):
    """Gibt (vorhergesagte Klasse, Konfidenz, alle Wahrscheinlichkeiten) zurück."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred_idx = probs.max(0)

    return pred_idx.item(), conf.item(), probs.cpu().numpy()


# --- CNN Mislabel Scan ---

def scan_cnn_splits(model, idx_to_class, class_to_idx, transform, device, splits, threshold):
    """Scannt CNN-Klassifikationsdaten und findet verdächtige Labels."""
    suspects = []

    for split in splits:
        split_dir = SCRIPT_DIR / split
        if not split_dir.exists():
            print(f"  {split}/ nicht gefunden, überspringe...")
            continue

        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        total = 0
        flagged = 0

        # Alle Bilder vorab zählen für Fortschrittsanzeige
        all_images = []
        for class_dir in class_dirs:
            if class_dir.name not in class_to_idx:
                continue
            imgs = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
            all_images.extend((class_dir, img) for img in imgs)
        total_images = len(all_images)

        for i, (class_dir, img_path) in enumerate(all_images):
            folder_label = class_dir.name
            true_idx = class_to_idx[folder_label]

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            total += 1
            pred_idx, conf, probs = predict_image(model, img, transform, device)

            if pred_idx != true_idx and conf >= threshold:
                pred_label = idx_to_class.get(pred_idx, "?")
                suspects.append({
                    "path": img_path,
                    "split": split,
                    "folder_label": folder_label,
                    "pred_label": pred_label,
                    "pred_idx": pred_idx,
                    "confidence": conf,
                    "probs": probs,
                })
                flagged += 1

            print(f"\r  {split}: {i+1}/{total_images} Bilder ({flagged} verdächtig)", end="", flush=True)

        print(f"\r  {split}: {total} Bilder geprüft, {flagged} verdächtig{' ' * 20}")

    # Nach Konfidenz sortieren (höchste zuerst = wahrscheinlichste Fehler)
    suspects.sort(key=lambda x: x["confidence"], reverse=True)
    return suspects


def review_cnn_suspects(suspects, idx_to_class):
    """Interaktive Überprüfung der verdächtigen CNN-Labels."""
    if not suspects:
        print("\nKeine verdächtigen Labels gefunden!")
        return

    print(f"\n{'='*60}")
    print(f"{len(suspects)} verdächtige Bilder gefunden.")
    print(f"Steuerung: 1-9/F1-F4/j=verschieben, k/Enter=korrekt, d=löschen, q=beenden")
    print(f"{'='*60}\n")

    moved = 0
    deleted = 0
    kept = 0
    idx = 0
    hard_examples_file = open(str(HARD_EXAMPLES_PATH), "a", encoding="utf-8")

    while idx < len(suspects):
        s = suspects[idx]
        img = cv2.imread(str(s["path"]))
        if img is None:
            idx += 1
            continue

        # Info-Overlay erstellen
        h, w = img.shape[:2]
        scale = max(1.0, 300 / min(h, w))
        display = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        dh, dw = display.shape[:2]

        # Info-Panel oben
        panel_h = 120
        canvas = np.zeros((dh + panel_h, max(dw, 500), 3), dtype=np.uint8)
        canvas[panel_h:panel_h + dh, :dw] = display

        # Text
        cv2.putText(canvas, f"[{idx+1}/{len(suspects)}] {s['path'].name}",
                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"Ordner-Label: {s['folder_label']}",
                     (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(canvas, f"CNN sagt: {s['pred_label']} ({s['confidence']:.0%})",
                     (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Top-3 Vorhersagen
        top3_idx = np.argsort(s["probs"])[::-1][:3]
        top3_str = ", ".join(f"{idx_to_class.get(i, '?')}:{s['probs'][i]:.0%}" for i in top3_idx)
        cv2.putText(canvas, f"Top-3: {top3_str}",
                     (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.putText(canvas, "1-9/F1-F4/j=verschieben  k/Enter=ok  d=loeschen  q=quit",
                     (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        cv2.namedWindow("Mislabel Review", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mislabel Review", canvas.shape[1], canvas.shape[0])
        cv2.imshow("Mislabel Review", canvas)

        key = cv2.waitKey(0) & 0xFFFF

        if key == ord('q'):
            break
        elif key in (ord('k'), 13):  # k oder Enter = korrekt
            hard_examples_file.write(str(s["path"]) + "\n")
            kept += 1
            idx += 1
        elif key in (ord('d'), 65535, 3014656):  # d oder Entf = löschen
            s["path"].unlink()
            print(f"  GELÖSCHT: {s['path'].name}")
            deleted += 1
            idx += 1
        elif key == ord('j'):
            _move_image(s["path"], "joker")
            moved += 1
            idx += 1
        else:
            # Zifferntasten 1-9
            new_label = None
            if ord('1') <= key <= ord('9'):
                new_label = str(key - ord('0'))
            elif key == 65470:   # F1 = 10
                new_label = "10"
            elif key == 65471:   # F2 = 11
                new_label = "11"
            elif key == 65472:   # F3 = 12
                new_label = "12"
            elif key == 65473:   # F4 = 13
                new_label = "13"

            if new_label:
                _move_image(s["path"], new_label)
                moved += 1
                idx += 1

    cv2.destroyAllWindows()
    hard_examples_file.close()

    print(f"\n--- Zusammenfassung ---")
    print(f"  Überprüft: {idx}/{len(suspects)}")
    print(f"  Korrekt bestätigt: {kept} (gespeichert in {HARD_EXAMPLES_PATH.name})")
    print(f"  Verschoben: {moved}")
    print(f"  Gelöscht: {deleted}")


def _move_image(src_path: Path, new_label: str):
    """Verschiebt ein Bild in den korrekten Klassen-Ordner (gleicher Split)."""
    # Beispiel: dataset/train/5/img.png → dataset/train/7/img.png
    split_dir = src_path.parent.parent
    target_dir = split_dir / new_label
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / src_path.name
    if target_path.exists():
        # Dateiname anpassen um Konflikte zu vermeiden
        stem = src_path.stem
        suffix = src_path.suffix
        i = 1
        while target_path.exists():
            target_path = target_dir / f"{stem}_{i}{suffix}"
            i += 1

    shutil.move(str(src_path), str(target_path))
    print(f"  VERSCHOBEN: {src_path.parent.name}/{src_path.name} → {new_label}/{target_path.name}")


# --- YOLO Mislabel Scan ---

def scan_yolo_splits(splits, threshold):
    """Scannt YOLO-Labels mit dem YOLO-Modell und findet Widersprüche."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics nicht installiert. Bitte: pip install ultralytics")
        sys.exit(1)

    if not YOLO_MODEL_PATH.exists():
        print(f"YOLO-Modell nicht gefunden: {YOLO_MODEL_PATH}")
        sys.exit(1)

    model = YOLO(str(YOLO_MODEL_PATH))
    print(f"YOLO-Modell geladen: {YOLO_MODEL_PATH.name}")

    suspects = []

    for split in splits:
        img_dir = YOLO_DIR / split / "images"
        lbl_dir = YOLO_DIR / split / "labels"
        if not img_dir.exists():
            print(f"  {split}/images/ nicht gefunden, überspringe...")
            continue

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        total = 0
        flagged = 0
        total_images = len(images)

        for i, img_path in enumerate(images):
            total += 1
            label_path = lbl_dir / (img_path.stem + ".txt")

            # Labels aus Datei laden
            file_labels = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        xc, yc, bw, bh = (float(p) for p in parts[1:])
                        file_labels.append((cls, xc, yc, bw, bh))

            # YOLO-Vorhersage
            results = model.predict(str(img_path), verbose=False, conf=0.3)
            pred_boxes = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    pred_cls = int(box.cls[0])
                    pred_conf = float(box.conf[0])
                    pred_boxes.append((pred_cls, pred_conf, box.xyxyn[0].cpu().numpy()))

            # Vergleich: Für jede Label-Box die nächste Prediction finden
            mismatches = []
            for lbl_cls, xc, yc, bw, bh in file_labels:
                lbl_x1, lbl_y1 = xc - bw / 2, yc - bh / 2
                lbl_x2, lbl_y2 = xc + bw / 2, yc + bh / 2
                best_iou = 0
                best_pred = None

                for pred_cls, pred_conf, xyxyn in pred_boxes:
                    px1, py1, px2, py2 = xyxyn
                    iou = _calc_iou(lbl_x1, lbl_y1, lbl_x2, lbl_y2, px1, py1, px2, py2)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = (pred_cls, pred_conf)

                if best_pred and best_iou > 0.3 and best_pred[0] != lbl_cls and best_pred[1] >= threshold:
                    mismatches.append({
                        "label_cls": lbl_cls,
                        "pred_cls": best_pred[0],
                        "pred_conf": best_pred[1],
                        "iou": best_iou,
                        "box": (xc, yc, bw, bh),
                    })

            if mismatches:
                suspects.append({
                    "img_path": img_path,
                    "label_path": label_path,
                    "split": split,
                    "mismatches": mismatches,
                    "all_labels": file_labels,
                    "max_conf": max(m["pred_conf"] for m in mismatches),
                })
                flagged += 1

            print(f"\r  {split}: {i+1}/{total_images} Bilder ({flagged} Widersprüche)", end="", flush=True)

        print(f"\r  {split}: {total} Bilder geprüft, {flagged} mit Label-Widersprüchen{' ' * 20}")

    suspects.sort(key=lambda x: x["max_conf"], reverse=True)
    return suspects


def _calc_iou(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b):
    """Berechnet IoU (Intersection over Union) zweier Boxen."""
    ix1 = max(x1a, x1b)
    iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b)
    iy2 = min(y2a, y2b)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def review_yolo_suspects(suspects):
    """Interaktive Überprüfung der verdächtigen YOLO-Labels."""
    if not suspects:
        print("\nKeine verdächtigen YOLO-Labels gefunden!")
        return

    print(f"\n{'='*60}")
    print(f"{len(suspects)} Bilder mit verdächtigen YOLO-Labels.")
    print(f"Steuerung: e=im Editor bearbeiten, k/Enter=ok, q=beenden")
    print(f"{'='*60}\n")

    edited = 0
    confirmed = 0
    idx = 0
    hard_examples_file = open(str(HARD_EXAMPLES_YOLO_PATH), "a", encoding="utf-8")

    while idx < len(suspects):
        s = suspects[idx]
        img = cv2.imread(str(s["img_path"]))
        if img is None:
            idx += 1
            continue

        h, w = img.shape[:2]

        # Alle Labels + Mismatches zeichnen
        for lbl_cls, xc, yc, bw, bh in s["all_labels"]:
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            # Prüfen ob dieser Label einen Mismatch hat
            is_mismatch = False
            pred_info = ""
            for m in s["mismatches"]:
                if m["label_cls"] == lbl_cls and m["box"] == (xc, yc, bw, bh):
                    is_mismatch = True
                    pred_name = CLASS_NAMES.get(m["pred_cls"], "?")
                    pred_info = f" -> CNN:{pred_name} ({m['pred_conf']:.0%})"
                    break

            color = (0, 0, 255) if is_mismatch else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 5 if is_mismatch else 3)
            label = CLASS_NAMES.get(lbl_cls, "?") + pred_info
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Skalieren
        scale = min(1600 / w, 900 / h, 1.0)
        display = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

        mismatch_details = []
        for m in s["mismatches"]:
            lbl = CLASS_NAMES.get(m["label_cls"], "?")
            pred = CLASS_NAMES.get(m["pred_cls"], "?")
            mismatch_details.append(f"{lbl}→{pred}({m['pred_conf']:.0%})")

        title = f"[{idx+1}/{len(suspects)}] {s['img_path'].name} - Mismatches: {', '.join(mismatch_details)}"
        print(f"  {title}")

        cv2.namedWindow("YOLO Mislabel Review", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Mislabel Review", display.shape[1], display.shape[0])
        cv2.imshow("YOLO Mislabel Review", display)

        key = cv2.waitKey(0) & 0xFFFF

        if key == ord('q'):
            break
        elif key == ord('e'):
            # Im Label-Editor öffnen
            cv2.destroyAllWindows()
            import subprocess
            subprocess.run([
                sys.executable, str(SCRIPT_DIR / "label_for_yolo.py"),
                "--edit-mode", s["img_path"].name,
                "--split", s["split"],
            ])
            edited += 1

            # Nach Editor: Bild nochmal zeigen, k = korrekt → Hard Example speichern
            img2 = cv2.imread(str(s["img_path"]))
            if img2 is not None:
                h2, w2 = img2.shape[:2]
                scale2 = min(1600 / w2, 900 / h2, 1.0)
                disp2 = cv2.resize(img2, (int(w2 * scale2), int(h2 * scale2))) if scale2 < 1.0 else img2
                cv2.namedWindow("YOLO Mislabel Review", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLO Mislabel Review", disp2.shape[1], disp2.shape[0])
                cv2.imshow("YOLO Mislabel Review", disp2)
                print(f"    → Editor geschlossen. k=korrekt (Hard Example), andere Taste=weiter")
                key2 = cv2.waitKey(0) & 0xFFFF
                if key2 in (ord('k'), 13):
                    hard_examples_file.write(str(s["img_path"]) + "\n")
                    confirmed += 1
            idx += 1
        elif key in (ord('k'), 13):
            hard_examples_file.write(str(s["img_path"]) + "\n")
            confirmed += 1
            idx += 1

    cv2.destroyAllWindows()
    hard_examples_file.close()

    print(f"\n--- Zusammenfassung ---")
    print(f"  Überprüft: {idx}/{len(suspects)}")
    print(f"  Im Editor bearbeitet: {edited}")
    print(f"  Korrekt bestätigt: {confirmed} (gespeichert in {HARD_EXAMPLES_YOLO_PATH.name})")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Sucht nach falsch gelabelten Trainingsbildern",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--split", nargs="+", default=["train", "val", "test"],
                        help="Welche Splits prüfen (default: train val test)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Min. Konfidenz für Verdacht (default: 0.7)")
    parser.add_argument("--yolo", action="store_true",
                        help="YOLO-Labels statt CNN-Klassifikation prüfen")
    parser.add_argument("--scan-only", action="store_true",
                        help="Nur scannen, nicht interaktiv reviewen")
    args = parser.parse_args()

    print(f"Mislabel-Finder")
    print(f"Modus: {'YOLO' if args.yolo else 'CNN'}")
    print(f"Splits: {', '.join(args.split)}")
    print(f"Schwelle: {args.threshold:.0%}\n")

    if args.yolo:
        suspects = scan_yolo_splits(args.split, args.threshold)
        if args.scan_only:
            _print_yolo_report(suspects)
        else:
            review_yolo_suspects(suspects)
    else:
        model, idx_to_class, class_to_idx, transform, device = load_cnn_model()
        suspects = scan_cnn_splits(model, idx_to_class, class_to_idx, transform, device,
                                   args.split, args.threshold)
        if args.scan_only:
            _print_cnn_report(suspects)
        else:
            review_cnn_suspects(suspects, idx_to_class)


def _print_cnn_report(suspects):
    """Gibt einen Bericht ohne GUI aus."""
    if not suspects:
        print("\nKeine verdächtigen Labels gefunden!")
        return

    print(f"\n{len(suspects)} verdächtige Bilder:")
    print(f"{'Datei':<50} {'Ordner':>8} {'CNN sagt':>10} {'Konfidenz':>10}")
    print("-" * 80)
    for s in suspects:
        rel = f"{s['split']}/{s['folder_label']}/{s['path'].name}"
        print(f"{rel:<50} {s['folder_label']:>8} {s['pred_label']:>10} {s['confidence']:>9.0%}")


def _print_yolo_report(suspects):
    """Gibt YOLO-Bericht ohne GUI aus."""
    if not suspects:
        print("\nKeine verdächtigen YOLO-Labels gefunden!")
        return

    print(f"\n{len(suspects)} Bilder mit verdächtigen Labels:")
    for s in suspects:
        print(f"\n  {s['img_path'].name} ({s['split']}):")
        for m in s["mismatches"]:
            lbl = CLASS_NAMES.get(m["label_cls"], "?")
            pred = CLASS_NAMES.get(m["pred_cls"], "?")
            print(f"    Label: {lbl}  →  YOLO sagt: {pred} ({m['pred_conf']:.0%}, IoU: {m['iou']:.2f})")


if __name__ == "__main__":
    main()
