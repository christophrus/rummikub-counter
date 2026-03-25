"""Prüft welche Bilder im YOLO-Dataset Portrait- vs Landscape-Format haben."""
import cv2
from pathlib import Path

img_dir = Path(__file__).parent.parent / "yolo_dataset" / "train" / "images"
all_imgs = sorted(img_dir.glob("*.jpg"))
print(f"Gesamt: {len(all_imgs)} Bilder")

portrait = []
landscape = []
for f in all_imgs:
    img = cv2.imread(str(f))
    if img is None:
        print(f"  FEHLER: {f.name} konnte nicht geladen werden")
        continue
    h, w = img.shape[:2]
    if h > w:
        portrait.append((f.name, w, h))
    else:
        landscape.append((f.name, w, h))

print(f"Portrait (h>w): {len(portrait)}")
print(f"Landscape (w>h): {len(landscape)}")

if portrait:
    print("\nAlle Portrait-Bilder:")
    for name, w, h in portrait:
        print(f"  {name}: {w}x{h}")
