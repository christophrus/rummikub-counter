"""
Augmentation: Erzeugt zusätzliche Trainingsbilder durch Transformationen.

Nutzung:
    python augment_dataset.py
    python augment_dataset.py --factor 10    (10x statt Standard 5x)

Transformationen:
- Rotation (±15°)
- Helligkeit/Kontrast
- Horizontales/Vertikales Spiegeln
- Gaussian Blur
- Leichtes Zuschneiden
- Perspektivische Verzerrung
"""

import argparse
import random
import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TILES_DIR = SCRIPT_DIR / "tiles"
AUGMENTED_DIR = SCRIPT_DIR / "tiles_augmented"


def random_rotate(image: np.ndarray) -> np.ndarray:
    """Zufällige Rotation ±15°."""
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_brightness_contrast(image: np.ndarray) -> np.ndarray:
    """Zufällige Helligkeit und Kontrast."""
    alpha = random.uniform(0.7, 1.3)  # Kontrast
    beta = random.randint(-30, 30)     # Helligkeit
    return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)


def random_flip(image: np.ndarray) -> np.ndarray:
    """Zufälliges Spiegeln."""
    choice = random.choice(["none", "h", "v", "both"])
    if choice == "h":
        return cv2.flip(image, 1)
    elif choice == "v":
        return cv2.flip(image, 0)
    elif choice == "both":
        return cv2.flip(image, -1)
    return image


def random_blur(image: np.ndarray) -> np.ndarray:
    """Zufälliger leichter Blur."""
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    return image


def random_crop(image: np.ndarray) -> np.ndarray:
    """Leichtes zufälliges Zuschneiden (5-10% pro Seite)."""
    h, w = image.shape[:2]
    crop_pct = random.uniform(0.02, 0.08)
    top = int(h * crop_pct * random.random())
    bottom = h - int(h * crop_pct * random.random())
    left = int(w * crop_pct * random.random())
    right = w - int(w * crop_pct * random.random())
    cropped = image[top:bottom, left:right]
    return cv2.resize(cropped, (w, h))


def random_perspective(image: np.ndarray) -> np.ndarray:
    """Leichte perspektivische Verzerrung."""
    h, w = image.shape[:2]
    offset = int(min(h, w) * 0.05)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, offset), random.randint(0, offset)],
        [w - random.randint(0, offset), random.randint(0, offset)],
        [w - random.randint(0, offset), h - random.randint(0, offset)],
        [random.randint(0, offset), h - random.randint(0, offset)]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_noise(image: np.ndarray) -> np.ndarray:
    """Leichtes Gaussian Noise."""
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(3, 10), image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy
    return image


def augment_image(image: np.ndarray) -> np.ndarray:
    """Wendet eine zufällige Kombination von Augmentierungen an."""
    transforms = [
        random_rotate,
        random_brightness_contrast,
        random_flip,
        random_blur,
        random_crop,
        random_perspective,
        random_noise,
    ]

    # 2-4 zufällige Transformationen anwenden
    num_transforms = random.randint(2, 4)
    selected = random.sample(transforms, num_transforms)

    result = image.copy()
    for transform in selected:
        result = transform(result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Rummikub Dataset Augmentation")
    parser.add_argument("--factor", type=int, default=5,
                        help="Wieviele augmentierte Bilder pro Original (Standard: 5)")
    parser.add_argument("--size", type=int, nargs=2, default=[64, 96],
                        help="Zielgröße WxH (Standard: 64 96)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Rummikub Dataset Augmentation")
    print("=" * 60)

    if not TILES_DIR.exists():
        print("Fehler: tiles/ Ordner nicht gefunden. Zuerst extract_tiles.py ausführen!")
        return

    target_w, target_h = args.size

    total_original = 0
    total_augmented = 0

    labels = sorted([d.name for d in TILES_DIR.iterdir() if d.is_dir()])

    for label in labels:
        src_dir = TILES_DIR / label
        dst_dir = AUGMENTED_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg"))
        if not images:
            continue

        total_original += len(images)
        count = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Original auch speichern (resized)
            resized = cv2.resize(img, (target_w, target_h))
            cv2.imwrite(str(dst_dir / f"orig_{img_path.stem}.png"), resized)
            count += 1

            # Augmentierte Versionen
            for aug_idx in range(args.factor):
                augmented = augment_image(img)
                augmented = cv2.resize(augmented, (target_w, target_h))
                cv2.imwrite(str(dst_dir / f"aug_{img_path.stem}_{aug_idx:02d}.png"), augmented)
                count += 1

        total_augmented += count
        print(f"  {label:>5}: {len(images)} Original → {count} Bilder")

    print(f"\nGesamt: {total_original} Original → {total_augmented} Bilder")
    print(f"Gespeichert in: {AUGMENTED_DIR}")
    print(f"\nNächster Schritt: python split_dataset.py")


if __name__ == "__main__":
    main()
