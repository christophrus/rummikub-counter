"""
Dataset-Split: Teilt augmentierte Bilder in Train/Val/Test auf.

Nutzung:
    python split_dataset.py
    python split_dataset.py --train 0.8 --val 0.1 --test 0.1

Standard-Split: 80% Train, 10% Validation, 10% Test
"""

import argparse
import random
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AUGMENTED_DIR = SCRIPT_DIR / "tiles_augmented"
TRAIN_DIR = SCRIPT_DIR / "train"
VAL_DIR = SCRIPT_DIR / "val"
TEST_DIR = SCRIPT_DIR / "test"


def main():
    parser = argparse.ArgumentParser(description="Rummikub Dataset Split")
    parser.add_argument("--train", type=float, default=0.8, help="Train-Anteil (Standard: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Validation-Anteil (Standard: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test-Anteil (Standard: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed (Standard: 42)")
    args = parser.parse_args()

    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Fehler: Train + Val + Test = {total_ratio} (muss 1.0 sein)")
        return

    print("=" * 60)
    print("  Rummikub Dataset Split")
    print(f"  Train: {args.train:.0%} | Val: {args.val:.0%} | Test: {args.test:.0%}")
    print("=" * 60)

    if not AUGMENTED_DIR.exists():
        print("Fehler: tiles_augmented/ nicht gefunden. Zuerst augment_dataset.py ausführen!")
        return

    random.seed(args.seed)

    # Alte Splits löschen
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if d.exists():
            shutil.rmtree(d)

    total_stats = {"train": 0, "val": 0, "test": 0}

    labels = sorted([d.name for d in AUGMENTED_DIR.iterdir() if d.is_dir()])

    for label in labels:
        src_dir = AUGMENTED_DIR / label
        images = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))

        if not images:
            continue

        # Shufflen
        random.shuffle(images)

        n = len(images)
        n_train = int(n * args.train)
        n_val = int(n * args.val)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Kopieren
        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_dir = SCRIPT_DIR / split_name / label
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_imgs:
                shutil.copy2(str(img_path), str(split_dir / img_path.name))

            total_stats[split_name] += len(split_imgs)

        print(f"  {label:>5}: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

    print(f"\nGesamt:")
    print(f"  Train: {total_stats['train']}")
    print(f"  Val:   {total_stats['val']}")
    print(f"  Test:  {total_stats['test']}")
    print(f"\nNächster Schritt: python train_cnn.py")


if __name__ == "__main__":
    main()
