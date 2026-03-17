import os
import json
import random
import shutil
from pathlib import Path

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPS_DIR = PROJECT_ROOT / "data" / "synthetic_maps"
META_DIR = PROJECT_ROOT / "data" / "meta"

YOLO_DIR = PROJECT_ROOT / "data" / "yolo"
IMAGES_TRAIN = YOLO_DIR / "images" / "train"
IMAGES_VAL = YOLO_DIR / "images" / "val"
LABELS_TRAIN = YOLO_DIR / "labels" / "train"
LABELS_VAL = YOLO_DIR / "labels" / "val"

# Train/Val split
VAL_RATIO = 0.2

# Class IDs
HOUSE_ID = 0
OBSTACLE_ID = 1
# ----------------------------------------


def rect_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel rectangle to YOLO normalized (cx, cy, w, h)."""
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2
    cy = y1 + box_h / 2
    return (cx / img_w, cy / img_h, box_w / img_w, box_h / img_h)


def write_label_file(label_path: Path, houses, obstacles, img_w: int, img_h: int):
    lines = []

    # Houses
    for h in houses:
        cx, cy, w, hh = rect_to_yolo(h["x1"], h["y1"], h["x2"], h["y2"], img_w, img_h)
        lines.append(f"{HOUSE_ID} {cx:.6f} {cy:.6f} {w:.6f} {hh:.6f}")

    # Obstacles (bounding boxes)
    for o in obstacles:
        cx, cy, w, hh = rect_to_yolo(o["x1"], o["y1"], o["x2"], o["y2"], img_w, img_h)
        lines.append(f"{OBSTACLE_ID} {cx:.6f} {cy:.6f} {w:.6f} {hh:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    # Create output dirs
    for p in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        p.mkdir(parents=True, exist_ok=True)

    # Collect meta files
    meta_files = sorted(META_DIR.glob("map_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No meta JSON files found in {META_DIR}")

    # Shuffle for split
    random.seed(42)
    random.shuffle(meta_files)

    val_count = int(len(meta_files) * VAL_RATIO)
    val_set = set(meta_files[:val_count])
    train_set = set(meta_files[val_count:])

    print(f"Total: {len(meta_files)} | Train: {len(train_set)} | Val: {len(val_set)}")

    # Copy images + write labels
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        img_name = meta["image"]

        # ✅ Correct dimensions (works for both 512x512 and 1280x720)
        if "img_w" in meta and "img_h" in meta:
            img_w = int(meta["img_w"])
            img_h = int(meta["img_h"])
        else:
            # fallback for older simple meta format
            img_size = int(meta.get("img_size", 512))
            img_w = img_size
            img_h = img_size

        img_src = MAPS_DIR / img_name
        if not img_src.exists():
            raise FileNotFoundError(f"Missing image: {img_src}")

        is_val = meta_path in val_set

        img_dst = (IMAGES_VAL if is_val else IMAGES_TRAIN) / img_name
        label_dst = (LABELS_VAL if is_val else LABELS_TRAIN) / (Path(img_name).stem + ".txt")

        shutil.copy2(img_src, img_dst)
        write_label_file(label_dst, meta["houses"], meta["obstacles"], img_w, img_h)

    # Create data.yaml for YOLO
    yaml_text = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: house
  1: obstacle
"""
    (YOLO_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("\n✅ YOLO dataset created at:", YOLO_DIR)
    print("✅ data.yaml created at:", YOLO_DIR / "data.yaml")
    print("✅ Sample label file:", next(LABELS_TRAIN.glob("*.txt"), None))


if __name__ == "__main__":
    main()
