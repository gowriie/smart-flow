import os
import json
import shutil
from pathlib import Path

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_MAPS_DIR = PROJECT_ROOT / "data" / "synthetic_test_maps"
TEST_META_DIR = PROJECT_ROOT / "data" / "meta_test"

YOLO_DIR = PROJECT_ROOT / "data" / "yolo"
IMAGES_TEST = YOLO_DIR / "images" / "test"
LABELS_TEST = YOLO_DIR / "labels" / "test"

# Class IDs (must match training)
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
    for h in houses:
        cx, cy, w, hh = rect_to_yolo(h["x1"], h["y1"], h["x2"], h["y2"], img_w, img_h)
        lines.append(f"{HOUSE_ID} {cx:.6f} {cy:.6f} {w:.6f} {hh:.6f}")
    for o in obstacles:
        cx, cy, w, hh = rect_to_yolo(o["x1"], o["y1"], o["x2"], o["y2"], img_w, img_h)
        lines.append(f"{OBSTACLE_ID} {cx:.6f} {cy:.6f} {w:.6f} {hh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def clear_dir(p: Path):
    if not p.exists():
        return
    for item in p.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink(missing_ok=True)
        else:
            shutil.rmtree(item, ignore_errors=True)


def ensure_test_in_data_yaml():
    """Add 'test: images/test' to data/yolo/data.yaml if missing."""
    yaml_path = YOLO_DIR / "data.yaml"
    if not yaml_path.exists():
        # minimal YAML if training file was removed for some reason
        yaml_text = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: house
  1: obstacle
"""
        yaml_path.write_text(yaml_text, encoding="utf-8")
        return

    text = yaml_path.read_text(encoding="utf-8")
    if "test:" not in text:
        # insert after 'val:' line, or append
        lines = text.splitlines()
        inserted = False
        for i, line in enumerate(lines):
            if line.strip().startswith("val:"):
                lines.insert(i + 1, "test: images/test")
                inserted = True
                break
        if not inserted:
            lines.append("test: images/test")
        yaml_path.write_text("\n".join(lines) + ("\n" if not lines[-1].endswith("\n") else ""), encoding="utf-8")


def prepare_test(clear_first: bool = True):
    # Create dirs
    IMAGES_TEST.mkdir(parents=True, exist_ok=True)
    LABELS_TEST.mkdir(parents=True, exist_ok=True)

    if clear_first:
        clear_dir(IMAGES_TEST)
        clear_dir(LABELS_TEST)

    # Accept test_*.json primarily; fall back to map_*.json if needed
    meta_files = sorted(TEST_META_DIR.glob("test_*.json"))
    if not meta_files:
        meta_files = sorted(TEST_META_DIR.glob("map_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No test meta JSON files found in {TEST_META_DIR}")

    print(f"[TEST] Found {len(meta_files)} meta files")

    for j, meta_path in enumerate(meta_files, 1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        img_name = meta["image"]

        # Handle both (img_w,img_h) or legacy img_size
        if "img_w" in meta and "img_h" in meta:
            img_w = int(meta["img_w"])
            img_h = int(meta["img_h"])
        else:
            s = int(meta.get("img_size", 512))
            img_w = s
            img_h = s

        img_src = TEST_MAPS_DIR / img_name
        if not img_src.exists():
            raise FileNotFoundError(f"Missing test image: {img_src}")

        img_dst = IMAGES_TEST / img_name
        lbl_dst = LABELS_TEST / (Path(img_name).stem + ".txt")

        shutil.copy2(img_src, img_dst)
        write_label_file(lbl_dst, meta["houses"], meta["obstacles"], img_w, img_h)

        if j % 25 == 0 or j == len(meta_files):
            print(f"[TEST] Prepared {j}/{len(meta_files)}")

    ensure_test_in_data_yaml()
    print("\n✅ YOLO TEST set prepared.")
    print("   Images:", IMAGES_TEST)
    print("   Labels:", LABELS_TEST)
    print("   data.yaml updated with 'test: images/test' (if needed).")


def main():
    prepare_test(clear_first=True)


if __name__ == "__main__":
    main()
