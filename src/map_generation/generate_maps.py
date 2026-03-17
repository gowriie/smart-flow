import os
import json
import random
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple

import cv2
import numpy as np


# ---------- Data structure ----------
@dataclass
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int


def overlap(a: Rect, b: Rect, pad: int = 0) -> bool:
    return not (
        a.x2 + pad <= b.x1 or
        a.x1 >= b.x2 + pad or
        a.y2 + pad <= b.y1 or
        a.y1 >= b.y2 + pad
    )


# ============================================================
#                  PART A: SIMPLE MAPS (512x512)
#          (This is your original generate_maps.py logic)
# ============================================================

def random_rect_simple(img_size: int, size_range: Tuple[int, int], margin: int = 8) -> Rect:
    w = random.randint(size_range[0], size_range[1])
    h = random.randint(size_range[0], size_range[1])
    x1 = random.randint(margin, img_size - margin - w)
    y1 = random.randint(margin, img_size - margin - h)
    return Rect(x1, y1, x1 + w, y1 + h)


def draw_grid_simple(img: np.ndarray, step: int = 32) -> None:
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (245, 245, 245), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (245, 245, 245), 1)


def draw_roads_simple(img: np.ndarray, n_roads: int = 7, thickness: int = 7) -> List[Rect]:
    """Draw light gray road bands and return their rectangles."""
    h, w = img.shape[:2]
    road_rects: List[Rect] = []

    for _ in range(n_roads):
        if random.random() < 0.5:
            y = random.randint(30, h - 30)
            cv2.line(img, (0, y), (w, y), (230, 230, 230), thickness=thickness)
            road_rects.append(Rect(0, y - thickness // 2, w, y + thickness // 2))
        else:
            x = random.randint(30, w - 30)
            cv2.line(img, (x, 0), (x, h), (230, 230, 230), thickness=thickness)
            road_rects.append(Rect(x - thickness // 2, 0, x + thickness // 2, h))

    return road_rects


def draw_river_simple(img: np.ndarray, thickness: int = 18) -> None:
    """Draw a soft constraint like a river (light blue)."""
    h, w = img.shape[:2]
    pts = []
    x = 0
    y = random.randint(int(h * 0.2), int(h * 0.8))
    while x < w:
        pts.append((x, y))
        x += random.randint(40, 90)
        y = max(20, min(h - 20, y + random.randint(-60, 60)))

    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (200, 220, 255), thickness=thickness)


def draw_obstacles_polygon_simple(img: np.ndarray, n: int = 7, size_range: Tuple[int, int] = (70, 140)) -> List[Rect]:
    """Draw irregular obstacles (polygons). Return their bounding boxes."""
    rects: List[Rect] = []
    attempts = 0

    while len(rects) < n and attempts < 8000:
        attempts += 1
        r = random_rect_simple(img.shape[0], size_range)
        if any(overlap(r, rr, pad=12) for rr in rects):
            continue
        rects.append(r)

    for r in rects:
        cx = (r.x1 + r.x2) // 2
        cy = (r.y1 + r.y2) // 2
        rx = (r.x2 - r.x1) // 2
        ry = (r.y2 - r.y1) // 2

        pts = []
        k = random.randint(5, 8)
        for i in range(k):
            ang = 2 * math.pi * i / k + random.random() * 0.3
            px = int(cx + rx * math.cos(ang) * random.uniform(0.7, 1.0))
            py = int(cy + ry * math.sin(ang) * random.uniform(0.7, 1.0))
            pts.append([px, py])

        pts_np = np.array([pts], dtype=np.int32)
        cv2.fillPoly(img, pts_np, (60, 60, 60))

    return rects


def draw_houses_simple(
    img: np.ndarray,
    n: int = 40,
    size_range: Tuple[int, int] = (16, 30),
    forbidden: List[Rect] | None = None,
    min_gap: int = 4
) -> List[Rect]:
    """Draw houses as small black rectangles. Return house rectangles."""
    if forbidden is None:
        forbidden = []

    houses: List[Rect] = []
    attempts = 0

    while len(houses) < n and attempts < 20000:
        attempts += 1
        r = random_rect_simple(img.shape[0], size_range)
        if any(overlap(r, f, pad=min_gap) for f in forbidden):
            continue
        if any(overlap(r, h, pad=min_gap) for h in houses):
            continue
        houses.append(r)

    for h in houses:
        cv2.rectangle(img, (h.x1, h.y1), (h.x2, h.y2), (0, 0, 0), -1)

    return houses


def add_scan_effect_simple(img: np.ndarray) -> np.ndarray:
    """Make the map look like a scanned blueprint (slight noise + blur + tiny rotation)."""
    h, w = img.shape[:2]

    noisy = img.astype(np.int16)
    speckle = (np.random.randn(h, w, 1) * 4).astype(np.int16)
    noisy = np.clip(noisy + speckle, 0, 255).astype(np.uint8)

    noisy = cv2.GaussianBlur(noisy, (3, 3), 0)

    angle = random.uniform(-2.0, 2.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    noisy = cv2.warpAffine(noisy, M, (w, h), borderValue=(255, 255, 255))

    return noisy


def draw_legend_simple(img: np.ndarray) -> None:
    cv2.rectangle(img, (10, 10), (200, 95), (255, 255, 255), -1)
    cv2.rectangle(img, (18, 22), (38, 42), (0, 0, 0), -1)
    cv2.putText(img, "House", (46, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (18, 55), (38, 75), (60, 60, 60), -1)
    cv2.putText(img, "Obstacle", (46, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def generate_simple_map(img_size: int = 512, style: str = "scan"):
    """
    style:
      - "full": grid + roads + river + polygon obstacles + houses
      - "scan": same as "full" but with scan effect
    """
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    draw_grid_simple(img, step=32)
    draw_roads_simple(img, n_roads=7, thickness=7)
    draw_river_simple(img, thickness=18)

    obstacles = draw_obstacles_polygon_simple(img, n=random.randint(6, 9))
    houses = draw_houses_simple(img, n=random.randint(30, 50), forbidden=obstacles)

    draw_legend_simple(img)

    if style == "scan":
        img = add_scan_effect_simple(img)

    return img, houses, obstacles


# ============================================================
#                 PART B: COMPLEX MAPS (1280x720)
#        (This is your original generate_maps_v2 logic)
# ============================================================

def random_rect_v2(w: int, h: int, size_range: Tuple[int, int], margin: int = 10) -> Rect:
    rw = random.randint(size_range[0], size_range[1])
    rh = random.randint(size_range[0], size_range[1])
    x1 = random.randint(margin, w - margin - rw)
    y1 = random.randint(margin, h - margin - rh)
    return Rect(x1, y1, x1 + rw, y1 + rh)


def draw_background_v2(img: np.ndarray):
    base = img.copy()
    noise = (np.random.randn(*img.shape[:2], 1) * 3).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    img[:] = base


def draw_grid_v2(img: np.ndarray, step: int = 40):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (245, 245, 245), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (245, 245, 245), 1)


def draw_roads_v2(img: np.ndarray, n: int = 8) -> List[Rect]:
    """Thicker colored roads with borders."""
    h, w = img.shape[:2]
    road_rects: List[Rect] = []

    for _ in range(n):
        if random.random() < 0.5:
            y = random.randint(40, h - 40)
            thickness = random.choice([10, 14, 18])
            cv2.line(img, (0, y), (w, y), (170, 170, 170), thickness=thickness + 4)
            cv2.line(img, (0, y), (w, y), (220, 220, 220), thickness=thickness)
            road_rects.append(Rect(0, y - (thickness // 2 + 2), w, y + (thickness // 2 + 2)))
        else:
            x = random.randint(40, w - 40)
            thickness = random.choice([10, 14, 18])
            cv2.line(img, (x, 0), (x, h), (170, 170, 170), thickness=thickness + 4)
            cv2.line(img, (x, 0), (x, h), (220, 220, 220), thickness=thickness)
            road_rects.append(Rect(x - (thickness // 2 + 2), 0, x + (thickness // 2 + 2), h))

    return road_rects


def draw_river_v2(img: np.ndarray):
    """Blue river with darker edge."""
    h, w = img.shape[:2]
    pts = []
    x = 0
    y = random.randint(int(h * 0.25), int(h * 0.75))
    while x < w:
        pts.append((x, y))
        x += random.randint(70, 130)
        y = max(30, min(h - 30, y + random.randint(-80, 80)))

    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (170, 200, 255), thickness=26)
        cv2.line(img, pts[i], pts[i + 1], (200, 225, 255), thickness=18)


def draw_parks_v2(img: np.ndarray, n: int = 3) -> List[Rect]:
    h, w = img.shape[:2]
    parks: List[Rect] = []
    for _ in range(n):
        r = random_rect_v2(w, h, (120, 220), margin=20)
        parks.append(r)
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (210, 245, 210), -1)
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (160, 210, 160), 2)
    return parks


def draw_obstacles_v2(img: np.ndarray, n: int = 8) -> List[Rect]:
    h, w = img.shape[:2]
    rects: List[Rect] = []
    attempts = 0
    while len(rects) < n and attempts < 12000:
        attempts += 1
        r = random_rect_v2(w, h, (120, 220), margin=30)
        if any(overlap(r, rr, pad=18) for rr in rects):
            continue
        rects.append(r)

    for r in rects:
        cx = (r.x1 + r.x2) // 2
        cy = (r.y1 + r.y2) // 2
        rx = (r.x2 - r.x1) // 2
        ry = (r.y2 - r.y1) // 2

        pts = []
        k = random.randint(6, 10)
        for i in range(k):
            ang = 2 * math.pi * i / k + random.random() * 0.3
            px = int(cx + rx * math.cos(ang) * random.uniform(0.7, 1.0))
            py = int(cy + ry * math.sin(ang) * random.uniform(0.7, 1.0))
            pts.append([px, py])

        pts_np = np.array([pts], dtype=np.int32)
        cv2.fillPoly(img, pts_np, (70, 70, 70))
        cv2.polylines(img, pts_np, True, (40, 40, 40), 2)

    return rects


def draw_houses_v2(img: np.ndarray, forbidden: List[Rect], n: int = 80) -> List[Rect]:
    h, w = img.shape[:2]
    houses: List[Rect] = []
    attempts = 0
    while len(houses) < n and attempts < 60000:
        attempts += 1
        r = random_rect_v2(w, h, (18, 36), margin=12)
        if any(overlap(r, f, pad=6) for f in forbidden):
            continue
        if any(overlap(r, hh, pad=4) for hh in houses):
            continue
        houses.append(r)

    for r in houses:
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (60, 60, 150), -1)
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (30, 30, 90), 1)

    return houses


def draw_legend_v2(img: np.ndarray):
    cv2.rectangle(img, (12, 12), (260, 110), (255, 255, 255), -1)
    cv2.rectangle(img, (22, 26), (46, 50), (60, 60, 150), -1)
    cv2.putText(img, "House", (60, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    cv2.rectangle(img, (22, 66), (46, 90), (70, 70, 70), -1)
    cv2.putText(img, "Obstacle", (60, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)


def add_scan_style_v2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.astype(np.int16)
    speckle = (np.random.randn(h, w, 1) * 2).astype(np.int16)
    out = np.clip(out + speckle, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    angle = random.uniform(-1.2, 1.2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderValue=(245, 245, 245))
    return out


def generate_complex_map(width: int = 1280, height: int = 720, style: str = "pretty"):
    img = np.ones((height, width, 3), dtype=np.uint8) * 250
    draw_background_v2(img)
    draw_grid_v2(img, step=40)
    draw_roads_v2(img, n=8)
    draw_river_v2(img)
    parks = draw_parks_v2(img, n=3)
    obstacles = draw_obstacles_v2(img, n=random.randint(7, 10))
    forbidden = obstacles + parks
    houses = draw_houses_v2(img, forbidden=forbidden, n=random.randint(70, 110))
    draw_legend_v2(img)

    if style == "scan":
        img = add_scan_style_v2(img)

    return img, houses, obstacles


# ============================================================
#                       MAIN (COMBINED)
# ============================================================
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    out_img_dir = os.path.join(project_root, "data", "synthetic_maps")
    out_meta_dir = os.path.join(project_root, "data", "meta")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_meta_dir, exist_ok=True)

    num_simple = 500
    num_complex = 500

    # ----------- Generate SIMPLE maps: 001 - 500 -----------
    for i in range(1, num_simple + 1):
        style = "scan" if random.random() < 0.7 else "full"
        img, houses, obstacles = generate_simple_map(img_size=512, style=style)

        name = f"map_{i:03d}"
        img_path = os.path.join(out_img_dir, name + ".png")
        meta_path = os.path.join(out_meta_dir, name + ".json")

        cv2.imwrite(img_path, img)

        meta = {
            "image": os.path.basename(img_path),
            "style": style,
            "img_size": 512,
            "houses": [asdict(h) for h in houses],
            "obstacles": [asdict(o) for o in obstacles],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if i % 50 == 0:
            print(f"[SIMPLE OK] saved {i}/{num_simple}")

    # ----------- Generate COMPLEX maps: 501 - 1000 -----------
    width, height = 1280, 720
    for j in range(1, num_complex + 1):
        idx = num_simple + j  # 501..1000
        style = "scan" if random.random() < 0.5 else "pretty"
        img, houses, obstacles = generate_complex_map(width, height, style=style)

        name = f"map_{idx:03d}"
        img_path = os.path.join(out_img_dir, name + ".png")
        meta_path = os.path.join(out_meta_dir, name + ".json")

        cv2.imwrite(img_path, img)

        meta = {
            "image": os.path.basename(img_path),
            "img_w": width,
            "img_h": height,
            "style": style,
            "houses": [asdict(h) for h in houses],
            "obstacles": [asdict(o) for o in obstacles],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if j % 50 == 0:
            print(f"[COMPLEX OK] saved {j}/{num_complex} (overall {idx}/1000)")

    print("\nDone ✅")
    print("Images:", out_img_dir)
    print("Meta  :", out_meta_dir)
    print("Output: map_001..map_1000 (first 500 simple, next 500 complex)")


if __name__ == "__main__":
    main()
