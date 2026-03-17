import os
import json
import random
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple

import cv2
import numpy as np


# ------------------- Data structure -------------------
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


def random_rect(w: int, h: int, size_range: Tuple[int, int], margin: int = 10) -> Rect:
    rw = random.randint(size_range[0], size_range[1])
    rh = random.randint(size_range[0], size_range[1])
    x1 = random.randint(margin, w - margin - rw)
    y1 = random.randint(margin, h - margin - rh)
    return Rect(x1, y1, x1 + rw, y1 + rh)


# ------------------- Visual helpers (complicated look) -------------------
def draw_paper_background(img: np.ndarray):
    # paper noise + mild blur (keeps color)
    noise = (np.random.randn(*img.shape[:2], 1) * 4).astype(np.int16)
    base = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[:] = cv2.GaussianBlur(base, (3, 3), 0)


def draw_blueprint_grid(img: np.ndarray, step: int):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (245, 245, 245), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (245, 245, 245), 1)

    # extra micro grid (complexity)
    if random.random() < 0.6:
        micro = step // 2
        for x in range(0, w, micro):
            cv2.line(img, (x, 0), (x, h), (248, 248, 248), 1)
        for y in range(0, h, micro):
            cv2.line(img, (0, y), (w, y), (248, 248, 248), 1)


def draw_roads_complex(img: np.ndarray, n: int = 14) -> List[Rect]:
    """Thicker colored roads with borders (more roads than training)."""
    h, w = img.shape[:2]
    roads: List[Rect] = []

    for _ in range(n):
        thickness = random.choice([10, 14, 18, 22])
        if random.random() < 0.5:
            y = random.randint(40, h - 40)
            cv2.line(img, (0, y), (w, y), (165, 165, 165), thickness=thickness + 6)
            cv2.line(img, (0, y), (w, y), (225, 225, 225), thickness=thickness)
            roads.append(Rect(0, y - (thickness // 2 + 3), w, y + (thickness // 2 + 3)))
        else:
            x = random.randint(40, w - 40)
            cv2.line(img, (x, 0), (x, h), (165, 165, 165), thickness=thickness + 6)
            cv2.line(img, (x, 0), (x, h), (225, 225, 225), thickness=thickness)
            roads.append(Rect(x - (thickness // 2 + 3), 0, x + (thickness // 2 + 3), h))

    return roads


def draw_river_complex(img: np.ndarray):
    """River with branches (harder than training)."""
    h, w = img.shape[:2]

    def polyline(start_y):
        pts = []
        x = 0
        y = start_y
        while x < w:
            pts.append((x, y))
            x += random.randint(60, 130)
            y = max(30, min(h - 30, y + random.randint(-90, 90)))
        return pts

    y0 = random.randint(int(h * 0.2), int(h * 0.8))
    main_pts = polyline(y0)

    # draw main river edge + fill
    for i in range(len(main_pts) - 1):
        cv2.line(img, main_pts[i], main_pts[i + 1], (160, 195, 255), thickness=34)
        cv2.line(img, main_pts[i], main_pts[i + 1], (200, 230, 255), thickness=24)

    # optional branch
    if random.random() < 0.7:
        y1 = max(40, min(h - 40, y0 + random.randint(-160, 160)))
        branch_pts = polyline(y1)
        for i in range(len(branch_pts) - 1):
            cv2.line(img, branch_pts[i], branch_pts[i + 1], (170, 205, 255), thickness=22)
            cv2.line(img, branch_pts[i], branch_pts[i + 1], (210, 235, 255), thickness=14)


def draw_parks_complex(img: np.ndarray, n: int = 5) -> List[Rect]:
    """Green soft areas (not obstacles)."""
    h, w = img.shape[:2]
    parks: List[Rect] = []
    for _ in range(n):
        r = random_rect(w, h, (120, 260), margin=25)
        parks.append(r)
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (205, 245, 205), -1)
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (150, 215, 150), 2)
        # add internal pattern
        if random.random() < 0.8:
            for k in range(8):
                x = random.randint(r.x1 + 10, r.x2 - 10)
                y = random.randint(r.y1 + 10, r.y2 - 10)
                cv2.circle(img, (x, y), random.randint(2, 5), (170, 220, 170), -1)
    return parks


def draw_obstacles_irregular(img: np.ndarray, n: int = 14) -> List[Rect]:
    """Hard obstacles (labels) – more + more irregular than training."""
    h, w = img.shape[:2]
    rects: List[Rect] = []
    attempts = 0

    while len(rects) < n and attempts < 50000:
        attempts += 1
        r = random_rect(w, h, (90, 260), margin=25)
        if any(overlap(r, rr, pad=18) for rr in rects):
            continue
        rects.append(r)

    for r in rects:
        cx = (r.x1 + r.x2) // 2
        cy = (r.y1 + r.y2) // 2
        rx = (r.x2 - r.x1) // 2
        ry = (r.y2 - r.y1) // 2

        pts = []
        k = random.randint(7, 12)
        for i in range(k):
            ang = 2 * math.pi * i / k + random.random() * 0.4
            px = int(cx + rx * math.cos(ang) * random.uniform(0.6, 1.05))
            py = int(cy + ry * math.sin(ang) * random.uniform(0.6, 1.05))
            pts.append([px, py])

        pts_np = np.array([pts], dtype=np.int32)
        cv2.fillPoly(img, pts_np, (60, 60, 60))
        cv2.polylines(img, pts_np, True, (35, 35, 35), 2)

    return rects


def draw_houses_dense(img: np.ndarray, forbidden: List[Rect], n: int = 160) -> List[Rect]:
    """More houses, smaller sizes, clusters (labels)."""
    h, w = img.shape[:2]
    houses: List[Rect] = []
    attempts = 0

    while len(houses) < n and attempts < 120000:
        attempts += 1
        # Smaller houses than training to make it harder
        r = random_rect(w, h, (14, 32), margin=10)

        if any(overlap(r, f, pad=6) for f in forbidden):
            continue
        if any(overlap(r, hh, pad=3) for hh in houses):
            continue
        houses.append(r)

    for r in houses:
        # darker blue blocks
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (55, 55, 150), -1)
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (25, 25, 90), 1)

        # add tiny rooftop lines sometimes (visual complexity only)
        if random.random() < 0.15:
            cv2.line(img, (r.x1, r.y1), (r.x2, r.y2), (30, 30, 90), 1)

    return houses


def add_scan_style(img: np.ndarray) -> np.ndarray:
    """Scan effect (mild) to keep colors visible."""
    h, w = img.shape[:2]
    out = img.astype(np.int16)
    speckle = (np.random.randn(h, w, 1) * 3).astype(np.int16)
    out = np.clip(out + speckle, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    angle = random.uniform(-1.8, 1.8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderValue=(248, 248, 248))
    return out


def draw_legend(img: np.ndarray):
    cv2.rectangle(img, (12, 12), (290, 130), (255, 255, 255), -1)
    cv2.rectangle(img, (22, 28), (50, 56), (55, 55, 150), -1)
    cv2.putText(img, "House", (65, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    cv2.rectangle(img, (22, 78), (50, 106), (60, 60, 60), -1)
    cv2.putText(img, "Obstacle", (65, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    # extra legend lines (complexity)
    cv2.putText(img, "Road/River/Park: context", (22, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)


# ------------------- Main map generator -------------------
def generate_test_map(width: int = 1280, height: int = 720, style: str = "pretty"):
    img = np.ones((height, width, 3), dtype=np.uint8) * 248

    draw_paper_background(img)

    # more grid variety
    step = random.choice([24, 32, 40, 48])
    draw_blueprint_grid(img, step=step)

    # heavy context elements (not labeled)
    draw_roads_complex(img, n=random.randint(12, 16))
    draw_river_complex(img)
    parks = draw_parks_complex(img, n=random.randint(4, 6))

    # hard obstacles (labeled)
    obstacles = draw_obstacles_irregular(img, n=random.randint(12, 18))

    # forbidden regions for houses
    forbidden = obstacles + parks

    # many houses (labeled)
    houses = draw_houses_dense(img, forbidden=forbidden, n=random.randint(140, 200))

    draw_legend(img)

    # make it look different from training distribution
    if style == "scan":
        img = add_scan_style(img)

    return img, houses, obstacles


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    out_img_dir = os.path.join(project_root, "data", "synthetic_test_maps")
    out_meta_dir = os.path.join(project_root, "data", "meta_test")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_meta_dir, exist_ok=True)

    num_maps = 250
    width, height = 1280, 720

    # ✅ different seed => truly unseen maps
    random.seed(2026)
    np.random.seed(2026)

    for i in range(1, num_maps + 1):
        style = "scan" if random.random() < 0.55 else "pretty"
        img, houses, obstacles = generate_test_map(width, height, style=style)

        name = f"test_{i:03d}"
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

        if i % 25 == 0:
            print(f"[TEST OK] saved {i}/{num_maps}")

    print("\n✅ Done generating EXTREMELY complicated unseen TEST maps!")
    print("✅ Images:", out_img_dir)
    print("✅ Meta  :", out_meta_dir)


if __name__ == "__main__":
    main()
