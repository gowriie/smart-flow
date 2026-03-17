from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import deque

import numpy as np


@dataclass
class GridBuildResult:
    grid: np.ndarray              # 0 = free, 1 = blocked
    cell_size: int                # pixels per grid cell
    img_w: int
    img_h: int


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def box_to_grid_rect(
    box_xyxy,
    cell_size: int,
    img_w: int,
    img_h: int,
    padding_px: int = 0
) -> Tuple[int, int, int, int]:
    """
    Converts an xyxy pixel box into a (gx1, gy1, gx2, gy2) grid rect.
    gx = col, gy = row
    """
    x1, y1, x2, y2 = box_xyxy

    # clamp in pixel space (with padding)
    x1 = clamp(int(x1 - padding_px), 0, img_w - 1)
    y1 = clamp(int(y1 - padding_px), 0, img_h - 1)
    x2 = clamp(int(x2 + padding_px), 0, img_w - 1)
    y2 = clamp(int(y2 + padding_px), 0, img_h - 1)

    # convert to grid coords (col,row)
    gx1 = x1 // cell_size
    gy1 = y1 // cell_size
    gx2 = x2 // cell_size
    gy2 = y2 // cell_size

    return gx1, gy1, gx2, gy2


def build_grid_from_detections(
    img_w: int,
    img_h: int,
    detections: Dict[str, List[dict]],
    cell_size: int = 8,
    obstacle_padding_px: int = 6,
) -> GridBuildResult:
    """
    Build occupancy grid from detections.
    grid: 0 = free, 1 = blocked (obstacle)

    detections dict:
      {
        "houses": [{"x1","y1","x2","y2","conf"}, ...],
        "obstacles": [{"x1","y1","x2","y2","conf"}, ...]
      }
    """
    gw = int(np.ceil(img_w / cell_size))
    gh = int(np.ceil(img_h / cell_size))
    grid = np.zeros((gh, gw), dtype=np.uint8)

    # Mark obstacles as blocked
    for o in detections.get("obstacles", []):
        gx1, gy1, gx2, gy2 = box_to_grid_rect(
            (o["x1"], o["y1"], o["x2"], o["y2"]),
            cell_size=cell_size,
            img_w=img_w,
            img_h=img_h,
            padding_px=obstacle_padding_px
        )

        # clamp in grid space
        gx1 = clamp(gx1, 0, gw - 1)
        gx2 = clamp(gx2, 0, gw - 1)
        gy1 = clamp(gy1, 0, gh - 1)
        gy2 = clamp(gy2, 0, gh - 1)

        grid[gy1:gy2 + 1, gx1:gx2 + 1] = 1
        

    return GridBuildResult(grid=grid, cell_size=cell_size, img_w=img_w, img_h=img_h)


def pixel_to_cell(x: int, y: int, cell_size: int) -> Tuple[int, int]:
    """pixel -> grid cell (row, col)"""
    return (y // cell_size, x // cell_size)


def cell_to_pixel_center(r: int, c: int, cell_size: int) -> Tuple[int, int]:
    """grid cell center -> pixel point"""
    x = c * cell_size + cell_size // 2
    y = r * cell_size + cell_size // 2
    return x, y


def is_cell_blocked(grid: np.ndarray, cell: Tuple[int, int]) -> bool:
    r, c = cell
    if r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1]:
        return True
    return grid[r, c] == 1


def nearest_free_cell(grid: np.ndarray, start: Tuple[int, int], max_radius: int = 50) -> Tuple[int, int]:
    """
    If start cell falls inside an obstacle, find nearest free cell using BFS.
    This is more correct than scanning squares (always returns closest by steps).

    max_radius limits search depth to keep it fast.
    """
    sr, sc = start

    # if already valid and free
    if 0 <= sr < grid.shape[0] and 0 <= sc < grid.shape[1] and grid[sr, sc] == 0:
        return (sr, sc)

    q = deque([(sr, sc, 0)])
    visited = {(sr, sc)}

    # 8-neighborhood helps find near free cell quickly
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while q:
        r, c, d = q.popleft()
        if d > max_radius:
            break

        for dr, dc in dirs:
            rr, cc = r + dr, c + dc
            if (rr, cc) in visited:
                continue
            visited.add((rr, cc))

            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if grid[rr, cc] == 0:
                    return (rr, cc)
                q.append((rr, cc, d + 1))

    # fallback (routing may fail but no crash)
    return start