from __future__ import annotations
from typing import Tuple, List, Optional, Dict
import heapq
import numpy as np


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors_4(r: int, c: int, grid: np.ndarray):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            yield rr, cc


def astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        return None
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]):
        return None
    if grid[start] == 1 or grid[goal] == 1:
        return None
    if start == goal:
        return [start]

    open_heap: List[Tuple[int, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], int] = {start: 0}

    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cr, cc = current
        for nr, nc in neighbors_4(cr, cc, grid):
            if grid[nr, nc] == 1:
                continue

            nxt = (nr, nc)
            tentative_g = gscore[current] + 1

            if tentative_g < gscore.get(nxt, 10**9):
                came_from[nxt] = current
                gscore[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    return None


def astar_cost(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    penalty: Optional[np.ndarray] = None,
) -> Optional[List[Tuple[int, int]]]:
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        return None
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]):
        return None
    if grid[start] == 1 or grid[goal] == 1:
        return None
    if start == goal:
        return [start]

    if penalty is None:
        penalty = np.zeros_like(grid, dtype=np.float32)

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], float] = {start: 0.0}

    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cr, cc = current
        for nr, nc in neighbors_4(cr, cc, grid):
            if grid[nr, nc] == 1:
                continue

            nxt = (nr, nc)
            step_cost = 1.0 + float(penalty[nr, nc])
            tentative_g = gscore[current] + step_cost

            if tentative_g < gscore.get(nxt, 1e18):
                came_from[nxt] = current
                gscore[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    return None


def _add_penalty_disk(penalty: np.ndarray, cell: Tuple[int, int], radius: int, amount: float):
    r0, c0 = cell
    r1 = max(0, r0 - radius)
    r2 = min(penalty.shape[0] - 1, r0 + radius)
    c1 = max(0, c0 - radius)
    c2 = min(penalty.shape[1] - 1, c0 + radius)

    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if abs(r - r0) + abs(c - c0) <= radius:
                penalty[r, c] += amount


def k_alternative_paths(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    k: int = 5,
    penalty_step: float = 2.0,
    spread_radius: int = 2,
) -> List[List[Tuple[int, int]]]:
    """
    K alternative routes with REAL diversity.

    Key idea:
    After each found path, we penalize:
      - the path cells
      - AND a small radius around the path (spread_radius)
    This forces later routes to move away, so "All Routes" becomes visible as multiple distinct lines.
    """
    paths: List[List[Tuple[int, int]]] = []
    penalty = np.zeros_like(grid, dtype=np.float32)

    for _ in range(max(1, k)):
        p = astar_cost(grid, start, goal, penalty=penalty)
        if p is None:
            break

        if any(p == old for old in paths):
            break

        paths.append(p)

        for cell in p:
            _add_penalty_disk(penalty, cell, radius=spread_radius, amount=penalty_step)

    return paths