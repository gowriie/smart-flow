from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np


def fuzzy_score(path: List[Tuple[int, int]], grid: np.ndarray) -> Dict[str, Any]:
    """
    Explainable fuzzy-like scoring for a simulated pipeline route.

    Inputs:
      - path: list of (row, col) cells (A* output)
      - grid: 2D numpy array where 1 indicates obstacle, 0 indicates free cell

    Scoring idea (simple + explainable):
      - Shorter path => better pressure efficiency (less loss)
      - Closer to obstacles => higher risk and higher cost
      - A weighted fuzzy-style combination produces final scores
    """
    if not path:
        return {
            "ok": False,
            "reason": "No path",
            "path_cells": 0,
            "risk_cells": 0,
            "cost_score": 1.0,
            "pressure_efficiency": 0.0,
            "safety_score": 0.0,
            "explanation": "No valid route could be generated."
        }

    length = len(path)

    # Count "risky" cells: any obstacle within 1-cell radius around the route cell
    risk = 0
    rows, cols = grid.shape[:2]
    for r, c in path:
        r1, r2 = max(0, r - 1), min(rows - 1, r + 1)
        c1, c2 = max(0, c - 1), min(cols - 1, c + 1)
        if np.any(grid[r1:r2 + 1, c1:c2 + 1] == 1):
            risk += 1

    # Normalize (kept stable for your dataset sizes)
    # If you later increase map sizes, you can adjust 600 accordingly.
    length_norm = min(1.0, length / 600.0)
    risk_norm = min(1.0, risk / max(1, length))  # ratio of risky cells

    # Fuzzy-style scoring rules (simple weights)
    # - cost increases with both length and risk
    # - pressure efficiency mainly decreases with length (longer route => more loss)
    # - safety decreases with risk
    cost = 0.6 * length_norm + 0.4 * risk_norm
    pressure_eff = 1.0 - length_norm
    safety = 1.0 - risk_norm

    # Clean rounding for UI display
    cost = round(float(cost), 3)
    pressure_eff = round(float(pressure_eff), 3)
    safety = round(float(safety), 3)

    explanation = (
        "Fuzzy Evaluation Summary:\n"
        "• Cost score increases when the pipeline route is longer or passes close to obstacles.\n"
        "• Pressure efficiency improves when the route is shorter (reduced pressure loss).\n"
        "• Safety score improves when the route stays farther away from obstacles.\n"
        "These scores help explain the routing decision in a transparent way."
    )

    return {
        "ok": True,
        "path_cells": int(length),
        "risk_cells": int(risk),
        "length_norm": round(float(length_norm), 3),
        "risk_norm": round(float(risk_norm), 3),
        "cost_score": cost,                     # lower is better
        "pressure_efficiency": pressure_eff,     # higher is better
        "safety_score": safety,                 # higher is better
        "explanation": explanation,
    }