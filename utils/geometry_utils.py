"""
Spatial geometry utilities and continuous ray clearance math.
"""

import math
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.map_data import MapData


def check_ray_wall_clearance(
    start_pt: Tuple[float, float],
    end_pt: Tuple[float, float],
    map_data: "MapData",
    eff_radius: float,
    samples: int = 12
) -> bool:
    """
    Checks if a line segment with body clearance clears all wall AABBs.
    """
    sx, sy = start_pt
    ex, ey = end_pt
    dx: float = ex - sx
    dy: float = ey - sy
    dist: float = math.sqrt((dx * dx) + (dy * dy))

    if dist < 1e-4:
        return True

    r_sq: float = eff_radius * eff_radius
    num_steps: int = max(samples, int(dist / 0.15))

    for step in range(num_steps + 1):
        t: float = float(step) / float(num_steps)
        px: float = sx + (t * dx)
        py: float = sy + (t * dy)

        min_tx: int = max(0, int(math.floor(px - eff_radius)))
        max_tx: int = min(
            map_data.width - 1, int(math.floor(px + eff_radius))
        )
        min_ty: int = max(0, int(math.floor(py - eff_radius)))
        max_ty: int = min(
            map_data.height - 1, int(math.floor(py + eff_radius))
        )

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if not map_data.is_wall(tx, ty):
                    continue

                cx: float = max(float(tx), min(px, float(tx) + 1.0))
                cy: float = max(float(ty), min(py, float(ty) + 1.0))

                dist_sq: float = (px - cx) ** 2 + (py - cy) ** 2
                if dist_sq < r_sq:
                    return False

    return True
