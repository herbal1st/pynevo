"""
Numba JIT-accelerated core computational kernels.
"""

import math
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True, cache=True)
def cast_single_ray_jit(
    ox: float,
    oy: float,
    angle_rad: float,
    grid: np.ndarray,      # 2D uint8 array
    width: int,
    height: int,
    max_dist: float
) -> tuple[float, float]:
    """Amanatides-Woo fast grid traversal compiled with LLVM."""
    dir_x = math.cos(angle_rad)
    dir_y = math.sin(angle_rad)

    eps = 1e-9
    if abs(dir_x) < eps:
        dir_x = eps if dir_x >= 0.0 else -eps
    if abs(dir_y) < eps:
        dir_y = eps if dir_y >= 0.0 else -eps

    tx = int(math.floor(ox))
    ty = int(math.floor(oy))

    if tx < 0 or tx >= width or ty < 0 or ty >= height or grid[ty, tx] == 1:
        return 1.0, 0.0

    step_x = 1 if dir_x > 0.0 else -1
    step_y = 1 if dir_y > 0.0 else -1

    t_delta_x = abs(1.0 / dir_x)
    t_delta_y = abs(1.0 / dir_y)

    if dir_x > 0.0:
        t_max_x = (float(tx + 1) - ox) * t_delta_x
    else:
        t_max_x = (ox - float(tx)) * t_delta_x

    if dir_y > 0.0:
        t_max_y = (float(ty + 1) - oy) * t_delta_y
    else:
        t_max_y = (oy - float(ty)) * t_delta_y

    current_dist = 0.0
    while current_dist < max_dist:
        if t_max_x < t_max_y:
            current_dist = t_max_x
            t_max_x += t_delta_x
            tx += step_x
        else:
            current_dist = t_max_y
            t_max_y += t_delta_y
            ty += step_y

        if tx < 0 or tx >= width or ty < 0 or ty >= height or grid[ty, tx] == 1:
            hit_dist = min(current_dist, max_dist)
            wall_prox = 1.0 - (hit_dist / max_dist)
            return max(0.0, float(wall_prox)), float(hit_dist)

    return 0.0, float(max_dist)


@njit(fastmath=True, cache=True)
def resolve_circle_aabb_jit(
    px: float,
    py: float,
    radius: float,
    grid: np.ndarray,
    width: int,
    height: int,
    passes: int = 2
) -> tuple[float, float, bool]:
    """Circle-to-AABB multi-pass penetration resolution compiled with LLVM."""
    r = radius
    has_collided = False

    min_x = r
    max_x = float(width) - r
    min_y = r
    max_y = float(height) - r

    if px < min_x or px > max_x or py < min_y or py > max_y:
        has_collided = True
        px = max(min_x, min(max_x, px))
        py = max(min_y, min(max_y, py))

    r_sq = r * r

    for _ in range(passes):
        min_tx = max(0, int(math.floor(px - r)))
        max_tx = min(width - 1, int(math.floor(px + r)))
        min_ty = max(0, int(math.floor(py - r)))
        max_ty = min(height - 1, int(math.floor(py + r)))

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if grid[ty, tx] == 0:
                    continue

                cx = max(float(tx), min(px, float(tx) + 1.0))
                cy = max(float(ty), min(py, float(ty) + 1.0))

                dx = px - cx
                dy = py - cy
                dist_sq = (dx * dx) + (dy * dy)

                if dist_sq < r_sq:
                    has_collided = True
                    dist = math.sqrt(dist_sq)

                    if dist > 1e-6:
                        overlap = r - dist
                        px += (dx / dist) * overlap
                        py += (dy / dist) * overlap
                    else:
                        tile_cx = float(tx) + 0.5
                        tile_cy = float(ty) + 0.5
                        push_x = 1.0 if px >= tile_cx else -1.0
                        push_y = 1.0 if py >= tile_cy else -1.0

                        if abs(px - tile_cx) < abs(py - tile_cy):
                            py = float(ty + 1) + r if push_y > 0.0 else float(ty) - r
                        else:
                            px = float(tx + 1) + r if push_x > 0.0 else float(tx) - r

    px = max(min_x, min(max_x, px))
    py = max(min_y, min(max_y, py))

    return px, py, has_collided


@njit(fastmath=True, cache=True)
def get_bilinear_bfs_dist_jit(
    cx: float,
    cy: float,
    dist_grid: np.ndarray,
    width: int,
    height: int,
    cand_x: float,
    cand_y: float
) -> float:
    """Bilinear BFS distance interpolation compiled with LLVM."""
    u = cx - 0.5
    v = cy - 0.5

    x0 = int(math.floor(u))
    y0 = int(math.floor(v))
    x1 = x0 + 1
    y1 = y0 + 1

    uf = u - float(x0)
    vf = v - float(y0)

    fb_tx = int(math.floor(cand_x))
    fb_ty = int(math.floor(cand_y))

    fallback_d = 9999.0
    if 0 <= fb_tx < width and 0 <= fb_ty < height:
        fallback_d = float(dist_grid[fb_ty, fb_tx])

    if fallback_d >= 9999.0:
        return 9999.0

    def sample_d(tx: int, ty: int) -> float:
        if 0 <= tx < width and 0 <= ty < height:
            d = dist_grid[ty, tx]
            if d < 9999:
                return float(d)
        return fallback_d

    d00 = sample_d(x0, y0)
    d10 = sample_d(x1, y0)
    d01 = sample_d(x0, y1)
    d11 = sample_d(x1, y1)

    return (
        (1.0 - uf) * (1.0 - vf) * d00
        + uf * (1.0 - vf) * d10
        + (1.0 - uf) * vf * d01
        + uf * vf * d11
    )
