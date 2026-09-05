"""
JIT kernel warmup routines to pre-compile Numba functions before simulation loop.
"""

import numpy as np
from core.kinematics.engine import (
    resolve_circle_aabb_jit,
    batch_resolve_circle_aabb_jit
)
from core.kinematics.endless_engine import resolve_endless_circle_aabb_jit
from perception.vision_arc import (
    cast_single_ray_jit,
    sample_vision_arc_jit,
    batch_sample_vision_arc_jit
)


def warmup_jit() -> None:
    """
    Executes dummy JIT compilation passes to eliminate runtime frame stutters.
    """
    dummy_grid = np.zeros((10, 10), dtype=np.uint8)
    dummy_grid[0, :] = 1
    dummy_grid[-1, :] = 1
    dummy_grid[:, 0] = 1
    dummy_grid[:, -1] = 1

    # 1. Warmup collision kernels
    resolve_circle_aabb_jit(5.0, 5.0, 0.25, 10, 10, dummy_grid, 2)
    batch_resolve_circle_aabb_jit(
        np.array([5.0, 6.0], dtype=np.float64),
        np.array([5.0, 6.0], dtype=np.float64),
        0.25, 10, 10, dummy_grid, 2
    )

    # 2. Warmup endless collision kernel
    stx = np.array([0, 1], dtype=np.int32)
    sty = np.array([0, 0], dtype=np.int32)
    resolve_endless_circle_aabb_jit(0.5, 0.5, 0.25, stx, sty, 2)

    # 3. Warmup raycasting kernels
    angles = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    cast_single_ray_jit(5.0, 5.0, 0.0, 5.0, dummy_grid, 10, 10)
    sample_vision_arc_jit(5.0, 5.0, 0.0, angles, 5.0, dummy_grid, 10, 10)
    batch_sample_vision_arc_jit(
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        angles, 5.0, dummy_grid, 10, 10
    )