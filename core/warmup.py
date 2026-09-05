"""
JIT kernel warmup routines to pre-compile Numba functions before simulation loop.
"""

import numpy as np
from core.kinematics.engine import (
    resolve_circle_aabb_jit,
    batch_resolve_circle_aabb_jit
)
from core.kinematics.endless_engine import resolve_endless_circle_aabb_jit
from core.map_data import march_los_segment_jit
from perception.exit_compass import check_5point_los_jit
from perception.vision_arc import (
    cast_single_ray_jit,
    sample_vision_arc_jit,
    batch_sample_vision_arc_jit
)
from perception.spatial.gps_sensor import (
    get_bilinear_bfs_distance,
    compute_stereo_gps_jit
)
from core.pathfinder import lookup_step_distance_jit, get_step_distance


def warmup_jit() -> None:
    """
    Executes dummy JIT compilation passes to eliminate runtime frame stutters.
    """
    dummy_grid = np.zeros((10, 10), dtype=np.uint8)
    dummy_grid[0, :] = 1
    dummy_grid[-1, :] = 1
    dummy_grid[:, 0] = 1
    dummy_grid[:, -1] = 1

    dummy_int32 = np.zeros((2, 10, 10), dtype=np.int32)

    # 1. Collision kernels
    resolve_circle_aabb_jit(5.0, 5.0, 0.25, 10, 10, dummy_grid, 2)
    batch_resolve_circle_aabb_jit(
        np.array([5.0, 6.0], dtype=np.float64),
        np.array([5.0, 6.0], dtype=np.float64),
        0.25, 10, 10, dummy_grid, 2
    )

    stx = np.array([0, 1], dtype=np.int32)
    sty = np.array([0, 0], dtype=np.int32)
    resolve_endless_circle_aabb_jit(0.5, 0.5, 0.25, stx, sty, 2)

    # 2. Raycasting kernels
    angles = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    cast_single_ray_jit(5.0, 5.0, 0.0, 5.0, dummy_grid, 10, 10)
    sample_vision_arc_jit(5.0, 5.0, 0.0, angles, 5.0, dummy_grid, 10, 10)
    batch_sample_vision_arc_jit(
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([5.0, 5.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        angles, 5.0, dummy_grid, 10, 10
    )

    # 3. Line-of-sight kernels
    march_los_segment_jit(5.0, 5.0, 7.0, 7.0, dummy_grid, 10, 10, 0.2)
    check_5point_los_jit(5.0, 5.0, 7, 7, dummy_grid, 10, 10)

    # 4. GPS and Pathfinder distance kernels
    get_bilinear_bfs_distance(dummy_int32[0], 5.0, 5.0, 10, 10)
    compute_stereo_gps_jit(
        5.0, 5.0, 0.0, 0.39, 0.25, dummy_int32[0], 10, 10, 9998.0, 10.0, 10.0, 0.125, True
    )
    lookup_step_distance_jit(dummy_int32, 5, 5, 0, 10, 10, 2, 9999)
    get_step_distance(1.0, 1.0, 4.0, 5.0)