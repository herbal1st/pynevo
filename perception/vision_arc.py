"""
Amanatides-Woo fast voxel traversal raycaster for wall proximity sensing.
"""

import math
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from core.map_data import MapData


@njit(fastmath=True, cache=True)
def cast_single_ray_jit(
    ox: float,
    oy: float,
    angle_rad: float,
    max_dist: float,
    grid_array: NDArray[np.uint8],
    map_width: int,
    map_height: int
) -> Tuple[float, float]:
    """
    Fast DDA raycast stepping in compiled C code.
    """
    dir_x = math.cos(angle_rad)
    dir_y = math.sin(angle_rad)

    eps = 1e-9
    if abs(dir_x) < eps:
        dir_x = eps if dir_x >= 0.0 else -eps
    if abs(dir_y) < eps:
        dir_y = eps if dir_y >= 0.0 else -eps

    tx = int(math.floor(ox))
    ty = int(math.floor(oy))

    if tx < 0 or tx >= map_width or ty < 0 or ty >= map_height:
        return 1.0, 0.0

    if grid_array[ty, tx] == 1:
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

        if tx < 0 or tx >= map_width or ty < 0 or ty >= map_height:
            hit_dist = min(current_dist, max_dist)
            wall_prox = 1.0 - (hit_dist / max_dist)
            return max(0.0, float(wall_prox)), float(hit_dist)

        if grid_array[ty, tx] == 1:
            hit_dist = min(current_dist, max_dist)
            wall_prox = 1.0 - (hit_dist / max_dist)
            return max(0.0, float(wall_prox)), float(hit_dist)

    return 0.0, float(max_dist)


@njit(fastmath=True, cache=True)
def sample_vision_arc_jit(
    origin_x: float,
    origin_y: float,
    heading_rad: float,
    relative_angles: NDArray[np.float64],
    max_dist: float,
    grid_array: NDArray[np.uint8],
    map_width: int,
    map_height: int
) -> NDArray[np.float32]:
    """
    Samples full arc across pre-allocated NumPy angles.
    """
    n_rays = len(relative_angles)
    channels = np.empty(n_rays, dtype=np.float32)
    for i in range(n_rays):
        ray_angle = heading_rad + relative_angles[i]
        wall_prox, _ = cast_single_ray_jit(
            origin_x, origin_y, ray_angle, max_dist, grid_array, map_width, map_height
        )
        channels[i] = np.float32(wall_prox)
    return channels


@njit(parallel=True, fastmath=True, cache=True)
def batch_sample_vision_arc_jit(
    origins_x: NDArray[np.float64],
    origins_y: NDArray[np.float64],
    headings: NDArray[np.float64],
    relative_angles: NDArray[np.float64],
    max_dist: float,
    grid_array: NDArray[np.uint8],
    map_width: int,
    map_height: int
) -> NDArray[np.float32]:
    """
    Parallel multi-agent raycast sweep across all available CPU threads.
    """
    n_agents = len(origins_x)
    n_rays = len(relative_angles)
    out = np.empty((n_agents, n_rays), dtype=np.float32)
    for a in prange(n_agents):
        for r in range(n_rays):
            ray_angle = headings[a] + relative_angles[r]
            wall_prox, _ = cast_single_ray_jit(
                origins_x[a], origins_y[a], ray_angle, max_dist, grid_array, map_width, map_height
            )
            out[a, r] = np.float32(wall_prox)
    return out


class VisionArcSampler:
    """
    Casts probe rays measuring wall tile proximity across visual arc.
    """

    def __init__(
        self,
        num_rays: int = 13,
        arc_angle_deg: float = 240.0,
        max_dist: float = 6.0
    ) -> None:
        """
        Pre-allocates SIMD relative ray angles using np.linspace.
        """
        self.num_rays: int = num_rays
        self.max_dist: float = max_dist

        half_arc: float = math.radians(arc_angle_deg / 2.0)
        if num_rays > 1:
            self.relative_angles: NDArray[np.float64] = np.linspace(
                -half_arc, half_arc, num_rays, dtype=np.float64
            )
        else:
            self.relative_angles = np.array([0.0], dtype=np.float64)

    def sample_vision_channels(
        self,
        origin_x: float,
        origin_y: float,
        heading_rad: float,
        map_data: MapData
    ) -> NDArray[np.float32]:
        """
        Casts probe rays using compiled JIT kernels when grid_array is present.
        """
        if hasattr(map_data, "grid_array"):
            return sample_vision_arc_jit(
                origin_x,
                origin_y,
                heading_rad,
                self.relative_angles,
                self.max_dist,
                map_data.grid_array,
                map_data.width,
                map_data.height
            )

        channels: NDArray[np.float32] = np.zeros(
            self.num_rays, dtype=np.float32
        )
        for i, rel_angle in enumerate(self.relative_angles):
            ray_angle: float = heading_rad + float(rel_angle)
            wall_prox, _ = self._cast_single_ray(
                origin_x, origin_y, ray_angle, map_data
            )
            channels[i] = wall_prox

        return channels

    def _cast_single_ray(
        self,
        ox: float,
        oy: float,
        angle_rad: float,
        map_data: MapData
    ) -> Tuple[float, float]:
        """
        Amanatides-Woo fast grid traversal raycast to wall boundary.
        """
        if hasattr(map_data, "grid_array"):
            return cast_single_ray_jit(
                ox, oy, angle_rad, self.max_dist,
                map_data.grid_array, map_data.width, map_data.height
            )

        dir_x: float = math.cos(angle_rad)
        dir_y: float = math.sin(angle_rad)

        eps: float = 1e-9
        if abs(dir_x) < eps:
            dir_x = eps if dir_x >= 0.0 else -eps
        if abs(dir_y) < eps:
            dir_y = eps if dir_y >= 0.0 else -eps

        tx: int = int(math.floor(ox))
        ty: int = int(math.floor(oy))

        if map_data.is_wall(tx, ty):
            return 1.0, 0.0

        step_x: int = 1 if dir_x > 0.0 else -1
        step_y: int = 1 if dir_y > 0.0 else -1

        t_delta_x: float = abs(1.0 / dir_x)
        t_delta_y: float = abs(1.0 / dir_y)

        if dir_x > 0.0:
            t_max_x: float = (float(tx + 1) - ox) * t_delta_x
        else:
            t_max_x = (ox - float(tx)) * t_delta_x

        if dir_y > 0.0:
            t_max_y: float = (float(ty + 1) - oy) * t_delta_y
        else:
            t_max_y = (oy - float(ty)) * t_delta_y

        current_dist: float = 0.0

        while current_dist < self.max_dist:
            if t_max_x < t_max_y:
                current_dist = t_max_x
                t_max_x += t_delta_x
                tx += step_x
            else:
                current_dist = t_max_y
                t_max_y += t_delta_y
                ty += step_y

            if map_data.is_wall(tx, ty):
                hit_dist: float = min(current_dist, self.max_dist)
                wall_prox: float = 1.0 - (hit_dist / self.max_dist)
                return max(0.0, float(wall_prox)), float(hit_dist)

        return 0.0, float(self.max_dist)