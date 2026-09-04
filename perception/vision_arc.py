"""
Amanatides-Woo fast voxel traversal raycaster for wall proximity sensing.
"""

import math
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData


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
        Initializes relative ray angles across the visual arc.
        """
        self.num_rays: int = num_rays
        self.max_dist: float = max_dist

        half_arc: float = math.radians(arc_angle_deg / 2.0)
        if num_rays > 1:
            step: float = (2.0 * half_arc) / float(num_rays - 1)
            self.relative_angles: List[float] = [
                -half_arc + (i * step) for i in range(num_rays)
            ]
        else:
            self.relative_angles = [0.0]

    def sample_vision_channels(
        self,
        origin_x: float,
        origin_y: float,
        heading_rad: float,
        map_data: MapData
    ) -> NDArray[np.float32]:
        """
        Casts probe rays and returns a (VISION_RAYS,) wall proximity array.
        """
        channels: NDArray[np.float32] = np.zeros(
            self.num_rays, dtype=np.float32
        )

        for i, rel_angle in enumerate(self.relative_angles):
            ray_angle: float = heading_rad + rel_angle
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
