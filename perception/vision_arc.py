"""
Vectorized 2D DDA raycaster for wall proximity sensing.
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
        DDA grid-line intersection marching to exact wall tile boundary.
        """
        dir_x: float = math.cos(angle_rad)
        dir_y: float = math.sin(angle_rad)

        if abs(dir_x) < 1e-9:
            dir_x = 1e-9
        if abs(dir_y) < 1e-9:
            dir_y = 1e-9

        tx: int = int(math.floor(ox))
        ty: int = int(math.floor(oy))

        delta_tx: float = abs(1.0 / dir_x)
        delta_ty: float = abs(1.0 / dir_y)

        step_x: int = 1 if dir_x > 0 else -1
        step_y: int = 1 if dir_y > 0 else -1

        if dir_x > 0:
            side_tx: float = ((tx + 1.0) - ox) * delta_tx
        else:
            side_tx = (ox - float(tx)) * delta_tx

        if dir_y > 0:
            side_ty: float = ((ty + 1.0) - oy) * delta_ty
        else:
            side_ty = (oy - float(ty)) * delta_ty

        dist: float = 0.0

        while dist < self.max_dist:
            if side_tx < side_ty:
                dist = side_tx
                side_tx += delta_tx
                tx += step_x
            else:
                dist = side_ty
                side_ty += delta_ty
                ty += step_y

            if map_data.is_wall(tx, ty):
                hit_dist: float = min(dist, self.max_dist)
                wall_prox: float = 1.0 - (hit_dist / self.max_dist)
                return max(0.0, wall_prox), 0.0

        return 0.0, 0.0
