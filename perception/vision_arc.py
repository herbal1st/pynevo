"""
Amanatides-Woo fast voxel traversal raycaster for wall proximity sensing.
"""

import math
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData
from core.accelerated import cast_single_ray_jit


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
        Amanatides-Woo fast grid traversal raycast to wall boundary via Numba.
        """
        if not hasattr(map_data, "numpy_grid") or map_data.numpy_grid is None:
            map_data.numpy_grid = np.array(map_data.grid, dtype=np.uint8)

        return cast_single_ray_jit(
            ox,
            oy,
            angle_rad,
            map_data.numpy_grid,
            map_data.width,
            map_data.height,
            self.max_dist
        )
