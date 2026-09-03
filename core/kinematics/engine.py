"""
Candidate continuous movement physics and Circle-to-AABB wall collisions.
"""

import math
from typing import Tuple, Optional
import numpy as np

import config
from core.map_data import MapData
from core.kinematics.profiles import (
    KinematicsProfile,
    CarProfile,
    TankProfile
)
from entities.map_profile_registry import MapProfileRegistry
from core.accelerated import resolve_circle_aabb_jit


class CandidateKinematics:
    """
    Handles 2D movement physics and Circle-to-AABB penetration resolution.
    """

    def __init__(
        self,
        move_speed: float = 0.15,
        turn_speed_dpsec: float = 1800.0,
        agent_diameter_ratio: float = 0.45,
        fps: int = config.FPS,
        profile_style: str = "TANK"
    ) -> None:
        """
        Initializes movement constants and binds active steering profile.
        """
        self.move_speed: float = move_speed
        self.radius: float = 0.5 * agent_diameter_ratio
        self.rad_per_frame: float = (
            math.radians(turn_speed_dpsec) / float(fps)
        )
        if profile_style.upper() == "TANK":
            self.profile: KinematicsProfile = TankProfile()
        else:
            self.profile = CarProfile()

    def apply_rotation(
        self,
        heading_rad: float,
        turn_effort: float,
        move_effort: float = 1.0
    ) -> Tuple[float, bool]:
        """
        Delegates rotational calculation to active steering profile.
        """
        return self.profile.calculate_rotation(
            heading_rad, turn_effort, move_effort, self.rad_per_frame
        )

    def calculate_forward_step(
        self,
        curr_x: float,
        curr_y: float,
        heading_rad: float,
        move_effort: float,
        map_data: MapData
    ) -> Tuple[float, float, bool]:
        """
        Calculates step and resolves Circle-to-AABB penetration pushback.
        """
        clamped_effort: float = max(-1.0, min(1.0, move_effort))
        if abs(clamped_effort) < 1e-4:
            return curr_x, curr_y, False

        step_dist: float = clamped_effort * self.move_speed
        next_x: float = curr_x + (math.cos(heading_rad) * step_dist)
        next_y: float = curr_y + (math.sin(heading_rad) * step_dist)

        resolved_x, resolved_y, hit = self._resolve_circle_aabb(
            next_x, next_y, map_data
        )
        return resolved_x, resolved_y, hit

    def interpolate_pixel_pos(
        self,
        tile_x: float,
        tile_y: float,
        tile_size: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Converts continuous tile coordinates to screen pixel positions.
        """
        if tile_size is None:
            tile_size = MapProfileRegistry().get_profile(
                config.ACTIVE_MAP_PROFILE
            ).tile_size

        pixel_x: int = int(round(tile_x * float(tile_size)))
        pixel_y: int = int(round(tile_y * float(tile_size)))
        return pixel_x, pixel_y

    def _resolve_circle_aabb(
        self,
        px: float,
        py: float,
        map_data: MapData,
        passes: int = 2
    ) -> Tuple[float, float, bool]:
        """
        Resolves circle penetration against surrounding wall tile AABBs via Numba.
        """
        if not hasattr(map_data, "numpy_grid") or map_data.numpy_grid is None:
            map_data.numpy_grid = np.array(map_data.grid, dtype=np.uint8)

        return resolve_circle_aabb_jit(
            px,
            py,
            self.radius,
            map_data.numpy_grid,
            map_data.width,
            map_data.height,
            passes
        )
