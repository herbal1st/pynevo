"""
Candidate continuous movement physics and Circle-to-AABB wall collisions.
"""

import math
from typing import Tuple, Optional

import config
from core.map_data import MapData
from core.kinematics.profiles import (
    KinematicsProfile,
    CarProfile,
    TankProfile
)
from entities.map_profile_registry import MapProfileRegistry


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
        Resolves circle penetration against surrounding wall tile AABBs.
        """
        r: float = self.radius
        has_collided: bool = False

        min_x: float = r
        max_x: float = float(map_data.width) - r
        min_y: float = r
        max_y: float = float(map_data.height) - r

        if px < min_x or px > max_x or py < min_y or py > max_y:
            has_collided = True
            px = max(min_x, min(max_x, px))
            py = max(min_y, min(max_y, py))

        for _ in range(passes):
            min_tx: int = max(0, int(math.floor(px - r)))
            max_tx: int = min(
                map_data.width - 1, int(math.floor(px + r))
            )
            min_ty: int = max(0, int(math.floor(py - r)))
            max_ty: int = min(
                map_data.height - 1, int(math.floor(py + r))
            )

            for ty in range(min_ty, max_ty + 1):
                for tx in range(min_tx, max_tx + 1):
                    if not map_data.is_wall(tx, ty):
                        continue

                    cx: float = max(
                        float(tx), min(px, float(tx) + 1.0)
                    )
                    cy: float = max(
                        float(ty), min(py, float(ty) + 1.0)
                    )

                    dx: float = px - cx
                    dy: float = py - cy
                    dist_sq: float = (dx * dx) + (dy * dy)

                    if dist_sq < (r * r):
                        has_collided = True
                        dist: float = math.sqrt(dist_sq)

                        if dist > 1e-6:
                            overlap: float = r - dist
                            nx_dir: float = dx / dist
                            ny_dir: float = dy / dist
                            px += nx_dir * overlap
                            py += ny_dir * overlap
                        else:
                            tile_cx: float = float(tx) + 0.5
                            tile_cy: float = float(ty) + 0.5
                            push_x: float = (
                                1.0 if px >= tile_cx else -1.0
                            )
                            push_y: float = (
                                1.0 if py >= tile_cy else -1.0
                            )

                            if abs(px - tile_cx) < abs(py - tile_cy):
                                py = (
                                    float(ty + 1) + r if push_y > 0.0
                                    else float(ty) - r
                                )
                            else:
                                px = (
                                    float(tx + 1) + r if push_x > 0.0
                                    else float(tx) - r
                                )

        px = max(min_x, min(max_x, px))
        py = max(min_y, min(max_y, py))

        return px, py, has_collided
