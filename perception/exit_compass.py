"""
Stereo binocular target compass sensor with profile-driven bounds.
"""

import math
from typing import Tuple, Optional, List

from core.map_data import MapData
from entities.agent_profile_registry import ResolvedAgentProfile
from utils.math_utils import (
    calculate_angle_delta,
    calculate_euclidean_distance
)


class ExitCompass:
    """
    Computes 4-channel Focus/Peripheral Exit Lock Radar signals.
    """

    def compute_stereo_channels(
        self,
        candidate_x: float,
        candidate_y: float,
        heading_rad: float,
        map_data: MapData,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> Tuple[float, float, float, float]:
        """
        Calculates Exit eye intensities [0.0, 1.0] across focus & periphere.
        """
        if profile is not None and not profile.activate_exit_compass:
            return 0.0, 0.0, 0.0, 0.0

        ex_tile, ey_tile = map_data.exit_pos
        ex_center: float = float(ex_tile) + 0.5
        ey_center: float = float(ey_tile) + 0.5

        dist: float = calculate_euclidean_distance(
            candidate_x, candidate_y, ex_center, ey_center
        )

        max_range: float = (
            profile.range_exit_compass if profile is not None else 20.0
        )
        if dist > max_range:
            return 0.0, 0.0, 0.0, 0.0

        use_los_gating: bool = (
            profile.exit_compass_los_gating
            if profile is not None else True
        )

        if use_los_gating:
            tile_x: int = int(math.floor(candidate_x))
            tile_y: int = int(math.floor(candidate_y))

            if not map_data.has_line_of_sight_to_exit(tile_x, tile_y):
                return 0.0, 0.0, 0.0, 0.0

            if not self._check_5point_line_of_sight(
                candidate_x, candidate_y, ex_tile, ey_tile, map_data
            ):
                return 0.0, 0.0, 0.0, 0.0

        dx: float = ex_center - candidate_x
        dy: float = ey_center - candidate_y

        target_angle: float = math.atan2(dy, dx)
        angle_delta: float = calculate_angle_delta(
            heading_rad, target_angle
        )

        prox_factor: float = max(0.0, min(1.0, 1.0 - (dist / max_range)))

        offset_deg: float = (
            profile.target_compasses_offset_angle
            if profile is not None else 22.5
        )
        offset_rad: float = math.radians(offset_deg)
        focus_half_rad: float = math.radians(
            (profile.focus_field_of_view if profile is not None else 180.0)
            / 2.0
        )
        perip_half_rad: float = math.radians(
            (
                profile.periphere_field_of_view
                if profile is not None else 360.0
            ) / 2.0
        )

        efl = self._compute_eye_intensity(
            angle_delta - offset_rad, focus_half_rad
        ) * prox_factor
        efr = self._compute_eye_intensity(
            angle_delta + offset_rad, focus_half_rad
        ) * prox_factor

        epl = self._compute_eye_intensity(
            angle_delta - offset_rad, perip_half_rad
        ) * prox_factor
        epr = self._compute_eye_intensity(
            angle_delta + offset_rad, perip_half_rad
        ) * prox_factor

        return efl, efr, epl, epr

    def _check_5point_line_of_sight(
        self,
        cx: float,
        cy: float,
        ex_tile: int,
        ey_tile: int,
        map_data: MapData
    ) -> bool:
        """
        Evaluates 5-point inset targets on exit tile for unblocked LOS.
        """
        targets = [
            (float(ex_tile) + 0.5, float(ey_tile) + 0.5),
            (float(ex_tile) + 0.1, float(ey_tile) + 0.1),
            (float(ex_tile) + 0.9, float(ey_tile) + 0.1),
            (float(ex_tile) + 0.1, float(ey_tile) + 0.9),
            (float(ex_tile) + 0.9, float(ey_tile) + 0.9)
        ]

        for tx, ty in targets:
            if map_data._march_los_segment(cx, cy, tx, ty):
                return True

        return False

    def _compute_eye_intensity(
        self,
        delta_angle: float,
        half_fov_rad: float
    ) -> float:
        """
        Computes eye directional intensity decaying linearly over half_fov.
        """
        abs_delta: float = abs(calculate_angle_delta(0.0, delta_angle))
        if abs_delta > half_fov_rad or half_fov_rad < 1e-4:
            return 0.0
        intensity: float = 1.0 - (abs_delta / half_fov_rad)
        return max(0.0, min(1.0, intensity))
