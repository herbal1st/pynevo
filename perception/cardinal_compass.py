"""
World cardinal reference compasses (Binocular North and 4-Needle Cardinal).
"""

import math
from typing import Tuple, Optional

from entities.agent_profile_registry import ResolvedAgentProfile
from utils.math_utils import calculate_angle_delta


class BinocularNorthCompass:
    """
    Computes 4-channel Focus/Peripheral North orientation signals.
    """

    def compute_stereo_channels(
        self,
        heading_rad: float,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> Tuple[float, float, float, float]:
        """
        Calculates North eye intensities [0.0, 1.0] across focus & periphere.
        """
        if profile is not None and not profile.activate_north_compass:
            return 0.0, 0.0, 0.0, 0.0

        target_north_angle: float = -math.pi / 2.0
        angle_delta: float = calculate_angle_delta(
            heading_rad, target_north_angle
        )

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

        nfl = self._compute_eye_intensity(
            angle_delta - offset_rad, focus_half_rad
        )
        nfr = self._compute_eye_intensity(
            angle_delta + offset_rad, focus_half_rad
        )

        npl = self._compute_eye_intensity(
            angle_delta - offset_rad, perip_half_rad
        )
        npr = self._compute_eye_intensity(
            angle_delta + offset_rad, perip_half_rad
        )

        return nfl, nfr, npl, npr

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


class CardinalNeedleCompass:
    """
    Computes 4-needle view-facing cardinal signals (C-N, C-E, C-S, C-W).
    """

    def compute_cardinal_channels(
        self,
        heading_rad: float,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> Tuple[float, float, float, float]:
        """
        Calculates C-N, C-E, C-S, C-W needle intensities [0.0, 1.0].
        """
        if profile is not None and not profile.activate_cardinal_compass:
            return 0.0, 0.0, 0.0, 0.0

        half_arc: float = math.pi / 2.0

        north_angle: float = -math.pi / 2.0
        east_angle: float = 0.0
        south_angle: float = math.pi / 2.0
        west_angle: float = math.pi

        cn: float = self._compute_needle_intensity(
            heading_rad, north_angle, half_arc
        )
        ce: float = self._compute_needle_intensity(
            heading_rad, east_angle, half_arc
        )
        cs: float = self._compute_needle_intensity(
            heading_rad, south_angle, half_arc
        )
        cw: float = self._compute_needle_intensity(
            heading_rad, west_angle, half_arc
        )

        return cn, ce, cs, cw

    def _compute_needle_intensity(
        self,
        heading_rad: float,
        target_angle: float,
        half_arc: float
    ) -> float:
        """
        Computes linear needle intensity decaying from 1.0 to 0.0 over half_arc.
        """
        delta: float = abs(calculate_angle_delta(heading_rad, target_angle))
        if delta > half_arc:
            return 0.0
        intensity: float = 1.0 - (delta / half_arc)
        return max(0.0, min(1.0, intensity))
