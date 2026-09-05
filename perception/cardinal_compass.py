"""
World cardinal reference compasses (Binocular North and 4-Needle Cardinal).
"""

import math
from typing import Tuple, Optional

from entities.agent_profile_registry import ResolvedAgentProfile

TWO_PI: float = 2.0 * math.pi
HALF_PI: float = math.pi / 2.0
NORTH_ANGLE: float = -HALF_PI


class BinocularNorthCompass:
    """
    Computes 4-channel Focus/Peripheral North orientation signals with zero-call inlining.
    """

    def __init__(self) -> None:
        self._last_profile: Optional[ResolvedAgentProfile] = None
        self.offset_rad: float = math.radians(22.5)
        self.focus_half_rad: float = math.radians(90.0)
        self.perip_half_rad: float = math.radians(180.0)

    def _sync_profile(self, profile: ResolvedAgentProfile) -> None:
        if profile is not self._last_profile:
            self._last_profile = profile
            self.offset_rad = math.radians(profile.target_compasses_offset_angle)
            self.focus_half_rad = math.radians(profile.focus_field_of_view / 2.0)
            self.perip_half_rad = math.radians(profile.periphere_field_of_view / 2.0)

    def compute_stereo_channels(
        self,
        heading_rad: float,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> Tuple[float, float, float, float]:
        """
        Calculates North eye intensities across focus & periphere in a single arithmetic pass.
        """
        if profile is not None:
            if not profile.activate_north_compass:
                return 0.0, 0.0, 0.0, 0.0
            self._sync_profile(profile)

        # Inlined angle delta: (NORTH_ANGLE - heading_rad) % TWO_PI
        angle_delta = (NORTH_ANGLE - heading_rad) % TWO_PI
        if angle_delta > math.pi:
            angle_delta -= TWO_PI

        off = self.offset_rad
        f_half = self.focus_half_rad
        p_half = self.perip_half_rad

        # Left eye angle delta
        dl = (angle_delta - off) % TWO_PI
        if dl > math.pi:
            dl -= TWO_PI
        abs_dl = abs(dl)

        # Right eye angle delta
        dr = (angle_delta + off) % TWO_PI
        if dr > math.pi:
            dr -= TWO_PI
        abs_dr = abs(dr)

        # Inlined channel intensities
        if abs_dl > f_half:
            nfl = 0.0
        else:
            vl = 1.0 - (abs_dl / f_half)
            nfl = vl if vl < 1.0 else 1.0

        if abs_dr > f_half:
            nfr = 0.0
        else:
            vr = 1.0 - (abs_dr / f_half)
            nfr = vr if vr < 1.0 else 1.0

        if abs_dl > p_half:
            npl = 0.0
        else:
            vl = 1.0 - (abs_dl / p_half)
            npl = vl if vl < 1.0 else 1.0

        if abs_dr > p_half:
            npr = 0.0
        else:
            vr = 1.0 - (abs_dr / p_half)
            npr = vr if vr < 1.0 else 1.0

        return nfl, nfr, npl, npr


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
        Calculates C-N, C-E, C-S, C-W needle intensities with inlined angular deltas.
        """
        if profile is not None and not profile.activate_cardinal_compass:
            return 0.0, 0.0, 0.0, 0.0

        half_arc = HALF_PI

        # North (-pi/2)
        dn = (NORTH_ANGLE - heading_rad) % TWO_PI
        if dn > math.pi:
            dn -= TWO_PI
        abs_dn = abs(dn)
        cn = 0.0 if abs_dn > half_arc else (1.0 - (abs_dn / half_arc))

        # East (0.0)
        de = (-heading_rad) % TWO_PI
        if de > math.pi:
            de -= TWO_PI
        abs_de = abs(de)
        ce = 0.0 if abs_de > half_arc else (1.0 - (abs_de / half_arc))

        # South (pi/2)
        ds = (HALF_PI - heading_rad) % TWO_PI
        if ds > math.pi:
            ds -= TWO_PI
        abs_ds = abs(ds)
        cs = 0.0 if abs_ds > half_arc else (1.0 - (abs_ds / half_arc))

        # West (pi)
        dw = (math.pi - heading_rad) % TWO_PI
        if dw > math.pi:
            dw -= TWO_PI
        abs_dw = abs(dw)
        cw = 0.0 if abs_dw > half_arc else (1.0 - (abs_dw / half_arc))

        return cn, ce, cs, cw