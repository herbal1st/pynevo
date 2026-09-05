"""
Stereo binocular target compass sensor with profile-driven bounds.
"""

import math
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

from core.map_data import MapData, march_los_segment_jit
from entities.agent_profile_registry import ResolvedAgentProfile

TWO_PI: float = 2.0 * math.pi


@njit(fastmath=True, cache=True)
def check_5point_los_jit(
    cx: float,
    cy: float,
    ex_tile: int,
    ey_tile: int,
    grid_array: NDArray[np.uint8],
    width: int,
    height: int
) -> bool:
    """
    Evaluates 5-point inset probes in a single JIT routine.
    """
    f_ex = float(ex_tile)
    f_ey = float(ey_tile)

    # Center
    if march_los_segment_jit(cx, cy, f_ex + 0.5, f_ey + 0.5, grid_array, width, height, 0.2):
        return True
    # Corners
    if march_los_segment_jit(cx, cy, f_ex + 0.1, f_ey + 0.1, grid_array, width, height, 0.2):
        return True
    if march_los_segment_jit(cx, cy, f_ex + 0.9, f_ey + 0.1, grid_array, width, height, 0.2):
        return True
    if march_los_segment_jit(cx, cy, f_ex + 0.1, f_ey + 0.9, grid_array, width, height, 0.2):
        return True
    if march_los_segment_jit(cx, cy, f_ex + 0.9, f_ey + 0.9, grid_array, width, height, 0.2):
        return True

    return False


class ExitCompass:
    """
    Computes 4-channel Focus/Peripheral Exit Lock Radar signals.
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
        candidate_x: float,
        candidate_y: float,
        heading_rad: float,
        map_data: MapData,
        profile: Optional[ResolvedAgentProfile] = None,
        stage_idx: int = 0
    ) -> Tuple[float, float, float, float]:
        if profile is not None:
            if not profile.activate_exit_compass:
                return 0.0, 0.0, 0.0, 0.0
            self._sync_profile(profile)

        # Direct call bypassing hasattr
        ex_tile, ey_tile = map_data.get_target_pos(stage_idx)

        ex_center: float = float(ex_tile) + 0.5
        ey_center: float = float(ey_tile) + 0.5

        dx: float = ex_center - candidate_x
        dy: float = ey_center - candidate_y
        dist: float = math.sqrt((dx * dx) + (dy * dy))

        max_range: float = profile.range_exit_compass if profile is not None else 20.0
        if dist > max_range:
            return 0.0, 0.0, 0.0, 0.0

        use_los_gating: bool = profile.exit_compass_los_gating if profile is not None else True
        if use_los_gating:
            tile_x: int = int(math.floor(candidate_x))
            tile_y: int = int(math.floor(candidate_y))

            if not map_data.has_line_of_sight_to_exit(tile_x, tile_y):
                return 0.0, 0.0, 0.0, 0.0

            if not check_5point_los_jit(
                candidate_x, candidate_y, ex_tile, ey_tile,
                map_data.grid_array, map_data.width, map_data.height
            ):
                return 0.0, 0.0, 0.0, 0.0

        target_angle: float = math.atan2(dy, dx)
        angle_delta: float = (target_angle - heading_rad) % TWO_PI
        if angle_delta > math.pi:
            angle_delta -= TWO_PI

        p_ratio = 1.0 - (dist / max_range)
        prox_factor: float = 1.0 if p_ratio > 1.0 else (0.0 if p_ratio < 0.0 else p_ratio)

        off = self.offset_rad
        f_half = self.focus_half_rad
        p_half = self.perip_half_rad

        dl = (angle_delta - off) % TWO_PI
        if dl > math.pi:
            dl -= TWO_PI
        abs_dl = abs(dl)

        dr = (angle_delta + off) % TWO_PI
        if dr > math.pi:
            dr -= TWO_PI
        abs_dr = abs(dr)

        if abs_dl > f_half:
            efl = 0.0
        else:
            vl = (1.0 - (abs_dl / f_half)) * prox_factor
            efl = vl if vl < 1.0 else 1.0

        if abs_dr > f_half:
            efr = 0.0
        else:
            vr = (1.0 - (abs_dr / f_half)) * prox_factor
            efr = vr if vr < 1.0 else 1.0

        if abs_dl > p_half:
            epl = 0.0
        else:
            vl = (1.0 - (abs_dl / p_half)) * prox_factor
            epl = vl if vl < 1.0 else 1.0

        if abs_dr > p_half:
            epr = 0.0
        else:
            vr = (1.0 - (abs_dr / p_half)) * prox_factor
            epr = vr if vr < 1.0 else 1.0

        return efl, efr, epl, epr