"""
Computes topological BFS step-distance GPS progress channels.
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

from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from entities.agent_profile_registry import ResolvedAgentProfile


@njit(fastmath=True, cache=True)
def get_bilinear_bfs_distance(
    grid_data: NDArray[np.int32],
    px: float,
    py: float,
    width: int,
    height: int
) -> float:
    """
    JIT-compiled bilinear interpolation directly over the BFS 2D distance slice.
    """
    x0 = int(px)
    y0 = int(py)
    x1 = (x0 + 1) if (x0 + 1) < (width - 1) else (width - 1)
    y1 = (y0 + 1) if (y0 + 1) < (height - 1) else (height - 1)

    fx = px - x0
    fy = py - y0

    d00 = float(grid_data[y0, x0])
    d10 = float(grid_data[y0, x1])
    d01 = float(grid_data[y1, x0])
    d11 = float(grid_data[y1, x1])

    top = d00 + fx * (d10 - d00)
    bottom = d01 + fx * (d11 - d01)
    return top + fy * (bottom - top)


@njit(fastmath=True, cache=True)
def compute_stereo_gps_jit(
    cx: float,
    cy: float,
    heading_rad: float,
    offset_rad: float,
    r_body: float,
    grid_data: NDArray[np.int32],
    width: int,
    height: int,
    max_active_dist: float,
    prev_left: float,
    prev_right: float,
    move_speed: float,
    has_history: bool
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluates both stereo eye probe positions and bilinear lookups in a single JIT pass.
    """
    left_heading = heading_rad + offset_rad
    right_heading = heading_rad - offset_rad

    lx = cx + (r_body * math.cos(left_heading))
    ly = cy + (r_body * math.sin(left_heading))
    rx = cx + (r_body * math.cos(right_heading))
    ry = cy + (r_body * math.sin(right_heading))

    max_u = float(width - 1)
    max_v = float(height - 1)

    ul = lx - 0.5
    vl = ly - 0.5
    cl_u = 0.0 if ul < 0.0 else (max_u if ul > max_u else ul)
    cl_v = 0.0 if vl < 0.0 else (max_v if vl > max_v else vl)
    dist_l = get_bilinear_bfs_distance(grid_data, cl_u, cl_v, width, height)

    ur = rx - 0.5
    vr = ry - 0.5
    cr_u = 0.0 if ur < 0.0 else (max_u if ur > max_u else ur)
    cr_v = 0.0 if vr < 0.0 else (max_v if vr > max_v else vr)
    dist_r = get_bilinear_bfs_distance(grid_data, cr_u, cr_v, width, height)

    min_curr = dist_l if dist_l < dist_r else dist_r
    if min_curr > max_active_dist or not has_history:
        return 0.0, 0.0, 0.0, 0.0, dist_l, dist_r

    denom = move_speed if move_speed > 1e-4 else 1e-4
    dl = (prev_left - dist_l) / denom
    dr = (prev_right - dist_r) / denom

    sspl_pos = 1.0 if dl > 1.0 else (0.0 if dl < 0.0 else dl)
    abs_l = -dl if dl < 0.0 else 0.0
    sspl_neg = 1.0 if abs_l > 1.0 else (0.0 if abs_l < 0.0 else abs_l)

    sspr_pos = 1.0 if dr > 1.0 else (0.0 if dr < 0.0 else dr)
    abs_r = -dr if dr < 0.0 else 0.0
    sspr_neg = 1.0 if abs_r > 1.0 else (0.0 if abs_r < 0.0 else abs_r)

    return sspl_pos, sspr_pos, sspl_neg, sspr_neg, dist_l, dist_r


class TopologicalGPSSensor:
    """
    Evaluates topological BFS GPS progress with JIT-accelerated bilinear interpolation.
    """

    def __init__(
        self,
        profile: Optional[ResolvedAgentProfile] = None,
        max_candidates: int = 128
    ) -> None:
        self.profile: Optional[ResolvedAgentProfile] = profile
        self.max_candidates: int = max_candidates
        self._cache: NDArray[np.float32] = np.full(
            (max_candidates, 2), 9999.0, dtype=np.float32
        )
        self._initialized: NDArray[np.bool_] = np.zeros(
            max_candidates, dtype=bool
        )
        self.last_gps_channels: Tuple[float, ...] = ()

    @property
    def last_gps_progress(self) -> Tuple[float, ...]:
        return self.last_gps_channels

    def reset_candidate_history(self, candidate_idx: int) -> None:
        self._ensure_capacity(candidate_idx)
        self._cache[candidate_idx].fill(9999.0)
        self._initialized[candidate_idx] = False

    def compute_gps_channels(
        self,
        cx: float,
        cy: float,
        heading_rad: float,
        map_data: MapData,
        pathfinder: BFSPathfinder,
        candidate_idx: int = 0,
        prev_x: Optional[float] = None,
        prev_y: Optional[float] = None,
        prev_heading: Optional[float] = None,
        stage_idx: int = 0
    ) -> Tuple[float, ...]:
        use_binocular: bool = (
            self.profile.use_binocular_gps_compasses
            if self.profile is not None
            else True
        )

        if self.profile is not None and not self.profile.activate_gps_compass:
            res = (0.0, 0.0, 0.0, 0.0) if use_binocular else (0.0, 0.0)
            self.last_gps_channels = res
            return res

        if hasattr(map_data, "get_target_pos"):
            ex_t, ey_t = map_data.get_target_pos(stage_idx)
        else:
            ex_t, ey_t = map_data.exit_pos

        t_cx: float = float(ex_t) + 0.5
        t_cy: float = float(ey_t) + 0.5
        dx_t: float = cx - t_cx
        dy_t: float = cy - t_cy
        dist_to_t: float = math.sqrt((dx_t * dx_t) + (dy_t * dy_t))

        hold_thresh: float = (
            self.profile.target_hold_distance_threshold
            if self.profile is not None else 0.25
        )

        if dist_to_t <= hold_thresh:
            res = (1.0, 1.0, 0.0, 0.0) if use_binocular else (1.0, 0.0)
            self.last_gps_channels = res
            return res

        sx, sy = map_data.start_pos
        initial_dist: int = pathfinder.get_step_distance(sx, sy, stage_idx=stage_idx)
        if initial_dist >= 9999 or initial_dist == 0:
            res = (0.0, 0.0, 0.0, 0.0) if use_binocular else (0.0, 0.0)
            self.last_gps_channels = res
            return res

        range_ratio: float = (
            self.profile.range_gps_compass if self.profile is not None else 1.0
        )
        max_active_dist: float = (
            9998.0 if range_ratio >= 1.0 else float(initial_dist) * range_ratio
        )

        is_endless: bool = hasattr(map_data, "chunk_manager")
        if is_endless and self.profile is not None:
            move_speed: float = self.profile.endless_move_speed
            rad_ratio: float = self.profile.endless_agent_radius_ratio
        elif self.profile is not None:
            move_speed = self.profile.move_speed
            rad_ratio = self.profile.agent_radius_ratio
        else:
            move_speed = 0.125
            rad_ratio = 0.45

        r_body: float = 0.5 * rad_ratio
        offset_deg: float = (
            self.profile.target_compasses_offset_angle if self.profile is not None else 22.5
        )
        offset_rad: float = math.radians(offset_deg)

        self._ensure_capacity(candidate_idx)
        has_history: bool = bool(self._initialized[candidate_idx])
        p_left: float = float(self._cache[candidate_idx, 0])
        p_right: float = float(self._cache[candidate_idx, 1])

        grid_matrix = pathfinder._matrix_buffer[stage_idx]

        # Single JIT call bypassing multiple FFI roundtrips
        (
            sspl_pos, sspr_pos, sspl_neg, sspr_neg, cur_l, cur_r
        ) = compute_stereo_gps_jit(
            cx, cy, heading_rad, offset_rad, r_body,
            grid_matrix, map_data.width, map_data.height,
            max_active_dist, p_left, p_right, move_speed, has_history
        )

        self._cache[candidate_idx, 0] = cur_l
        self._cache[candidate_idx, 1] = cur_r
        self._initialized[candidate_idx] = True

        if use_binocular:
            res = (sspl_pos, sspr_pos, sspl_neg, sspr_neg)
        else:
            res = (sspl_pos, sspl_neg)

        self.last_gps_channels = res
        return res

    def _ensure_capacity(self, candidate_idx: int) -> None:
        if candidate_idx >= self.max_candidates:
            need_cands: int = (
                self.max_candidates * 2
                if (self.max_candidates * 2) > (candidate_idx + 1)
                else (candidate_idx + 1)
            )
            new_cache = np.full((need_cands, 2), 9999.0, dtype=np.float32)
            new_cache[: self.max_candidates] = self._cache
            self._cache = new_cache

            new_init = np.zeros(need_cands, dtype=bool)
            new_init[: self.max_candidates] = self._initialized
            self._initialized = new_init
            self.max_candidates = need_cands