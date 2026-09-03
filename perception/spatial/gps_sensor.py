"""
Computes topological BFS step-distance GPS progress channels.
"""

import math
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from entities.agent_profile_registry import ResolvedAgentProfile
from core.accelerated import get_bilinear_bfs_dist_jit


class TopologicalGPSSensor:
    """
    Evaluates topological BFS GPS progress with bilinear interpolation.
    """

    def __init__(
        self,
        profile: Optional[ResolvedAgentProfile] = None,
        max_candidates: int = 128
    ) -> None:
        """
        Initializes profile reference and pre-allocated array cache.
        """
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
        """
        Returns last calculated GPS progress channels.
        """
        return self.last_gps_channels

    def reset_candidate_history(self, candidate_idx: int) -> None:
        """
        Zeroes out recorded GPS distance slot for candidate.
        """
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
        prev_heading: Optional[float] = None
    ) -> Tuple[float, ...]:
        """
        Computes frame-to-frame topological BFS GPS progress channels.
        """
        use_binocular: bool = (
            self.profile.use_binocular_gps_compasses
            if self.profile is not None
            else True
        )

        if (
            self.profile is not None
            and not self.profile.activate_gps_compass
        ):
            res = (0.0, 0.0, 0.0, 0.0) if use_binocular else (0.0, 0.0)
            self.last_gps_channels = res
            return res

        sx, sy = map_data.start_pos
        initial_dist: int = pathfinder.get_step_distance(sx, sy)
        if initial_dist >= 9999 or initial_dist == 0:
            res = (0.0, 0.0, 0.0, 0.0) if use_binocular else (0.0, 0.0)
            self.last_gps_channels = res
            return res

        range_ratio: float = (
            self.profile.range_gps_compass
            if self.profile is not None
            else 1.0
        )
        if range_ratio >= 1.0:
            max_active_dist: float = 9998.0
        else:
            max_active_dist = float(initial_dist) * range_ratio

        move_speed: float = (
            self.profile.move_speed if self.profile is not None else 0.125
        )
        rad_ratio: float = (
            self.profile.agent_radius_ratio
            if self.profile is not None
            else 0.45
        )
        r_body: float = 0.5 * rad_ratio

        is_stateless: bool = (
            prev_x is not None
            and prev_y is not None
            and prev_heading is not None
        )

        if not use_binocular:
            res = self._compute_mono_channels(
                cx,
                cy,
                heading_rad,
                map_data,
                pathfinder,
                candidate_idx,
                prev_x,
                prev_y,
                prev_heading,
                max_active_dist,
                move_speed,
                r_body,
                is_stateless
            )
            self.last_gps_channels = res
            return res

        res = self._compute_stereo_channels(
            cx,
            cy,
            heading_rad,
            map_data,
            pathfinder,
            candidate_idx,
            prev_x,
            prev_y,
            prev_heading,
            max_active_dist,
            move_speed,
            r_body,
            is_stateless
        )
        self.last_gps_channels = res
        return res

    def get_bilinear_bfs_distance(
        self,
        cx: float,
        cy: float,
        map_data: MapData,
        pathfinder: BFSPathfinder,
        cand_x: Optional[float] = None,
        cand_y: Optional[float] = None
    ) -> float:
        """
        Bilinearly interpolates continuous distance from BFS distance grid via Numba.
        """
        if not hasattr(pathfinder, "numpy_dist") or pathfinder.numpy_dist is None:
            pathfinder.numpy_dist = np.array(
                pathfinder.distance_matrix, dtype=np.int32
            )

        cand_x_val: float = cand_x if cand_x is not None else cx
        cand_y_val: float = cand_y if cand_y is not None else cy

        return get_bilinear_bfs_dist_jit(
            cx,
            cy,
            pathfinder.numpy_dist,
            map_data.width,
            map_data.height,
            cand_x_val,
            cand_y_val
        )

    def _compute_mono_channels(
        self,
        cx: float,
        cy: float,
        heading_rad: float,
        map_data: MapData,
        pathfinder: BFSPathfinder,
        candidate_idx: int,
        prev_x: Optional[float],
        prev_y: Optional[float],
        prev_heading: Optional[float],
        max_active_dist: float,
        move_speed: float,
        r_body: float,
        is_stateless: bool
    ) -> Tuple[float, float]:
        """
        Computes 2 mono GPS progress channels (BFS-, BFS+).
        """
        nose_x: float = cx + (r_body * math.cos(heading_rad))
        nose_y: float = cy + (r_body * math.sin(heading_rad))

        curr_dist_val = self.get_bilinear_bfs_distance(
            nose_x, nose_y, map_data, pathfinder, cx, cy
        )

        if curr_dist_val > max_active_dist:
            if not is_stateless:
                self._ensure_capacity(candidate_idx)
                self._cache[candidate_idx, 0] = curr_dist_val
                self._initialized[candidate_idx] = True
            return 0.0, 0.0

        if is_stateless and prev_x is not None and prev_y is not None:
            prev_nose_x: float = prev_x + (
                r_body * math.cos(prev_heading or 0.0)
            )
            prev_nose_y: float = prev_y + (
                r_body * math.sin(prev_heading or 0.0)
            )
            prev_dist = self.get_bilinear_bfs_distance(
                prev_nose_x,
                prev_nose_y,
                map_data,
                pathfinder,
                prev_x,
                prev_y
            )
        else:
            self._ensure_capacity(candidate_idx)
            if not self._initialized[candidate_idx]:
                self._cache[candidate_idx, 0] = curr_dist_val
                self._initialized[candidate_idx] = True
                return 0.0, 0.0

            prev_dist = float(self._cache[candidate_idx, 0])
            self._cache[candidate_idx, 0] = curr_dist_val

        if curr_dist_val >= 9999.0 or prev_dist >= 9999.0:
            return 0.0, 0.0

        delta_dist: float = (prev_dist - curr_dist_val) / max(
            1e-4, move_speed
        )
        d_closer: float = max(0.0, min(1.0, delta_dist))
        d_farther: float = max(0.0, min(1.0, abs(min(0.0, delta_dist))))

        return d_closer, d_farther

    def _compute_stereo_channels(
        self,
        cx: float,
        cy: float,
        heading_rad: float,
        map_data: MapData,
        pathfinder: BFSPathfinder,
        candidate_idx: int,
        prev_x: Optional[float],
        prev_y: Optional[float],
        prev_heading: Optional[float],
        max_active_dist: float,
        move_speed: float,
        r_body: float,
        is_stateless: bool
    ) -> Tuple[float, float, float, float]:
        """
        Computes 4 stereo GPS channels (BFSL-, BFSR-, BFSL+, BFSR+).
        """
        offset_deg: float = (
            self.profile.target_compasses_offset_angle
            if self.profile is not None
            else 22.5
        )
        offset_rad: float = math.radians(offset_deg)

        left_heading: float = heading_rad + offset_rad
        right_heading: float = heading_rad - offset_rad

        lx: float = cx + (r_body * math.cos(left_heading))
        ly: float = cy + (r_body * math.sin(left_heading))
        rx: float = cx + (r_body * math.cos(right_heading))
        ry: float = cy + (r_body * math.sin(right_heading))

        dist_left = self.get_bilinear_bfs_distance(
            lx, ly, map_data, pathfinder, cx, cy
        )
        dist_right = self.get_bilinear_bfs_distance(
            rx, ry, map_data, pathfinder, cx, cy
        )

        min_curr_dist: float = min(dist_left, dist_right)
        if min_curr_dist > max_active_dist:
            if not is_stateless:
                self._ensure_capacity(candidate_idx)
                self._cache[candidate_idx, 0] = dist_left
                self._cache[candidate_idx, 1] = dist_right
                self._initialized[candidate_idx] = True
            return 0.0, 0.0, 0.0, 0.0

        if is_stateless and prev_x is not None and prev_y is not None:
            p_head: float = prev_heading or 0.0
            prev_left_head: float = p_head + offset_rad
            prev_right_head: float = p_head - offset_rad

            plx: float = prev_x + (r_body * math.cos(prev_left_head))
            ply: float = prev_y + (r_body * math.sin(prev_left_head))
            prx: float = prev_x + (r_body * math.cos(prev_right_head))
            pry: float = prev_y + (r_body * math.sin(prev_right_head))

            prev_left = self.get_bilinear_bfs_distance(
                plx,
                ply,
                map_data,
                pathfinder,
                prev_x,
                prev_y
            )
            prev_right = self.get_bilinear_bfs_distance(
                prx,
                pry,
                map_data,
                pathfinder,
                prev_x,
                prev_y
            )
        else:
            self._ensure_capacity(candidate_idx)
            if not self._initialized[candidate_idx]:
                self._cache[candidate_idx, 0] = dist_left
                self._cache[candidate_idx, 1] = dist_right
                self._initialized[candidate_idx] = True
                return 0.0, 0.0, 0.0, 0.0

            prev_left = float(self._cache[candidate_idx, 0])
            prev_right = float(self._cache[candidate_idx, 1])

            self._cache[candidate_idx, 0] = dist_left
            self._cache[candidate_idx, 1] = dist_right

        d_left: float = (prev_left - dist_left) / max(1e-4, move_speed)
        d_right: float = (prev_right - dist_right) / max(1e-4, move_speed)

        sspl_pos: float = max(0.0, min(1.0, max(0.0, d_left)))
        sspl_neg: float = max(0.0, min(1.0, abs(min(0.0, d_left))))
        sspr_pos: float = max(0.0, min(1.0, max(0.0, d_right)))
        sspr_neg: float = max(0.0, min(1.0, abs(min(0.0, d_right))))

        return sspl_pos, sspr_pos, sspl_neg, sspr_neg

    def _ensure_capacity(self, candidate_idx: int) -> None:
        """
        Expands pre-allocated cache if candidate_idx exceeds bounds.
        """
        if candidate_idx >= self.max_candidates:
            need_cands: int = max(self.max_candidates * 2, candidate_idx + 1)
            new_cache = np.full(
                (need_cands, 2), 9999.0, dtype=np.float32
            )
            new_cache[: self.max_candidates] = self._cache
            self._cache = new_cache

            new_init = np.zeros(need_cands, dtype=bool)
            new_init[: self.max_candidates] = self._initialized
            self._initialized = new_init

            self.max_candidates = need_cands
