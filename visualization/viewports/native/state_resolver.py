"""
Extracts frame telemetry and derives physical candidate states in O(1) time.
"""

from dataclasses import dataclass
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile,
)
from entities.skin_profile_registry import ResolvedSkinProfile
from entities.entity_express import EntityExpress
import config


@dataclass(frozen=True, slots=True)
class ViewportFrameState:
    cand_idx: int
    frame_idx: int
    x: float
    y: float
    heading: float
    health: float
    dist: int
    hit_wall: bool
    is_alive: bool
    reached_exit: bool
    speed_ratio: float
    is_idle: bool
    is_healing: bool
    net_delta: float
    face_str: str
    score_val: int
    radius_ratio: float = 0.25
    skin: Optional[ResolvedSkinProfile] = None
    target_pos: Tuple[int, int] = (0, 0)
    checkpoint_pos: Tuple[int, int] = (0, 0)
    stage_idx: int = 0


class ViewportStateResolver:
    """
    Resolves O(1) telemetry array rows into ViewportFrameState containers.
    """

    def __init__(self) -> None:
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(
            config.ACTIVE_AGENT_PROFILE
        )
        self._target_stage_cache: Dict[Tuple[int, int], List[int]] = {}
        self._last_gen_key: Optional[int] = None

    def resolve_frame_state(
        self,
        gen_data: Dict[str, Any],
        cand_idx: int,
        active_step: int,
    ) -> Optional[ViewportFrameState]:
        telemetry = gen_data.get("telemetry", None)
        if telemetry is None:
            return None

        max_f: int = int(telemetry.shape[0])
        pop_s: int = int(telemetry.shape[1])
        c_idx: int = int(cand_idx)

        if c_idx < 0 or c_idx >= pop_s or max_f == 0:
            return None

        raw_scores = gen_data.get("raw_scores", [])
        score_val: int = (
            int(round(raw_scores[c_idx]))
            if c_idx < len(raw_scores)
            else 0
        )

        f_idx: int = min(max(0, int(active_step)), max_f - 1)
        row = telemetry[f_idx, c_idx]

        cx: float = float(row[0])
        cy: float = float(row[1])
        heading: float = float(row[2])
        health_val: float = float(row[3])
        dist_val: int = int(row[4])
        hit_wall: bool = bool(row[5] > 0.5)
        is_alive: bool = bool(row[6] > 0.5)
        reached_exit: bool = bool(row[7] > 0.5)

        spd_ratio: float = self._calculate_speed_ratio(
            telemetry, c_idx, f_idx, cx, cy
        )

        is_idle: bool = (
            spd_ratio < self.profile.idle_damage_speed_threshold
        )
        is_healing: bool = (
            spd_ratio >= self.profile.heal_speed_threshold and is_alive
        )

        face_str: str = EntityExpress.resolve_face(
            reached_exit, hit_wall, is_alive, profile=self.profile
        )

        bad_score: float = (
            (1.0 if hit_wall else 0.0) + (1.0 if is_idle else 0.0)
        )
        good_score: float = 1.0 if is_healing else 0.0
        net_delta: float = bad_score - good_score

        t_seq = gen_data.get(
            "target_sequence",
            [gen_data.get("exit_pos", (0, 0))]
        )
        curr_stage, safe_target = self._resolve_active_target_fast(
            gen_data, c_idx, f_idx, telemetry, t_seq
        )

        if curr_stage == 0:
            checkpoint_target = gen_data.get("start_pos", (0, 0))
        else:
            prev_stage_idx: int = min(
                curr_stage - 1, len(t_seq) - 1
            )
            checkpoint_target = t_seq[prev_stage_idx]

        return ViewportFrameState(
            cand_idx=c_idx,
            frame_idx=f_idx,
            x=cx,
            y=cy,
            heading=heading,
            health=health_val,
            dist=dist_val,
            hit_wall=hit_wall,
            is_alive=is_alive,
            reached_exit=reached_exit,
            speed_ratio=spd_ratio,
            is_idle=is_idle,
            is_healing=is_healing,
            net_delta=net_delta,
            face_str=face_str,
            score_val=score_val,
            radius_ratio=self.profile.agent_radius_ratio,
            skin=self.profile.skin,
            target_pos=safe_target,
            checkpoint_pos=checkpoint_target,
            stage_idx=curr_stage
        )

    def _resolve_active_target_fast(
        self,
        gen_data: Dict[str, Any],
        cand_idx: int,
        frame_idx: int,
        telemetry: np.ndarray,
        target_sequence: List[Tuple[int, int]]
    ) -> Tuple[int, Tuple[int, int]]:
        """
        O(1) pre-cached stage resolution eliminating O(T) nested frame loops.
        """
        if not target_sequence:
            return 0, (0, 0)

        gen_id = int(gen_data.get("generation", 0))
        cache_key = (gen_id, cand_idx)

        # Build transition timeline cache once per candidate
        if cache_key not in self._target_stage_cache or self._last_gen_key != gen_id:
            if self._last_gen_key != gen_id:
                self._target_stage_cache.clear()
                self._last_gen_key = gen_id

            max_f = telemetry.shape[0]
            stage_timeline = [0] * max_f
            hold_thresh_sq = 0.2304
            target_hold_frames = 15

            curr_stage = 0
            hold_count = 0
            n_seq = len(target_sequence)

            for s in range(max_f):
                if curr_stage >= n_seq:
                    stage_timeline[s] = n_seq - 1
                    continue

                tx, ty = target_sequence[curr_stage]
                tc_x = float(tx) + 0.5
                tc_y = float(ty) + 0.5

                cx = float(telemetry[s, cand_idx, 0])
                cy = float(telemetry[s, cand_idx, 1])

                dx = cx - tc_x
                dy = cy - tc_y
                if (dx * dx + dy * dy) <= hold_thresh_sq:
                    hold_count += 1
                    if hold_count >= target_hold_frames:
                        curr_stage += 1
                        hold_count = 0
                else:
                    hold_count = 0

                stage_timeline[s] = min(curr_stage, n_seq - 1)

            self._target_stage_cache[cache_key] = stage_timeline

        timeline = self._target_stage_cache[cache_key]
        safe_f = min(max(0, frame_idx), len(timeline) - 1)
        active_stage = timeline[safe_f]
        safe_stage = min(active_stage, len(target_sequence) - 1)

        return safe_stage, target_sequence[safe_stage]

    def _calculate_speed_ratio(
        self,
        telemetry: np.ndarray,
        cand_idx: int,
        frame_idx: int,
        curr_x: float,
        curr_y: float,
    ) -> float:
        if frame_idx <= 0:
            return 0.0

        prev_x: float = float(telemetry[frame_idx - 1, cand_idx, 0])
        prev_y: float = float(telemetry[frame_idx - 1, cand_idx, 1])

        dx: float = curr_x - prev_x
        dy: float = curr_y - prev_y
        disp_dist: float = (dx * dx + dy * dy) ** 0.5

        max_speed: float = max(1e-4, self.profile.move_speed)
        return min(1.0, max(0.0, disp_dist / max_speed))
