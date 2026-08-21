"""
Extracts frame telemetry and derives physical candidate states.
"""

from dataclasses import dataclass
import math
from typing import Dict, Any, Optional
import numpy as np

from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile,
)
from entities.player_express import PlayerExpress
import config


@dataclass(frozen=True, slots=True)
class ViewportFrameState:
    """
    Immutable container holding candidate frame metrics and state.
    """

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


class ViewportStateResolver:
    """
    Resolves $O(1)$ telemetry array rows into ViewportFrameState containers.
    """

    def __init__(self) -> None:
        """
        Initializes profile registry and active agent profile.
        """
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(
            config.ACTIVE_AGENT_PROFILE
        )

    def resolve_frame_state(
        self,
        gen_data: Dict[str, Any],
        cand_idx: int,
        active_step: int,
    ) -> Optional[ViewportFrameState]:
        """
        Extracts telemetry row and computes derived physical state.
        """
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

        f_idx: int = int(min(max(0, int(active_step)), max_f - 1))
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

        face_str: str = PlayerExpress.resolve_face(
            reached_exit, hit_wall, is_alive, profile=self.profile
        )

        bad_score: float = (
            (1.0 if hit_wall else 0.0) + (1.0 if is_idle else 0.0)
        )
        good_score: float = 1.0 if is_healing else 0.0
        net_delta: float = bad_score - good_score

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
        )

    def _calculate_speed_ratio(
        self,
        telemetry: np.ndarray,
        cand_idx: int,
        frame_idx: int,
        curr_x: float,
        curr_y: float,
    ) -> float:
        """
        Calculates physical displacement speed ratio over frame delta.
        """
        if frame_idx <= 0:
            return 0.0

        prev_x: float = float(telemetry[frame_idx - 1, cand_idx, 0])
        prev_y: float = float(telemetry[frame_idx - 1, cand_idx, 1])

        dx: float = curr_x - prev_x
        dy: float = curr_y - prev_y
        disp_dist: float = math.sqrt((dx * dx) + (dy * dy))

        max_speed: float = max(1e-4, self.profile.move_speed)
        return max(0.0, min(1.0, disp_dist / max_speed))
