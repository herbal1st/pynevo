"""
Pure first-person embodied perception compiler for foreign maze exploration.
The agent perceives ONLY what a human in a foreign maze with eyes and a compass would:
1. LiDAR / Vision rays: distance to walls across forward visual arc.
2. Corridor Openings / Clearance: left, front, and right path openness (derived from vision).
3. Proprioception: speed, health, collision tactile feedback, idle status, and angular velocity.
4. Compass Orientation: magnetic heading sense (cos heading, sin heading).
5. Visual Target Beacon: strictly LINE-OF-SIGHT ONLY within visual range; 0.0 when blocked by walls.
Zero omniscience: no global coordinates, no frontier cheat vectors, and no through-wall radar.
"""

import math
from typing import Optional, Any
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData, march_los_segment_jit
from core.pathfinder import BFSPathfinder
from entities.agent_profile_registry import ResolvedAgentProfile
from perception.vision_arc import VisionArcSampler
from utils.math_utils import calculate_angle_delta


class SingleFrameFeatureCompiler:
    def __init__(
        self,
        profile: Optional[ResolvedAgentProfile] = None,
        gps_sensor: Optional[Any] = None
    ) -> None:
        self.profile: Optional[ResolvedAgentProfile] = profile
        if profile is not None:
            self.sampler: VisionArcSampler = VisionArcSampler(
                num_rays=profile.vision_rays,
                arc_angle_deg=profile.vision_arc_angle,
                max_dist=profile.vision_max_dist
            )
        else:
            self.sampler = VisionArcSampler()

        self.v_rays: int = self.sampler.num_rays
        # 3 clearance + 7 proprioception + 2 compass + 3 visual exit
        self.embodied_channels: int = 3 + 7 + 2 + 3
        self.total_dim: int = self.v_rays + self.embodied_channels
        self.base_vector_buffer: NDArray[np.float32] = np.zeros(
            self.total_dim, dtype=np.float32
        )

    def reset_candidate(self, cand_idx: int) -> None:
        pass

    def compile_base_vector(
        self,
        candidate_x: float,
        candidate_y: float,
        heading_rad: float,
        speed_ratio: float,
        health_ratio: float,
        map_data: MapData,
        pathfinder: Optional[BFSPathfinder] = None,
        candidate_idx: int = 0,
        prev_x: Optional[float] = None,
        prev_y: Optional[float] = None,
        prev_heading: Optional[float] = None,
        is_collided: bool = False,
        is_idle: bool = False,
        is_healing: bool = False,
        rot_ratio: float = 0.0,
        stage_idx: int = 0,
        angular_velocity: float = 0.0,
        agent_state: Optional[Any] = None
    ) -> NDArray[np.float32]:
        buf = self.base_vector_buffer

        # 1. LiDAR / Vision Arc: forward rangefinder depth to walls
        wall_channels = self.sampler.sample_vision_channels(
            candidate_x, candidate_y, heading_rad, map_data
        )
        buf[:self.v_rays] = wall_channels

        # 2. Corridor Openings / Clearance (Left, Front, Right) derived directly from vision rays
        third = max(1, self.v_rays // 3)
        left_clearance = 1.0 - float(np.mean(wall_channels[:third]))
        front_clearance = 1.0 - float(np.mean(wall_channels[third : 2 * third]))
        right_clearance = 1.0 - float(np.mean(wall_channels[2 * third :]))

        idx = self.v_rays
        buf[idx]     = np.float32(max(0.0, min(1.0, left_clearance)))
        buf[idx + 1] = np.float32(max(0.0, min(1.0, front_clearance)))
        buf[idx + 2] = np.float32(max(0.0, min(1.0, right_clearance)))
        idx += 3

        # 3. Proprioception & Tactile Feedback (7 channels)
        buf[idx]     = np.float32(max(0.0, min(1.0, speed_ratio)))
        buf[idx + 1] = np.float32(max(0.0, min(1.0, health_ratio)))
        buf[idx + 2] = np.float32(1.0 if is_collided else 0.0)
        buf[idx + 3] = np.float32(1.0 if is_idle else 0.0)
        buf[idx + 4] = np.float32(max(0.0, min(1.0, rot_ratio)))
        buf[idx + 5] = np.float32(1.0 if is_healing else 0.0)
        buf[idx + 6] = np.float32(max(-1.0, min(1.0, angular_velocity)))
        idx += 7

        # 4. Vestibular Compass Sense (2 channels)
        buf[idx]     = np.float32(math.cos(heading_rad))
        buf[idx + 1] = np.float32(math.sin(heading_rad))
        idx += 2

        # 5. Visual Target Beacon (STRICTLY Line-of-Sight ONLY within visual range, 3 channels)
        # If the exit is hidden behind a wall, the human CANNOT see it (all channels remain 0.0)
        target_pos = (
            map_data.get_target_pos(stage_idx)
            if hasattr(map_data, "get_target_pos")
            else map_data.exit_pos
        )
        tx_c = float(target_pos[0]) + 0.5
        ty_c = float(target_pos[1]) + 0.5
        dx_t = tx_c - candidate_x
        dy_t = ty_c - candidate_y
        dist_t = math.sqrt(dx_t * dx_t + dy_t * dy_t)
        max_vis = self.sampler.max_dist

        if dist_t <= max_vis:
            has_los = march_los_segment_jit(
                candidate_x, candidate_y, tx_c, ty_c,
                map_data.grid_array, map_data.width, map_data.height, 0.2
            )
            if has_los:
                target_ang = math.atan2(dy_t, dx_t)
                t_delta = calculate_angle_delta(heading_rad, target_ang)
                buf[idx]     = np.float32(1.0 - (dist_t / max_vis))
                buf[idx + 1] = np.float32(math.cos(t_delta))
                buf[idx + 2] = np.float32(math.sin(t_delta))
            else:
                buf[idx]     = np.float32(0.0)
                buf[idx + 1] = np.float32(0.0)
                buf[idx + 2] = np.float32(0.0)
        else:
            buf[idx]     = np.float32(0.0)
            buf[idx + 1] = np.float32(0.0)
            buf[idx + 2] = np.float32(0.0)

        return buf
