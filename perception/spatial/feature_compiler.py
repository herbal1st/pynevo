"""
Compiles single-frame sensory observations, topological corridor flow, and proprioceptive state vectors.
"""

import math
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from entities.agent_profile_registry import ResolvedAgentProfile
from perception.vision_arc import VisionArcSampler
from perception.exit_compass import ExitCompass
from perception.cardinal_compass import (
    BinocularNorthCompass,
    CardinalNeedleCompass
)
from perception.spatial.gps_sensor import TopologicalGPSSensor
from utils.math_utils import calculate_angle_delta


class SingleFrameFeatureCompiler:
    """
    Compiles sensory perception with active Topological Corridor Compass.
    """

    def __init__(
        self,
        profile: Optional[ResolvedAgentProfile] = None,
        gps_sensor: Optional[TopologicalGPSSensor] = None
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

        self.exit_compass: ExitCompass = ExitCompass()
        self.north_compass: BinocularNorthCompass = BinocularNorthCompass()
        self.cardinal_compass: CardinalNeedleCompass = CardinalNeedleCompass()
        self.gps_sensor: TopologicalGPSSensor = (
            gps_sensor or TopologicalGPSSensor(profile)
        )

        self.v_rays: int = self.sampler.num_rays
        self.use_binocular: bool = (
            profile.use_binocular_gps_compasses if profile is not None else True
        )
        self.gps_dim: int = 4 if self.use_binocular else 2
        # Base channels: [v_rays] + [7 proprio (spd, hp, dmg-c, dmg-i, dmg-s, heal, ang_vel)] +
        #                [2 corridor compass (path_l, path_r)] + [gps_dim] + [4 cardinal] + [4 north] + [4 exit]
        self.total_dim: int = self.v_rays + 7 + 2 + self.gps_dim + 4 + 4 + 4
        self.base_vector_buffer: NDArray[np.float32] = np.zeros(
            self.total_dim, dtype=np.float32
        )

    def compile_base_vector(
        self,
        candidate_x: float,
        candidate_y: float,
        heading_rad: float,
        speed_ratio: float,
        health_ratio: float,
        map_data: MapData,
        pathfinder: BFSPathfinder,
        candidate_idx: int = 0,
        prev_x: Optional[float] = None,
        prev_y: Optional[float] = None,
        prev_heading: Optional[float] = None,
        is_collided: bool = False,
        is_idle: bool = False,
        is_healing: bool = False,
        rot_ratio: float = 0.0,
        stage_idx: int = 0,
        angular_velocity: float = 0.0
    ) -> NDArray[np.float32]:
        # 1. Vision Arc Rays
        wall_channels: NDArray[np.float32] = self.sampler.sample_vision_channels(
            candidate_x, candidate_y, heading_rad, map_data
        )
        self.base_vector_buffer[:self.v_rays] = wall_channels

        # 2. Topological Corridor Compass (Calculates true corridor flow vector at agent tile)
        tx = int(candidate_x)
        ty = int(candidate_y)
        d_center = pathfinder.get_step_distance(tx, ty, stage_idx=stage_idx)

        best_dx = 0.0
        best_dy = 0.0
        best_dist = d_center

        # Check 4 cardinal neighbor tiles for distance drop
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + dx, ty + dy
            if map_data.is_walkable(nx, ny):
                d_n = pathfinder.get_step_distance(nx, ny, stage_idx=stage_idx)
                if d_n < best_dist:
                    best_dist = d_n
                    best_dx = float(dx)
                    best_dy = float(dy)

        if best_dx != 0.0 or best_dy != 0.0:
            target_path_angle = math.atan2(best_dy, best_dx)
            path_delta = calculate_angle_delta(heading_rad, target_path_angle)
            # Project into stereo corridor channels (Path Left vs Path Right)
            path_l = max(0.0, min(1.0, -path_delta / math.pi))
            path_r = max(0.0, min(1.0, path_delta / math.pi))
        else:
            path_l = 0.0
            path_r = 0.0

        # 3. Compasses
        act_exit = self.profile.activate_exit_compass if self.profile is not None else True
        if act_exit:
            exit_channels = self.exit_compass.compute_stereo_channels(
                candidate_x, candidate_y, heading_rad, map_data, self.profile, stage_idx=stage_idx
            )
        else:
            exit_channels = (0.0, 0.0, 0.0, 0.0)

        act_gps = self.profile.activate_gps_compass if self.profile is not None else True
        if act_gps:
            gps_channels = self.gps_sensor.compute_gps_channels(
                candidate_x, candidate_y, heading_rad, map_data, pathfinder,
                candidate_idx, prev_x, prev_y, prev_heading, stage_idx=stage_idx
            )
        else:
            gps_channels = (0.0, 0.0, 0.0, 0.0) if self.use_binocular else (0.0, 0.0)

        act_north = self.profile.activate_north_compass if self.profile is not None else True
        if act_north:
            north_channels = self.north_compass.compute_stereo_channels(heading_rad, self.profile)
        else:
            north_channels = (0.0, 0.0, 0.0, 0.0)

        act_cardinal = self.profile.activate_cardinal_compass if self.profile is not None else True
        if act_cardinal:
            c_n, c_e, c_s, c_w = self.cardinal_compass.compute_cardinal_channels(heading_rad, self.profile)
        else:
            c_n, c_e, c_s, c_w = 0.0, 0.0, 0.0, 0.0

        # Proprioception
        clamped_spd = 1.0 if speed_ratio > 1.0 else (0.0 if speed_ratio < 0.0 else float(speed_ratio))
        clamped_hp = 1.0 if health_ratio > 1.0 else (0.0 if health_ratio < 0.0 else float(health_ratio))
        val_dmg_c = 1.0 if is_collided else 0.0
        val_dmg_i = 1.0 if is_idle else 0.0
        val_dmg_s = 1.0 if rot_ratio > 1.0 else (0.0 if rot_ratio < 0.0 else float(rot_ratio))
        val_heal = 1.0 if is_healing else 0.0
        val_ang_vel = max(-1.0, min(1.0, float(angular_velocity)))

        # Write directly into contiguous buffer
        idx = self.v_rays
        self.base_vector_buffer[idx] = clamped_spd
        self.base_vector_buffer[idx + 1] = clamped_hp
        self.base_vector_buffer[idx + 2] = val_dmg_c
        self.base_vector_buffer[idx + 3] = val_dmg_i
        self.base_vector_buffer[idx + 4] = val_dmg_s
        self.base_vector_buffer[idx + 5] = val_heal
        self.base_vector_buffer[idx + 6] = val_ang_vel
        self.base_vector_buffer[idx + 7] = path_l
        self.base_vector_buffer[idx + 8] = path_r
        idx += 9

        for g_val in gps_channels:
            self.base_vector_buffer[idx] = g_val
            idx += 1

        self.base_vector_buffer[idx] = c_n
        self.base_vector_buffer[idx + 1] = c_e
        self.base_vector_buffer[idx + 2] = c_s
        self.base_vector_buffer[idx + 3] = c_w
        idx += 4

        self.base_vector_buffer[idx] = north_channels[0]
        self.base_vector_buffer[idx + 1] = north_channels[1]
        self.base_vector_buffer[idx + 2] = north_channels[2]
        self.base_vector_buffer[idx + 3] = north_channels[3]
        idx += 4

        self.base_vector_buffer[idx] = exit_channels[0]
        self.base_vector_buffer[idx + 1] = exit_channels[1]
        self.base_vector_buffer[idx + 2] = exit_channels[2]
        self.base_vector_buffer[idx + 3] = exit_channels[3]

        return self.base_vector_buffer
