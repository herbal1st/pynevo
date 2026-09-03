"""
Compiles sensory observations, Dual-Layer LiDAR (Walls + SLAM Trail), and Odometry.
"""

import math
from typing import Optional, List
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


class SingleFrameFeatureCompiler:
    """
    Compiles sensory perception channels into 1D base observation vectors.
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
        rot_ratio: float = 0.0
    ) -> NDArray[np.float32]:
        # 1. Wall Distance Rays
        wall_channels = self.sampler.sample_vision_channels(
            candidate_x, candidate_y, heading_rad, map_data
        )

        # 2. SLAM Breadcrumb Scent Rays (0.0 for virgin paths)
        trail_channels = np.zeros_like(wall_channels)

        use_binocular: bool = (
            self.profile.use_binocular_gps_compasses
            if self.profile is not None else False
        )

        # 3. Compasses (Gated)
        if self.profile and self.profile.activate_exit_compass:
            exit_channels = self.exit_compass.compute_stereo_channels(
                candidate_x, candidate_y, heading_rad, map_data, self.profile
            )
        else:
            exit_channels = (0.0, 0.0, 0.0, 0.0)

        if self.profile and self.profile.activate_gps_compass:
            gps_channels = self.gps_sensor.compute_gps_channels(
                candidate_x, candidate_y, heading_rad,
                map_data, pathfinder, candidate_idx,
                prev_x, prev_y, prev_heading
            )
        else:
            gps_channels = (0.0, 0.0, 0.0, 0.0) if use_binocular else (0.0, 0.0)

        if self.profile and self.profile.activate_north_compass:
            north_channels = self.north_compass.compute_stereo_channels(
                heading_rad, self.profile
            )
        else:
            north_channels = (0.0, 0.0, 0.0, 0.0)

        if self.profile and self.profile.activate_cardinal_compass:
            c_n, c_e, c_s, c_w = self.cardinal_compass.compute_cardinal_channels(
                heading_rad, self.profile
            )
        else:
            c_n, c_e, c_s, c_w = 0.0, 0.0, 0.0, 0.0

        # 4. Proprioception & Dead-Reckoning Odometry
        start_x, start_y = map_data.start_pos
        rel_x = (candidate_x - (float(start_x) + 0.5)) / float(max(1, map_data.width))
        rel_y = (candidate_y - (float(start_y) + 0.5)) / float(max(1, map_data.height))
        vel_x = (candidate_x - (prev_x if prev_x is not None else candidate_x)) / 0.20
        vel_y = (candidate_y - (prev_y if prev_y is not None else candidate_y)) / 0.20

        state_list: List[float] = [
            max(0.0, min(1.0, float(speed_ratio))),
            max(0.0, min(1.0, float(health_ratio))),
            rel_x,
            rel_y,
            vel_x,
            vel_y
        ]
        state_list.extend(gps_channels)
        state_list.extend([c_n, c_e, c_s, c_w])
        state_list.extend(north_channels)
        state_list.extend(exit_channels)

        state_features = np.array(state_list, dtype=np.float32)
        return np.concatenate([wall_channels, trail_channels, state_features])
