"""
Compiles single-frame sensory observations and proprioceptive state vectors.
"""

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
        """
        Initializes perception sensors and topological GPS sensor.
        """
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
        self.north_compass: BinocularNorthCompass = (
            BinocularNorthCompass()
        )
        self.cardinal_compass: CardinalNeedleCompass = (
            CardinalNeedleCompass()
        )
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
        rot_ratio: float = 0.0,
        stage_idx: int = 0
    ) -> NDArray[np.float32]:
        """
        Compiles single-frame base vector for active or historical step.
        """
        wall_channels: NDArray[np.float32] = (
            self.sampler.sample_vision_channels(
                candidate_x, candidate_y, heading_rad, map_data
            )
        )

        use_binocular: bool = (
            self.profile.use_binocular_gps_compasses
            if self.profile is not None
            else True
        )

        act_exit = (
            self.profile.activate_exit_compass
            if self.profile is not None
            else True
        )
        if act_exit:
            exit_channels = self.exit_compass.compute_stereo_channels(
                candidate_x,
                candidate_y,
                heading_rad,
                map_data,
                self.profile,
                stage_idx=stage_idx
            )
        else:
            exit_channels = (0.0, 0.0, 0.0, 0.0)

        act_gps = (
            self.profile.activate_gps_compass
            if self.profile is not None
            else True
        )
        if act_gps:
            gps_channels = self.gps_sensor.compute_gps_channels(
                candidate_x, candidate_y, heading_rad,
                map_data, pathfinder, candidate_idx,
                prev_x, prev_y, prev_heading,
                stage_idx=stage_idx
            )
        else:
            gps_channels = (
                (0.0, 0.0, 0.0, 0.0) if use_binocular else (0.0, 0.0)
            )

        act_north = (
            self.profile.activate_north_compass
            if self.profile is not None
            else True
        )
        if act_north:
            north_channels = self.north_compass.compute_stereo_channels(
                heading_rad, self.profile
            )
        else:
            north_channels = (0.0, 0.0, 0.0, 0.0)

        act_cardinal = (
            self.profile.activate_cardinal_compass
            if self.profile is not None
            else True
        )
        if act_cardinal:
            c_n, c_e, c_s, c_w = (
                self.cardinal_compass.compute_cardinal_channels(
                    heading_rad, self.profile
                )
            )
        else:
            c_n, c_e, c_s, c_w = 0.0, 0.0, 0.0, 0.0

        clamped_spd: float = max(0.0, min(1.0, float(speed_ratio)))
        clamped_hp: float = max(0.0, min(1.0, float(health_ratio)))
        val_dmg_c: float = 1.0 if is_collided else 0.0
        val_dmg_i: float = 1.0 if is_idle else 0.0

        spin_dmg_rate: float = (
            self.profile.health_spin_dmg_per_frame
            if self.profile is not None else 0.0
        )
        if spin_dmg_rate <= 0.0:
            val_dmg_s: float = 0.0
        else:
            val_dmg_s = max(0.0, min(1.0, float(rot_ratio)))

        val_heal: float = 1.0 if is_healing else 0.0

        state_list: List[float] = [
            clamped_spd,
            clamped_hp,
            val_dmg_c,
            val_dmg_i,
            val_dmg_s,
            val_heal
        ]
        state_list.extend(gps_channels)
        state_list.extend([c_n, c_e, c_s, c_w])
        state_list.extend(north_channels)
        state_list.extend(exit_channels)

        state_features: NDArray[np.float32] = np.array(
            state_list, dtype=np.float32
        )

        return np.concatenate([wall_channels, state_features])
