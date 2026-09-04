"""
Compiles wall rays, state, compass, & GPS features with temporal memory.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from entities.agent_profile_registry import ResolvedAgentProfile
from perception.spawn_heading import SpawnHeadingGenerator
from perception.spatial.gps_sensor import TopologicalGPSSensor
from perception.spatial.memory_stacker import TemporalMemoryStacker
from perception.spatial.feature_compiler import SingleFrameFeatureCompiler


class SpatialTransformer:
    """
    Compiles sensory observations into stacked temporal feature vectors.
    """

    def __init__(
        self,
        profile: Optional[ResolvedAgentProfile] = None
    ) -> None:
        """
        Initializes perception sensors, GPS sensor, & temporal stacker.
        """
        self.profile: Optional[ResolvedAgentProfile] = profile
        self.gps_sensor: TopologicalGPSSensor = TopologicalGPSSensor(profile)
        self.compiler: SingleFrameFeatureCompiler = (
            SingleFrameFeatureCompiler(profile, self.gps_sensor)
        )
        self.memory_stacker: TemporalMemoryStacker = TemporalMemoryStacker()

    @property
    def last_gps_progress(self) -> Tuple[float, ...]:
        """
        Forwards last calculated GPS progress channels from GPS sensor.
        """
        return self.gps_sensor.last_gps_progress

    @property
    def sampler(self):
        """
        Forwarding property for backward compatibility with vision sampler.
        """
        return self.compiler.sampler

    @property
    def exit_compass(self):
        """
        Forwarding property for backward compatibility with exit compass.
        """
        return self.compiler.exit_compass

    @property
    def north_compass(self):
        """
        Forwarding property for backward compatibility with North compass.
        """
        return self.compiler.north_compass

    @property
    def cardinal_compass(self):
        """
        Forwarding property for backward compatibility with cardinal compass.
        """
        return self.compiler.cardinal_compass

    def reset_candidate_history(self, candidate_idx: int) -> None:
        """
        Clears temporal observation history & GPS distance for candidate.
        """
        self.memory_stacker.reset_candidate_history(candidate_idx)
        self.gps_sensor.reset_candidate_history(candidate_idx)

    def generate_random_heading(
        self,
        map_data: Optional[MapData] = None,
        start_pos: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Delegates spawn heading generation to SpawnHeadingGenerator.
        """
        use_bfs: bool = (
            self.profile.use_bfs_spawn_heading
            if self.profile is not None else True
        )
        return SpawnHeadingGenerator.generate_random_heading(
            map_data, start_pos, use_bfs_spawn_heading=use_bfs
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
        return self.compiler.compile_base_vector(
            candidate_x,
            candidate_y,
            heading_rad,
            speed_ratio,
            health_ratio,
            map_data,
            pathfinder,
            candidate_idx=candidate_idx,
            prev_x=prev_x,
            prev_y=prev_y,
            prev_heading=prev_heading,
            is_collided=is_collided,
            is_idle=is_idle,
            is_healing=is_healing,
            rot_ratio=rot_ratio,
            stage_idx=stage_idx
        )

    def compile_feature_vector(
        self,
        candidate_x: float,
        candidate_y: float,
        heading_rad: float,
        speed_ratio: float,
        health_ratio: float,
        map_data: MapData,
        pathfinder: BFSPathfinder,
        candidate_idx: int = 0,
        is_collided: bool = False,
        is_idle: bool = False,
        is_healing: bool = False,
        rot_ratio: float = 0.0,
        stage_idx: int = 0
    ) -> NDArray[np.float32]:
        """
        Samples active frame and stacks past memory_frames observations.
        """
        base_vector = self.compile_base_vector(
            candidate_x,
            candidate_y,
            heading_rad,
            speed_ratio,
            health_ratio,
            map_data,
            pathfinder,
            candidate_idx=candidate_idx,
            is_collided=is_collided,
            is_idle=is_idle,
            is_healing=is_healing,
            rot_ratio=rot_ratio,
            stage_idx=stage_idx
        )

        mem_k: int = (
            self.profile.memory_frames if self.profile is not None else 0
        )
        return self.memory_stacker.stack_base_vector(
            candidate_idx, base_vector, mem_k
        )
