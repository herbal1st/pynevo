"""
Live winner evaluation runner executing on the exact GPU simulation batch engine.
"""

import math
import sys
from typing import Dict, Any, Optional, List
import numpy as np

import config
from core.map_data import MapData
from core.pathfinder import BFSPathfinder
from core.map_generation.generator import MapGenerator
from entities.agent_profile_registry import (
    AgentProfileRegistry,
    ResolvedAgentProfile,
)
from entities.map_profile_registry import (
    MapProfileRegistry,
    ResolvedMapProfile,
)
from entities.agent_factory import AgentFactory
from entities.entity_state import AgentState
from neural.network import NeuralNetwork
from neural.brain_persistence import BrainPersistence, SavedBrainMetadata
from core.batch_engine import UniversalBatchRunner
from evolution.fitness import FitnessEvaluator


class LiveWinnerDummyRecorder:
    def __init__(self, max_steps: int = 3000) -> None:
        self.max_steps: int = max_steps
        self.buffer = np.zeros((max_steps, 1, 8), dtype=np.float32)
        self.recorded_steps: int = 0


class LiveWinnerRunner:
    """
    Coordinates live winner execution using the exact GPU rollout kernel.
    """

    def __init__(
        self,
        active_profile_name: str = config.ACTIVE_AGENT_PROFILE,
        active_map_name: str = config.ACTIVE_MAP_PROFILE,
    ) -> None:
        self.active_profile_name: str = active_profile_name
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(active_profile_name)

        self.map_registry: MapProfileRegistry = MapProfileRegistry()
        self.map_profile: ResolvedMapProfile = self.map_registry.get_profile(active_map_name)

        self.factory: AgentFactory = AgentFactory(self.registry, active_profile_name)
        self.map_generator: MapGenerator = MapGenerator(map_profile=self.map_profile)
        self.network: NeuralNetwork = self.factory.create_network()
        self.persistence: BrainPersistence = BrainPersistence()

        self.map_data: Optional[MapData] = None
        self.pathfinder: Optional[BFSPathfinder] = None
        self.state: Optional[AgentState] = None
        self.recorder: Optional[LiveWinnerDummyRecorder] = None
        self.total_run_steps: int = 0
        self.initial_bfs_dist: int = 9999
        self.active_brain_index: int = 0

        self.max_steps: int = getattr(config, "LIVE_RUNNER_MAX_STEPS", 2400)
        self.batch_runner: UniversalBatchRunner = UniversalBatchRunner(
            pop_size=1,
            max_steps=self.max_steps,
            num_rays=self.profile.vision_rays,
            hidden_dim=self.profile.neurons,
            memory_frames=self.profile.memory_frames
        )

        self._init_active_brain_selection()
        self.load_winner_brain(verbose=True)
        self.generate_fresh_maze()

    @property
    def active_brain_title(self) -> str:
        discovered = self.persistence.discover_saved_brains()
        if discovered and 0 <= self.active_brain_index < len(discovered):
            return discovered[self.active_brain_index].clean_title
        return self.active_profile_name

    def _init_active_brain_selection(self) -> None:
        discovered = self.persistence.discover_saved_brains()
        if not discovered:
            return

        for idx, meta in enumerate(discovered):
            if meta.clean_title == self.active_profile_name:
                self.active_brain_index = idx
                return

        self.active_brain_index = 0

    def cycle_brain(self, delta: int) -> bool:
        discovered = self.persistence.discover_saved_brains(force_refresh=True)
        if not discovered:
            return False

        self.active_brain_index = (self.active_brain_index + delta) % len(discovered)
        target_meta: SavedBrainMetadata = discovered[self.active_brain_index]

        if target_meta.clean_title in self.registry._profiles:
            self.active_profile_name = target_meta.clean_title
            self.profile = self.registry.get_profile(self.active_profile_name)

        self.factory = AgentFactory(self.registry, self.active_profile_name)
        self.network = self.factory.create_network()
        self.batch_runner = UniversalBatchRunner(
            pop_size=1,
            max_steps=self.max_steps,
            num_rays=self.profile.vision_rays,
            hidden_dim=self.profile.neurons,
            memory_frames=self.profile.memory_frames
        )

        loaded_ok = self.persistence.load_brain(
            self.active_profile_name,
            self.network,
            self.profile,
            context="live runner hot-swap",
            verbose=False,
        )

        self.generate_fresh_maze()
        return loaded_ok

    def load_winner_brain(self, verbose: bool = False) -> bool:
        return self.persistence.load_brain(
            self.active_profile_name,
            self.network,
            self.profile,
            context="live runner",
            verbose=verbose,
        )

    def generate_fresh_maze(self) -> None:
        map_data = self.map_generator.generate_solvable_map()
        pathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        sx, sy = map_data.start_pos
        initial_dist: int = pathfinder.get_step_distance(sx, sy)

        self.map_data = map_data
        self.pathfinder = pathfinder
        self.initial_bfs_dist = initial_dist

        self.state = AgentState(float(sx) + 0.5, float(sy) + 0.5)
        self.recorder = LiveWinnerDummyRecorder(max_steps=self.max_steps)
        self._run_upfront_simulation()

    def _run_upfront_simulation(self) -> None:
        if self.map_data is None or self.pathfinder is None:
            return

        flat_weights = self.network.export_flat_weights().astype(np.float32)
        pop_weights_single = flat_weights[np.newaxis, :]

        raw_scores, summary_matrix, telemetry_matrix = self.batch_runner.run_generation_from_tensor(
            self.map_data,
            self.pathfinder,
            pop_weights_single,
            profile=self.profile,
            move_speed=self.profile.move_speed
        )

        # Store telemetry in recorder
        self.recorder.buffer[:, 0, :] = telemetry_matrix[:, 0, :]
        exit_indices = np.where(telemetry_matrix[:, 0, 7] > 0.5)[0]
        if len(exit_indices) > 0:
            self.total_run_steps = int(exit_indices[0]) + 1
        else:
            alive_indices = np.where(telemetry_matrix[:, 0, 6] > 0.5)[0]
            self.total_run_steps = int(alive_indices[-1]) + 1 if len(alive_indices) > 0 else self.max_steps

        self.recorder.recorded_steps = self.total_run_steps

    def get_activations_for_step(self, step_idx: int) -> List[List[float]]:
        return self.network.export_live_activations()

    def to_gen_data_adapter(self) -> Dict[str, Any]:
        if self.map_data is None or self.recorder is None:
            print("[Error] Live runner not initialized.")
            sys.exit(1)

        flat_w = self.network.export_flat_weights()
        recorded_cnt: int = max(1, self.total_run_steps)
        trimmed_telemetry = self.recorder.buffer[:recorded_cnt].copy()

        return {
            "generation": -1,
            "bitmask_chunks": self.map_data.bitmask_chunks,
            "start_pos": self.map_data.start_pos,
            "exit_pos": self.map_data.exit_pos,
            "map_width": self.map_data.width,
            "map_height": self.map_data.height,
            "telemetry": trimmed_telemetry,
            "raw_scores": [1000.0],
            "normalized_scores": [1.0],
            "winner_index": 0,
            "pop_weights": np.array([flat_w], dtype=np.float16),
        }
