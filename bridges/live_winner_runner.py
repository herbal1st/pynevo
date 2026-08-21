"""
Live winner evaluation runner executing pre-calculated maze navigation.
"""

import math
import sys
from typing import Dict, Any, Optional, List
import numpy as np
from numpy.typing import NDArray

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
from entities.player_state import PlayerState
from neural.network import NeuralNetwork
from neural.brain_persistence import BrainPersistence, SavedBrainMetadata
from bridges.candidate_step_pipeline import CandidateStepPipeline
from evolution.fitness import FitnessEvaluator


class LiveWinnerDummyRecorder:
    """
    Dummy recorder capturing step data for live candidate telemetry.
    """

    def __init__(self, max_steps: int = 2000) -> None:
        """
        Pre-allocates single-candidate telemetry buffer array.
        """
        self.max_steps: int = max_steps
        self.buffer: NDArray[np.float32] = np.zeros(
            (max_steps, 1, 8), dtype=np.float32
        )
        self.recorded_steps: int = 0

    def record_step_data(
        self,
        step_idx: int,
        cand_idx: int,
        x: float,
        y: float,
        heading: float,
        health: float,
        dist: float,
        hit_wall: bool,
        is_alive: bool,
        reached_exit: bool,
    ) -> None:
        """
        Writes step data into single-candidate live telemetry buffer.
        """
        if 0 <= step_idx < self.max_steps:
            self.buffer[step_idx, 0, 0] = x
            self.buffer[step_idx, 0, 1] = y
            self.buffer[step_idx, 0, 2] = heading
            self.buffer[step_idx, 0, 3] = health
            self.buffer[step_idx, 0, 4] = dist
            self.buffer[step_idx, 0, 5] = 1.0 if hit_wall else 0.0
            self.buffer[step_idx, 0, 6] = 1.0 if is_alive else 0.0
            self.buffer[step_idx, 0, 7] = 1.0 if reached_exit else 0.0
            self.recorded_steps = max(self.recorded_steps, step_idx + 1)


class LiveWinnerRunner:
    """
    Coordinates pre-calculated live winner execution on fresh mazes.
    """

    def __init__(
        self,
        active_profile_name: str = config.ACTIVE_AGENT_PROFILE,
        active_map_name: str = config.ACTIVE_MAP_PROFILE,
    ) -> None:
        """
        Initializes registries, persistence discovery, network, and pipeline.
        """
        self.active_profile_name: str = active_profile_name
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.profile: ResolvedAgentProfile = self.registry.get_profile(
            active_profile_name
        )

        self.map_registry: MapProfileRegistry = MapProfileRegistry()
        self.map_profile: ResolvedMapProfile = (
            self.map_registry.get_profile(active_map_name)
        )

        self.factory: AgentFactory = AgentFactory(
            self.registry, active_profile_name
        )

        self.map_generator: MapGenerator = MapGenerator(
            map_profile=self.map_profile
        )
        self.transformer = self.factory.create_transformer()
        self.kinematics = self.factory.create_kinematics()
        self.network: NeuralNetwork = self.factory.create_network()
        self.persistence: BrainPersistence = BrainPersistence()

        self.pipeline: CandidateStepPipeline = CandidateStepPipeline(
            self.transformer, self.kinematics
        )

        self.map_data: Optional[MapData] = None
        self.pathfinder: Optional[BFSPathfinder] = None
        self.state: Optional[PlayerState] = None
        self.recorder: Optional[LiveWinnerDummyRecorder] = None
        self.total_run_steps: int = 0
        self.initial_bfs_dist: int = 9999
        self.is_paused: bool = False

        self.active_brain_index: int = 0
        self._init_active_brain_selection()
        self.load_winner_brain(verbose=True)
        self.generate_fresh_maze()

    @property
    def active_brain_title(self) -> str:
        """
        Returns clean display title of the currently selected brain archive.
        """
        discovered = self.persistence.discover_saved_brains()
        if discovered and 0 <= self.active_brain_index < len(discovered):
            return discovered[self.active_brain_index].clean_title
        return self.active_profile_name

    def _init_active_brain_selection(self) -> None:
        """
        Matches ACTIVE_AGENT_PROFILE with explicit CLI notifications.
        """
        discovered = self.persistence.discover_saved_brains()
        if not discovered:
            print(
                f"[LiveRunner] No saved brain archives found in "
                f"saved_brains/. Using default network for "
                f"'{self.active_profile_name}'."
            )
            return

        for idx, meta in enumerate(discovered):
            if meta.clean_title == self.active_profile_name:
                self.active_brain_index = idx
                print(
                    f"[LiveRunner] Active profile '{self.active_profile_name}' "
                    f"selected from saved_brains/ (Index {idx})."
                )
                return

        self.active_brain_index = 0
        selected_title: str = discovered[0].clean_title
        print(
            f"[LiveRunner] Active profile '{self.active_profile_name}' "
            f"not found in saved_brains/. Selecting available archive "
            f"'{selected_title}' (Index 0)."
        )

    def cycle_brain(self, delta: int) -> bool:
        """
        Cycles to next/previous brain archive with dynamic topology swap.
        """
        discovered = self.persistence.discover_saved_brains(
            force_refresh=True
        )
        if not discovered:
            print("[LiveRunner] Cannot cycle brain: saved_brains/ is empty.")
            return False

        self.active_brain_index = (
            self.active_brain_index + delta
        ) % len(discovered)
        target_meta: SavedBrainMetadata = discovered[self.active_brain_index]

        if target_meta.clean_title in self.registry._profiles:
            self.active_profile_name = target_meta.clean_title
            self.profile = self.registry.get_profile(self.active_profile_name)

        self.factory = AgentFactory(self.registry, self.active_profile_name)
        self.transformer = self.factory.create_transformer()
        self.kinematics = self.factory.create_kinematics()
        self.network = self.factory.create_network()
        self.pipeline = CandidateStepPipeline(
            self.transformer, self.kinematics
        )

        loaded_ok: bool = self.persistence.load_brain(
            self.active_profile_name,
            self.network,
            self.profile,
            context="live runner hot-swap",
            verbose=False,
        )

        self.generate_fresh_maze()
        return loaded_ok

    def load_winner_brain(self, verbose: bool = False) -> bool:
        """
        Loads trained winner weights from disk into local network.
        """
        return self.persistence.load_brain(
            self.active_profile_name,
            self.network,
            self.profile,
            context="live runner",
            verbose=verbose,
        )

    def generate_fresh_maze(self) -> None:
        """
        Generates new maze and executes upfront simulation run.
        """
        map_data = self.map_generator.generate_solvable_map()
        pathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        sx, sy = map_data.start_pos
        initial_dist: int = pathfinder.get_step_distance(sx, sy)

        self.map_data = map_data
        self.pathfinder = pathfinder
        self.initial_bfs_dist = initial_dist

        self.state = PlayerState(float(sx) + 0.5, float(sy) + 0.5)
        self.transformer.reset_candidate_history(0)
        self.state.heading = self.transformer.generate_random_heading(
            map_data, map_data.start_pos
        )
        self.state.best_step_dist = initial_dist

        max_limit: int = config.LIVE_RUNNER_MAX_STEPS
        self.recorder = LiveWinnerDummyRecorder(max_steps=max_limit)
        self._run_upfront_simulation()

    def _run_upfront_simulation(self) -> None:
        """
        Executes complete headless run upfront until solve, death, or limit.
        """
        if (
            self.map_data is None
            or self.pathfinder is None
            or self.state is None
            or self.recorder is None
        ):
            return

        max_steps: int = config.LIVE_RUNNER_MAX_STEPS
        for step in range(max_steps):
            can_continue: bool = self.pipeline.execute_step(
                step_idx=step,
                state=self.state,
                net=self.network,
                map_data=self.map_data,
                pathfinder=self.pathfinder,
                recorder=self.recorder,
                candidate_idx=0,
            )
            if not can_continue:
                break

        self.total_run_steps = max(1, self.recorder.recorded_steps)

    def get_activations_for_step(self, step_idx: int) -> List[List[float]]:
        """
        Computes network activations for candidate at specified step index.
        """
        if (
            self.map_data is None
            or self.pathfinder is None
            or self.recorder is None
        ):
            return self.network.export_live_activations()

        max_f: int = self.total_run_steps
        safe_step: int = max(0, min(step_idx, max_f - 1))
        row = self.recorder.buffer[safe_step, 0]

        cx: float = float(row[0])
        cy: float = float(row[1])
        head: float = float(row[2])
        hp: float = float(row[3])
        hit: bool = bool(row[5] > 0.5)

        prev_step: int = max(0, safe_step - 1)
        prev_row = self.recorder.buffer[prev_step, 0]
        prev_x: float = float(prev_row[0])
        prev_y: float = float(prev_row[1])

        dx: float = cx - prev_x
        dy: float = cy - prev_y
        disp: float = math.sqrt((dx * dx) + (dy * dy))
        max_sp: float = max(1e-4, self.profile.move_speed)
        spd_r: float = max(0.0, min(1.0, disp / max_sp))

        idle_t: float = self.profile.idle_damage_speed_threshold
        heal_t: float = self.profile.heal_speed_threshold
        is_idle: bool = spd_r < idle_t
        is_heal: bool = spd_r >= heal_t and bool(row[6] > 0.5)

        features = self.transformer.compile_feature_vector(
            cx,
            cy,
            head,
            spd_r,
            hp,
            self.map_data,
            self.pathfinder,
            candidate_idx=0,
            is_collided=hit,
            is_idle=is_idle,
            is_healing=is_heal,
        )
        self.network.forward(features)
        return self.network.export_live_activations()

    def to_gen_data_adapter(self) -> Dict[str, Any]:
        """
        Formats live state into standard gen_data dictionary for viewports.
        """
        if (
            self.map_data is None
            or self.state is None
            or self.recorder is None
        ):
            print(
                "[Error] Live runner not initialized for gen_data adapter."
            )
            sys.exit(1)

        raw_score: float = FitnessEvaluator.calculate_raw_score(
            self.state,
            self.initial_bfs_dist,
            max_steps=config.LIVE_RUNNER_MAX_STEPS,
            move_speed=self.profile.move_speed,
        )

        flat_w = self.network.export_flat_weights()
        recorded_cnt: int = max(1, self.recorder.recorded_steps)
        trimmed_telemetry = self.recorder.buffer[:recorded_cnt].copy()

        return {
            "generation": -1,
            "bitmask_chunks": self.map_data.bitmask_chunks,
            "start_pos": self.map_data.start_pos,
            "exit_pos": self.map_data.exit_pos,
            "map_width": self.map_data.width,
            "map_height": self.map_data.height,
            "telemetry": trimmed_telemetry,
            "raw_scores": [raw_score],
            "normalized_scores": [1.0],
            "winner_index": 0,
            "pop_weights": np.array([flat_w], dtype=np.float16),
        }
