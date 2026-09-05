"""
Live winner evaluation runner executing multi-champion swarm navigation on fresh mazes.
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
from entities.entity_state import AgentState
from neural.network import NeuralNetwork
from neural.brain_persistence import BrainPersistence, SavedBrainMetadata
from bridges.candidate_step_pipeline import CandidateStepPipeline
from evolution.fitness import FitnessEvaluator


class LiveWinnerDummyRecorder:
    """
    Recorder capturing step telemetry for all live champion instances.
    """

    def __init__(self, max_steps: int = 2000, num_champions: int = 15) -> None:
        self.max_steps: int = max_steps
        self.num_champions: int = num_champions
        self.buffer: NDArray[np.float32] = np.zeros(
            (max_steps, num_champions, 8), dtype=np.float32
        )
        self.recorded_steps: int = 0

    def resize(self, max_steps: int, num_champions: int) -> None:
        self.max_steps = max_steps
        self.num_champions = num_champions
        self.buffer = np.zeros((max_steps, num_champions, 8), dtype=np.float32)
        self.recorded_steps = 0

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
        if 0 <= step_idx < self.max_steps and 0 <= cand_idx < self.num_champions:
            self.buffer[step_idx, cand_idx, 0] = x
            self.buffer[step_idx, cand_idx, 1] = y
            self.buffer[step_idx, cand_idx, 2] = heading
            self.buffer[step_idx, cand_idx, 3] = health
            self.buffer[step_idx, cand_idx, 4] = dist
            self.buffer[step_idx, cand_idx, 5] = 1.0 if hit_wall else 0.0
            self.buffer[step_idx, cand_idx, 6] = 1.0 if is_alive else 0.0
            self.buffer[step_idx, cand_idx, 7] = 1.0 if reached_exit else 0.0
            if step_idx + 1 > self.recorded_steps:
                self.recorded_steps = step_idx + 1


class LiveWinnerRunner:
    """
    Coordinates multi-champion swarm simulation runs on fresh procedural mazes.
    """

    def __init__(
        self,
        active_profile_name: str = config.ACTIVE_AGENT_PROFILE,
        active_map_name: str = config.ACTIVE_MAP_PROFILE,
    ) -> None:
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

        self.num_champions: int = getattr(config, "LIVE_CHAMPION_COUNT", 15)
        self.transformers = [self.factory.create_transformer() for _ in range(self.num_champions)]
        self.kinematics = self.factory.create_kinematics()
        self.network: NeuralNetwork = self.factory.create_network()
        self.persistence: BrainPersistence = BrainPersistence()

        self.pipelines = [
            CandidateStepPipeline(t, self.kinematics) for t in self.transformers
        ]

        self.map_data: Optional[MapData] = None
        self.pathfinder: Optional[BFSPathfinder] = None
        self.states: List[AgentState] = []
        self.recorder: LiveWinnerDummyRecorder = LiveWinnerDummyRecorder(
            max_steps=config.LIVE_RUNNER_MAX_STEPS, num_champions=self.num_champions
        )
        self.total_run_steps: int = 0
        self.initial_bfs_dist: int = 9999
        self.active_brain_index: int = 0

        self._init_active_brain_selection()
        self.load_winner_brain(verbose=True)
        self.generate_fresh_maze()

    @property
    def active_brain_title(self) -> str:
        discovered = self.persistence.discover_saved_brains()
        if discovered and 0 <= self.active_brain_index < len(discovered):
            return discovered[self.active_brain_index].clean_title
        return self.active_profile_name

    def set_champion_count(self, count: int) -> None:
        count = max(1, min(100, count))
        if count == self.num_champions:
            return
        self.num_champions = count
        while len(self.transformers) < self.num_champions:
            t = self.factory.create_transformer()
            self.transformers.append(t)
            self.pipelines.append(CandidateStepPipeline(t, self.kinematics))
        self.recorder.resize(config.LIVE_RUNNER_MAX_STEPS, self.num_champions)
        self.generate_fresh_maze()

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
        target_meta = discovered[self.active_brain_index]

        if target_meta.clean_title in self.registry._profiles:
            self.active_profile_name = target_meta.clean_title
            self.profile = self.registry.get_profile(self.active_profile_name)

        self.factory = AgentFactory(self.registry, self.active_profile_name)
        self.transformers = [self.factory.create_transformer() for _ in range(self.num_champions)]
        self.kinematics = self.factory.create_kinematics()
        self.network = self.factory.create_network()
        self.pipelines = [
            CandidateStepPipeline(t, self.kinematics) for t in self.transformers
        ]

        loaded_ok = self.persistence.load_brain(
            self.active_profile_name,
            self.network,
            self.profile,
            context="live multi-champion swarm",
            verbose=False,
        )
        self.generate_fresh_maze()
        return loaded_ok

    def load_winner_brain(self, verbose: bool = False) -> bool:
        return self.persistence.load_brain(
            self.active_profile_name,
            self.network,
            self.profile,
            context="live multi-champion swarm",
            verbose=verbose,
        )

    def generate_fresh_maze(self) -> None:
        map_data = self.map_generator.generate_solvable_map()
        pathfinder = BFSPathfinder(map_data)
        pathfinder.compute_distance_matrix()

        sx, sy = map_data.start_pos
        initial_dist = pathfinder.get_step_distance(sx, sy)

        self.map_data = map_data
        self.pathfinder = pathfinder
        self.initial_bfs_dist = initial_dist

        self.states = [
            AgentState(float(sx) + 0.5, float(sy) + 0.5)
            for _ in range(self.num_champions)
        ]

        for idx, (st, tf) in enumerate(zip(self.states, self.transformers)):
            tf.reset_candidate_history(idx)
            base_heading = tf.generate_random_heading(map_data, map_data.start_pos)
            # Add subtle angular divergence so clones explore alternate branches
            angle_jitter = (float(idx) - (self.num_champions / 2.0)) * 0.08
            st.heading = (base_heading + angle_jitter) % (2.0 * math.pi)
            st.best_step_dist = initial_dist

        max_limit = config.LIVE_RUNNER_MAX_STEPS
        self.recorder.resize(max_steps=max_limit, num_champions=self.num_champions)
        self._run_upfront_simulation()

    def _run_upfront_simulation(self) -> None:
        if not self.map_data or not self.pathfinder or not self.states:
            return

        max_steps = config.LIVE_RUNNER_MAX_STEPS
        for step in range(max_steps):
            active_count = 0
            for idx in range(self.num_champions):
                st = self.states[idx]
                if not st.is_alive:
                    continue
                active_count += 1
                self.pipelines[idx].execute_step(
                    step_idx=step,
                    state=st,
                    net=self.network,
                    map_data=self.map_data,
                    pathfinder=self.pathfinder,
                    recorder=self.recorder,
                    candidate_idx=idx,
                    target_hold_frames=15,
                )

            if active_count == 0:
                break

        self.total_run_steps = max(1, self.recorder.recorded_steps)

    def get_activations_for_step(self, step_idx: int) -> List[List[float]]:
        if not self.map_data or not self.pathfinder or not self.recorder:
            return self.network.export_live_activations()

        max_f = self.total_run_steps
        safe_step = max(0, min(step_idx, max_f - 1))
        row = self.recorder.buffer[safe_step, 0]

        cx, cy, head, hp = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        hit = bool(row[5] > 0.5)

        prev_step = max(0, safe_step - 1)
        prev_row = self.recorder.buffer[prev_step, 0]
        dx, dy = cx - float(prev_row[0]), cy - float(prev_row[1])
        disp = (dx * dx + dy * dy) ** 0.5
        max_sp = max(1e-4, self.profile.move_speed)
        spd_r = min(1.0, max(0.0, disp / max_sp))

        is_idle = spd_r < self.profile.idle_damage_speed_threshold
        is_heal = spd_r >= self.profile.heal_speed_threshold and bool(row[6] > 0.5)

        features = self.transformers[0].compile_feature_vector(
            cx, cy, head, spd_r, hp, self.map_data, self.pathfinder,
            candidate_idx=0, is_collided=hit, is_idle=is_idle, is_healing=is_heal
        )
        self.network.forward(features)
        return self.network.export_live_activations()

    def to_gen_data_adapter(self) -> Dict[str, Any]:
        if not self.map_data or not self.states:
            print("[Error] Live runner not initialized.")
            sys.exit(1)

        raw_scores = [
            FitnessEvaluator.calculate_raw_score(st, self.initial_bfs_dist, max_steps=self.total_run_steps)
            for st in self.states
        ]
        winner_idx = int(np.argmax(raw_scores)) if raw_scores else 0
        flat_w = self.network.export_flat_weights()
        recorded_cnt = max(1, self.recorder.recorded_steps)
        trimmed_telemetry = self.recorder.buffer[:recorded_cnt].copy()

        return {
            "generation": -1,
            "bitmask_chunks": self.map_data.bitmask_chunks,
            "start_pos": self.map_data.start_pos,
            "exit_pos": self.map_data.exit_pos,
            "target_sequence": getattr(self.map_data, "target_sequence", [self.map_data.exit_pos]),
            "map_width": self.map_data.width,
            "map_height": self.map_data.height,
            "telemetry": trimmed_telemetry,
            "raw_scores": raw_scores,
            "normalized_scores": [1.0] * self.num_champions,
            "winner_index": winner_idx,
            "pop_weights": np.array([flat_w] * self.num_champions, dtype=np.float16),
        }
