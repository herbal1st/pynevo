"""
Headless neuroevolution simulation trainer running candidate runs.
"""

import time
from typing import List, Optional
import numpy as np

import config
from core.map_generation.generator import MapGenerator
from core.pathfinder import BFSPathfinder
from entities.agent_profile_registry import AgentProfileRegistry
from entities.training_profile_registry import (
    TrainingProfileRegistry,
    ResolvedTrainingProfile
)
from entities.map_profile_registry import (
    MapProfileRegistry,
    ResolvedMapProfile
)
from entities.agent_factory import AgentFactory
from entities.entity_state import AgentState
from evolution.fitness import FitnessEvaluator
from evolution.population import PopulationManager
from evolution.recorder import FrameRecorder
from bridges.candidate_step_pipeline import CandidateStepPipeline
from bridges.cli_presenter import CLIPresenter
from neural.brain_persistence import BrainPersistence


class HeadlessTrainer:
    """
    Coordinates headless training loops and frame timeline recording.
    """

    def __init__(
        self,
        active_profile_name: str = config.ACTIVE_AGENT_PROFILE,
        active_training_name: str = config.ACTIVE_TRAINING_PROFILE,
        active_map_name: str = config.ACTIVE_MAP_PROFILE
    ) -> None:
        """
        Initializes trainer with factory, persistence loader, & pipelines.
        """
        self.active_profile_name: str = active_profile_name
        self.registry: AgentProfileRegistry = AgentProfileRegistry()
        self.factory: AgentFactory = AgentFactory(
            self.registry, active_profile_name
        )

        self.training_registry: TrainingProfileRegistry = (
            TrainingProfileRegistry()
        )
        self.training_profile: ResolvedTrainingProfile = (
            self.training_registry.get_profile(active_training_name)
        )

        self.map_registry: MapProfileRegistry = MapProfileRegistry()
        self.map_profile: ResolvedMapProfile = (
            self.map_registry.get_profile(active_map_name)
        )

        self.pop_size: int = self.training_profile.population_size
        self.max_steps: int = self.training_profile.max_simulation_steps
        self.map_generator: MapGenerator = MapGenerator(
            map_profile=self.map_profile
        )
        self.transformer = self.factory.create_transformer()
        self.kinematics = self.factory.create_kinematics()
        self.population: PopulationManager = PopulationManager(
            factory=self.factory,
            pop_size=self.pop_size,
            mutation_rate=self.training_profile.mutation_rate,
            mutation_scale=self.training_profile.mutation_scale,
            elitism_ratio=self.training_profile.elitism_ratio
        )
        self.recorder: FrameRecorder = FrameRecorder()
        self.persistence: BrainPersistence = BrainPersistence()

        seed_net = self.factory.create_network()
        if self.persistence.load_brain(
            active_profile_name, seed_net, self.factory.profile,
            context="training"
        ):
            self.population.seed_population_from_brain(seed_net)

        self.step_pipeline: CandidateStepPipeline = CandidateStepPipeline(
            self.transformer, self.kinematics
        )
        self.cli_presenter: CLIPresenter = CLIPresenter(
            self.pop_size, self.max_steps
        )

    def run_training_session(
        self,
        num_generations: Optional[int] = None
    ) -> FrameRecorder:
        """
        Runs headless candidate simulations over multiple generations.
        """
        gens_count: int = (
            num_generations if num_generations is not None
            else self.training_profile.learning_generations
        )

        param_cnt: int = self.population.networks[0].param_count
        self.recorder.allocate_session_buffers(
            self.max_steps, self.pop_size, gens_count, param_cnt
        )

        self.cli_presenter.print_start_banner(
            agent_profile=self.active_profile_name,
            training_profile=self.training_profile.profile_name,
            map_profile=self.map_profile.profile_name,
            hold_frames=self.training_profile.target_hold_frames
        )
        for w_str in AgentProfileRegistry.get_clamped_warning_strings():
            print(w_str)

        min_diff_ratio: float = (
            self.training_profile.min_path_difficulty_ratio
        )
        max_diff_ratio: float = (
            self.training_profile.max_path_difficulty_ratio
        )

        for gen_idx in range(gens_count):
            gen_start_time: float = time.perf_counter()

            map_data = self.map_generator.generate_solvable_map(
                min_difficulty_ratio=min_diff_ratio,
                max_difficulty_ratio=max_diff_ratio
            )
            map_data.target_sequence = [map_data.exit_pos]

            pathfinder = BFSPathfinder(map_data)
            pathfinder.clear_cache()
            pathfinder.compute_distance_matrix_for_target(
                map_data.exit_pos, stage_idx=0
            )

            start_x, start_y = map_data.start_pos
            initial_bfs_dist: int = pathfinder.get_step_distance(
                start_x, start_y, stage_idx=0
            )

            candidate_states: List[AgentState] = [
                AgentState(float(start_x) + 0.5, float(start_y) + 0.5)
                for _ in range(self.pop_size)
            ]

            for c_idx, state in enumerate(candidate_states):
                self.transformer.reset_candidate_history(c_idx)
                state.heading = self.transformer.generate_random_heading(
                    map_data, map_data.start_pos
                )
                state.best_step_dist = initial_bfs_dist

            actual_steps: int = 0
            for step in range(1, self.max_steps + 1):
                step_idx: int = step - 1
                active_count: int = 0

                for idx in range(self.pop_size):
                    state = candidate_states[idx]
                    net = self.population.networks[idx]

                    if not state.is_alive:
                        if (
                            step_idx > 0 and
                            self.recorder.telemetry_bundler is not None and
                            self.recorder.telemetry_bundler._curr_buffer is not None
                        ):
                            self.recorder.telemetry_bundler._curr_buffer[
                                step_idx, idx, :
                            ] = self.recorder.telemetry_bundler._curr_buffer[
                                step_idx - 1, idx, :
                            ]
                        continue

                    active_count += 1

                    if state.active_target_idx >= len(map_data.target_sequence):
                        next_pos = map_data.append_next_target(
                            seed_offset=step_idx
                        )
                        next_stage: int = len(map_data.target_sequence) - 1
                        pathfinder.compute_distance_matrix_for_target(
                            next_pos, stage_idx=next_stage
                        )

                    self.step_pipeline.execute_step(
                        step_idx,
                        state,
                        net,
                        map_data,
                        pathfinder,
                        self.recorder,
                        candidate_idx=idx,
                        target_hold_frames=(
                            self.training_profile.target_hold_frames
                        )
                    )

                actual_steps = step
                if active_count == 0:
                    break

            raw_scores: List[float] = [
                FitnessEvaluator.calculate_raw_score(
                    c_state,
                    initial_bfs_dist,
                    lost_hp_impact=(
                        self.training_profile.lost_hp_score_impact_ratio
                    )
                )
                for c_state in candidate_states
            ]

            norm_scores: List[float] = FitnessEvaluator.normalize_scores(
                raw_scores
            )

            self.recorder.finalize_generation(
                gen_idx,
                map_data,
                raw_scores,
                norm_scores,
                actual_steps,
                pop_networks=self.population.networks
            )

            elapsed_sec: float = time.perf_counter() - gen_start_time

            self.cli_presenter.print_generation_row(
                gen_idx,
                raw_scores,
                norm_scores,
                candidate_states,
                elapsed_sec
            )

            if gen_idx == gens_count - 1:
                winner_idx: int = int(np.argmax(norm_scores))
                winner_net = self.population.networks[winner_idx]
                self.persistence.save_brain(
                    self.active_profile_name,
                    winner_net,
                    self.factory.profile,
                    context="training"
                )

            self.population.evolve_next_generation(norm_scores)

        self.cli_presenter.print_finish_footer()
        for w_str in AgentProfileRegistry.get_clamped_warning_strings():
            print(w_str)

        self.recorder.save_temporary_disk_archive()
        return self.recorder
